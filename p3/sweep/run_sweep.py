"""Paper 3 full Sintel sweep, as fixed by p3/PREREGISTRATION.md. Outputs under p3/sweep/.

Reuses p3/pilot_v2/run_pilot_v2.py unchanged: run_one (encode, MV extraction, metrics), the arm
constructors and knob sets (X, XB, N, NB, M), and the flip references. Changed vs v2, and only this:
  - arm list: the pre-registered arms (v2's x264_crf23_bf2_pb1 dropped; x264 QP 24 pair no longer "extra")
  - sequence list: all 23 training sequences
  - pass: FINAL (all arms), then CLEAN (x264 CRF series, NVENC QP series, the two matched-QP B pairs).
    v1's load_seq hard-codes "final", so load_seq below is that function with the pass as an argument.
  - loop order: sequence-outer (23 sequences do not fit in memory at once); veryslow runs after every
    other FINAL arm on every sequence; CLEAN runs last. Flip references run first within each sequence.
  - results.csv gets a leading "pass" column; encodes/ and blocks/ are split by pass.
  - every encode is checked (slices, x264 SEI, decoded QP where the arm fixes it); mismatches are logged.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import av
import numpy as np
import pandas as pd

P3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.environ.get("SWEEP_OUT", os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(P3)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "pilot_v2"))
sys.path.insert(0, os.path.join(P3, "pilot"))

import run_pilot_v2 as v2  # noqa: E402
from run_pilot_v2 import _e, X, XB, N, NB, M, X264_REF, OWN_REF  # noqa: E402
from run_pilot import SINTEL, C, y_plane  # noqa: E402
from lib.flo import SintelSeq  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402
from lib.mvs import extract_mvs  # noqa: E402
from lib.encode import sei_value  # noqa: E402

PREREG_COMMIT = "0bf4cacfbf0b1d25f0da6bb1326fc3028ce2afd1"
CRFS = (12, 18, 23, 28, 33, 38, 45)
NQPS = (18, 23, 28, 33, 38, 45)

CRF_SERIES = [_e(f"x264_crf{c}", "libx264", ["crf"] + (["ref", "preset", "keyint", "bframes", "merange", "encoder"]
                                                      if c == 23 else []), crf=c, **X) for c in CRFS]
X264_VARIANTS = (
    [_e("x264_crf23_ref2", "libx264", ["ref"], d_known=False, **{**X, "crf": 23, "ref": 2}),
     _e("x264_crf23_ref3", "libx264", ["ref"], d_known=False, **{**X, "crf": 23, "ref": 3}),
     _e("x264_crf23_keyint30", "libx264", ["keyint"], **{**X, "crf": 23, "keyint": 30}),
     _e("x264_crf23_ultrafast", "libx264", ["preset"], **{**X, "crf": 23, "preset": "ultrafast"})]
    + [_e(f"x264_crf23_umh{r}", "libx264", ["merange"], crf=23, me="umh", merange=r, **X) for r in (16, 32, 64)]
    + [_e("x264_crf23_bf2", "libx264", ["bframes"], crf=23, **XB)])
X264_QP_PAIR = [_e("x264_qp24", "libx264", ["bframes_cqp"], qp=24, **X),
                _e("x264_qp24_bf2_pb1", "libx264", ["bframes_cqp"], qp=24, pbratio=1.0, **XB)]
NVENC_SERIES = [_e(f"nvenc_qp{q}", "h264_nvenc", ["nvenc_qp"] + (["bframes", "encoder"] if q == 28 else []),
                   qp=q, **N) for q in NQPS]
NVENC_B = [_e("nvenc_qp28_bf2", "h264_nvenc", ["bframes"], qp=28, **NB),
           _e("nvenc_qp28_bf2_bqeq", "h264_nvenc", ["bframes"], qp=28, b_qfactor=1.0, b_qoffset=0, **NB)]
MPEG4 = [_e(f"mpeg4_q{q}", "mpeg4", ["mpeg4_q"] + (["encoder"] if q == 4 else []), q=q, **M) for q in (2, 4, 8, 16, 31)]
VERYSLOW = [_e("x264_crf23_veryslow", "libx264", ["preset"], **{**X, "crf": 23, "preset": "veryslow"})]

FINAL_MAIN = CRF_SERIES + X264_VARIANTS + X264_QP_PAIR + NVENC_SERIES + NVENC_B + MPEG4
CLEAN_ARMS = CRF_SERIES + NVENC_SERIES + X264_QP_PAIR + [NVENC_B[1]]
STAGES = [("final", "main", FINAL_MAIN), ("final", "veryslow", VERYSLOW), ("clean", "main", CLEAN_ARMS)]

# decoded-QP requirements: {encode_id: {frame type: required QP}} (every MB of every such frame)
QP_REQ = {"x264_qp24": {"P": 24}, "x264_qp24_bf2_pb1": {"P": 24, "B": 24},
          "nvenc_qp28_bf2_bqeq": {"P": 28, "B": 28}}
QP_REQ.update({f"nvenc_qp{q}": {"P": q} for q in NQPS})


def load_seq(name, pass_):
    """run_pilot.load_seq with the pass as an argument (GT is pass-independent)."""
    s = SintelSeq(SINTEL, name, pass_)
    imgs = [s.image(i) for i in range(s.n_frames)]
    H, W = imgs[0].shape[:2]
    gt = {}
    for t in range(1, s.n_frames):
        flow, bad = s.flow_into(t), s.bad_mask_into(t)
        g = GTGrid(flow, bad)
        g._build_sat()
        cgx, cgy, ccov, _ = g.cell_gt()
        gt[t] = {"g": g, "flow": flow, "bad": bad, "cgx": cgx, "cgy": cgy, "ccov": ccov}
    return {"name": name, "imgs": imgs, "yref": [y_plane(im, H) for im in imgs], "gt": gt,
            "H": H, "W": W, "Hc": -(-H // C), "Wc": -(-W // C)}


def check_encode(enc, path, lg):
    """Return a list of mismatch strings for one encode (empty = all checks pass)."""
    bad = []
    if lg["slices_min"] != 1 or lg["slices_max"] != 1:
        bad.append(f"slices per frame {lg['slices_min']}-{lg['slices_max']} (want 1)")
    kn = enc["knobs"]
    if enc["codec"] == "libx264":
        sei = lg["x264_sei"]
        want = {"threads": "1", "sliced_threads": "0", "scenecut": "0", "keyint": str(kn["keyint"]),
                "ref": str(kn["ref"]), "bframes": str(kn["bframes"])}
        if kn.get("bframes"):
            want.update(b_adapt=str(kn["b_adapt"]), b_pyramid="0")
        for k, v in want.items():
            got = sei_value(sei, k)
            if got != v:
                bad.append(f"SEI {k}={got} (want {v})")
        if sei_value(sei, "slices") is not None:
            bad.append(f"SEI slices={sei_value(sei, 'slices')} (want absent)")
    if kn.get("bframes", 0) == 0 and lg["n_B"]:
        bad.append(f"{lg['n_B']} B-frames with bframes 0")
    qps = {}
    frames, _ = extract_mvs(path, want_qp=True)
    for f in frames:
        if f["qp"] is not None:
            qps.setdefault(f["pict_type"], set()).update(np.unique(f["qp"]).tolist())
    lg["decoded_qp_values"] = {k: sorted(int(x) for x in v) for k, v in qps.items()}
    for ft, q in QP_REQ.get(enc["id"], {}).items():
        got = qps.get(ft)
        if got != {q}:
            bad.append(f"decoded {ft} QP values {sorted(got) if got else got} (want {{{q}}})")
    return bad


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def code_hash():
    files = [os.path.abspath(__file__), os.path.join(P3, "pilot_v2", "run_pilot_v2.py"),
             os.path.join(P3, "pilot", "run_pilot.py"), os.path.join(P3, "validate", "val_D_gt_grid.py")]
    files += sorted(glob.glob(os.path.join(P3, "lib", "*.py")))
    h = hashlib.sha256()
    per = {}
    for f in files:
        b = open(f, "rb").read()
        h.update(b)
        per[os.path.relpath(f, REPO).replace("\\", "/")] = hashlib.sha256(b).hexdigest()
    return h.hexdigest(), per


def main():
    T0 = time.perf_counter()
    sys.path.insert(0, os.path.join(P3, "pilot"))
    from step0 import x264_build
    seqs = sorted(os.listdir(os.path.join(SINTEL, "training", "final")))
    assert len(seqs) == 23, seqs
    if os.environ.get("SWEEP_SMOKE"):  # smoke test only: one short sequence, outputs go to SWEEP_OUT
        seqs = [os.environ["SWEEP_SMOKE"]]
    share = pd.read_csv(os.path.join(P3, "pilot", "static_scan.csv")).set_index("sequence")["static_share"]
    ch, per = code_hash()
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    man = {"start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_sha256": ch, "code_files_sha256": per,
           "git_hash": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
           "prereg_commit": PREREG_COMMIT, "code_frozen_commit": "d8a1d385e5265d025c7b6a39a0bd9b53219ef99c",
           "uncommitted_lib_changes": git("status", "--porcelain", "--", "p3/lib", "p3/pilot_v2/run_pilot_v2.py",
                                          "p3/pilot/run_pilot.py"),
           "python": sys.version.split()[0], "pyav": av.__version__,
           "ffmpeg_libs": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
           "x264_build": x264_build(), "nvidia": drv, "numpy": np.__version__, "pandas": pd.__version__,
           "sequences": seqs, "static_share": {s: float(share[s]) for s in seqs},
           "stages": [{"pass": p, "stage": st, "arms": [e["id"] for e in arms]} for p, st, arms in STAGES],
           "gt": "lib/gt_grid.py forward splat onto frame t grid, 4x4 cells, gt_cover >= 0.5"}
    man_p = os.path.join(HERE, "manifest.json")
    json.dump(man, open(man_p, "w"), indent=1)

    ref_ids = set(OWN_REF.values()) | {X264_REF}
    results, logs, failures, mismatches = [], [], [], []
    refs_by_pass = {}
    for pass_, stage, arms in STAGES:
        v2.ENC_DIR = os.path.join(HERE, "encodes", pass_)
        v2.BLK_DIR = os.path.join(HERE, "blocks", pass_)
        os.makedirs(v2.ENC_DIR, exist_ok=True)
        os.makedirs(v2.BLK_DIR, exist_ok=True)
        refs = refs_by_pass.setdefault(pass_, {rid: {} for rid in ref_ids})
        for s in seqs:
            tl = time.perf_counter()
            try:
                sq = load_seq(s, pass_)
            except Exception as e:
                for enc in arms:
                    failures.append({"pass": pass_, "encode_id": enc["id"], "sequence": s,
                                     "error": "load_seq: " + repr(e)})
                print("FAILED load", pass_, s, repr(e), flush=True)
                continue
            print(f"[{pass_}/{stage}] loaded {s} ({len(sq['imgs'])} frames) in {time.perf_counter() - tl:.0f}s",
                  flush=True)
            for enc in arms:
                lg = {"pass": pass_, "stage": stage, "encode_id": enc["id"], "sequence": s, "codec": enc["codec"],
                      "knobs": enc["knobs"]}
                try:
                    out, dec = v2.run_one(enc, sq, refs, lg)
                    if enc["id"] in ref_ids:
                        for (sq_, t, tau, key), v in dec.items():
                            if key in ("d1", "d1(assumed)"):
                                refs[enc["id"]][(sq_, t, tau)] = v
                    results += [{"pass": pass_, **r} for r in out]
                    lg["status"] = "ok"
                    try:
                        bad = check_encode(enc, os.path.join(v2.ENC_DIR, f"{enc['id']}__{s}.mp4"), lg)
                    except Exception as e:
                        bad = [f"check raised {e!r}"]
                    lg["check_mismatches"] = bad
                    for b in bad:
                        mismatches.append({"pass": pass_, "encode_id": enc["id"], "sequence": s, "mismatch": b})
                    print(f"{pass_:5s} {enc['id']:22s} {s:10s} {lg['bitrate_kbps']:8.0f} kbps PSNR {lg['psnr_y']:5.2f} "
                          f"sl {lg['slices_min']}-{lg['slices_max']} QP I/P/B {lg['mean_i_qp']:.1f}/"
                          f"{lg['mean_p_qp']:.1f}/{lg['mean_b_qp']:.1f} {lg['run_s']:.1f}s"
                          + (f"  MISMATCH {bad}" if bad else ""), flush=True)
                except Exception as e:
                    lg.update(status="FAILED", error=repr(e), traceback=traceback.format_exc())
                    failures.append({"pass": pass_, "encode_id": enc["id"], "sequence": s, "error": repr(e)})
                    print("FAILED", pass_, enc["id"], s, repr(e), flush=True)
                logs.append(lg)
            del sq
            pd.DataFrame(results).to_csv(os.path.join(HERE, "results.csv"), index=False)
            json.dump({"logs": logs, "failures": failures, "mismatches": mismatches},
                      open(os.path.join(HERE, "encode_log.json"), "w"), indent=1, default=str)
        if pass_ == "final" and stage == "veryslow":
            refs_by_pass.pop("final")
    total = time.perf_counter() - T0
    man.update({"end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime_s": total,
                "n_runs": len(logs), "n_failures": len(failures), "n_mismatches": len(mismatches)})
    json.dump(man, open(man_p, "w"), indent=1)
    print(f"done in {total:.0f}s, runs {len(logs)}, failures {len(failures)}, mismatches {len(mismatches)}")


if __name__ == "__main__":
    main()
