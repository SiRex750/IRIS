"""Probe: demuxed frame count vs CSV frame count, shot counts, timing. READ-ONLY."""
import csv, sys, time
from pathlib import Path

REPO = Path(r"C:\Users\akash\Documents\Iris")
sys.path.insert(0, str(REPO))
import iris.charon_v as charon_v

GT_DIR = REPO.parent / "Iris-ucfvad" / "tuning" / "ucfcrime_vad_exp1" / "per_frame_ground_truth_scores"
VIDEO_ROOT = Path(r"C:\Users\akash\Downloads\Anomaly-Videos-Part-1")
VIDEOS = ["Abuse028","Abuse030","Arrest001","Arrest007","Arrest024","Arrest030","Arrest039",
          "Arson007","Arson009","Arson010","Arson011","Arson016","Arson018","Arson022",
          "Arson035","Arson041","Assault006","Assault010","Assault011"]

print(f"{'video':<12}{'N_csv':>8}{'N_demux':>9}{'delta':>7}{'I':>6}{'shots':>7}{'k@10.5':>8}{'sec':>7}")
tot = 0.0
bad = []
for vid in VIDEOS:
    n_csv = sum(1 for _ in csv.DictReader((GT_DIR / f"{vid}_x264.csv").open(newline="")))
    cls = "".join(ch for ch in vid if not ch.isdigit())
    mp4 = VIDEO_ROOT / cls / f"{vid}_x264.mp4"
    t0 = time.time()
    afe, ifr, _, _ = charon_v._demux_packet_curve(str(mp4))
    fps = charon_v.get_stream_fps(str(mp4))
    spans = charon_v.compute_valley_scene_boundaries(afe, ifr, fps)
    el = time.time() - t0
    tot += el
    n_dem = len(afe)
    k = max(1, int(0.105 * n_csv))
    d = n_dem - n_csv
    if d != 0:
        bad.append((vid, n_csv, n_dem))
    print(f"{vid:<12}{n_csv:>8}{n_dem:>9}{d:>7}{len(ifr):>6}{len(spans):>7}{k:>8}{el:>7.2f}")

print(f"\ntotal demux+span wall: {tot:.1f}s")
print(f"frame-count mismatches: {len(bad)} {bad}")
