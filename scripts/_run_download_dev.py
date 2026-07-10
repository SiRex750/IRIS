import subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
log = REPO / "download_dev_output.log"
script_code = r"""
import json, os, sys
from huggingface_hub import hf_hub_download

REPO = "VLM2Vec/nextqa-rawvideo"
DEST = "eval/data/nextqa/NExTVideo_flat"
os.makedirs(DEST, exist_ok=True)

ids = sorted({json.loads(l)["video"] for l in open("eval/data/nextqa/dev_100.jsonl")})
print(f"{len(ids)} dev videos to fetch", flush=True)

ok, missing = [], []
for vid in ids:
    try:
        p = hf_hub_download(repo_id=REPO, repo_type="dataset",
                            filename=f"{vid}.mp4", local_dir=DEST)
        ok.append(vid)
        print(f"  OK  {vid}", flush=True)
    except Exception as e:
        missing.append((vid, str(e)[:80]))
        print(f"  ERR {vid}  {str(e)[:80]}", flush=True)

print(f"\nfetched {len(ok)}, missing {len(missing)}", flush=True)
for v, e in missing:
    print(f"  MISSING {v} {e}", flush=True)
"""
with open(log, "w", encoding="utf-8") as f:
    subprocess.run(
        [sys.executable, "-u", "-c", script_code],
        cwd=str(REPO),
        stdout=f, stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
print(f"done -> {log}")
