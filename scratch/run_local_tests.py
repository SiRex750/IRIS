import time
import os
import sys
import json
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Add root directory to python path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from iris.ingest import ingest
from iris.query import query
from iris.iris_config import IRISConfig
import iris.aria as aria
from iris.aria import LlamaBackend

# Override backend to use our newly imported Qwen model in Ollama
aria.set_backend(LlamaBackend(text_model='qwen3.5:4b'))




VIDEOS = [
    {
        "path": "test_data/Virat/VIRAT_S_000205_03_000860_000922.mp4",
        "question": "Is there a person loading a vehicle?"
    },
    {
        "path": "test_data/Virat/VIRAT_S_000205_05_001092_001124.mp4",
        "question": "Is there a person walking near the car?"
    },
    {
        "path": "test_data/Virat/VIRAT_S_000206_09_001714_001851.mp4",
        "question": "Is there a person walking near the building?"
    },
    {
        "path": "test_data/Virat/VIRAT_S_000207_02_000498_000530.mp4",
        "question": "Is there a vehicle parking?"
    }
]

print("==================================================", flush=True)
print("STARTING LOCAL INTEGRATION & STRESS TEST RUN", flush=True)
print("==================================================", flush=True)

results = []

for i, item in enumerate(VIDEOS):
    video_path = item["path"]
    q = item["question"]
    print(f"\n[{i+1}/{len(VIDEOS)}] Video: {video_path}", flush=True)
    print(f"Question: '{q}'", flush=True)
    
    # 1. Ingest
    t0 = time.time()
    try:
        idx = ingest(video_path)
        ingest_time = time.time() - t0
        print(f"  Ingest: OK ({ingest_time:.2f}s, {idx.frames_processed} frames, {idx.peak_count} peaks)", flush=True)
    except Exception as e:
        print(f"  Ingest FAILED: {e}", flush=True)
        continue
        
    # 2. Legacy Query Mode
    print("  Running Query (Legacy Mode)...", flush=True)
    t0 = time.time()
    try:
        cfg_legacy = IRISConfig(cerberus_mode='legacy')
        res_legacy = query(q, idx, config=cfg_legacy)
        legacy_time = time.time() - t0
        legacy_ans = res_legacy.get("answer", "N/A")
        legacy_verified = res_legacy.get("verified", False)
        print(f"    Legacy Raw Answer: {res_legacy.get('raw_answer')}", flush=True)
        print(f"    Legacy Answer: {legacy_ans}", flush=True)
        print(f"    Legacy Verified: {legacy_verified} (in {legacy_time:.2f}s)", flush=True)
        print(f"    Legacy Verified Claims: {res_legacy.get('verified_claims')}", flush=True)
        print(f"    Legacy Rejected Claims: {res_legacy.get('rejected_claims')}", flush=True)
        print(f"    Legacy Unverifiable Claims: {res_legacy.get('unverifiable_claims')}", flush=True)
    except Exception as e:
        print(f"    Legacy Query FAILED: {e}", flush=True)
        res_legacy = {}
        legacy_ans, legacy_verified, legacy_time = "ERROR", False, 0.0
        
    # Print retrieved frame indices and their captions
    print("    Retrieved Frames & Captions:", flush=True)
    frame_map = {fr.frame_idx: fr for fr in idx.frames}
    retrieved_idxs = res_legacy.get("retrieved_frame_idxs", []) if res_legacy else []
    for fidx in retrieved_idxs:
        fr = frame_map.get(fidx)
        caption = getattr(fr, "caption", None)
        print(f"      Frame {fidx}: {caption}", flush=True)

    # 3. Cerberus v2 Mode
    active_model = str(getattr(aria.get_backend(), "text_model", "Unknown"))
    print(f"  Running Query (v2 Mode with {active_model})...", flush=True)
    t0 = time.time()
    try:
        cfg_v2 = IRISConfig(cerberus_mode='v2')
        res_v2 = query(q, idx, config=cfg_v2)
        v2_time = time.time() - t0
        v2_badge = res_v2.get("badge", "N/A")
        v2_claims = res_v2.get("answer_claims", None)
        v2_attempts = res_v2.get("n_llm_attempts", 0)
        v2_failures = res_v2.get("compliance_failure_labels", [])
        
        print(f"    v2 Raw Answer: {res_v2.get('raw_answer')}", flush=True)
        print(f"    v2 Badge: {v2_badge}", flush=True)
        if v2_claims:
            print(f"    v2 Generated Claims:", flush=True)
            for c in v2_claims.claims:
                print(f"      - {c}", flush=True)
        else:
            print(f"    v2 Generated Claims: None", flush=True)
        print(f"    v2 Claim Verdicts: {res_v2.get('claim_verdicts')}", flush=True)
        print(f"    v2 Core Claim Verdict: {res_v2.get('core_claim_verdict')}", flush=True)
        print(f"    v2 Attempts: {v2_attempts} (Failures: {v2_failures}) (in {v2_time:.2f}s)", flush=True)
    except Exception as e:
        print(f"    v2 Query FAILED: {e}", flush=True)
        res_v2 = {}
        v2_badge, v2_claims, v2_attempts, v2_failures, v2_time = "ERROR", None, 0, [], 0.0

    results.append({
        "video": video_path,
        "question": q,
        "legacy_ans": legacy_ans,
        "legacy_verified": legacy_verified,
        "legacy_time": legacy_time,
        "v2_badge": v2_badge,
        "v2_claims_count": len(v2_claims.claims) if v2_claims else 0,
        "v2_attempts": v2_attempts,
        "v2_failures": v2_failures,
        "v2_time": v2_time
    })

# Format results as a markdown table
markdown_content = """# Local Verification Run Results
Executed on all 4 VIRAT dataset clips.

| Video | Question | Legacy Ans | Legacy Verified | Legacy Time | v2 Badge | v2 Claims | v2 Attempts (Failures) | v2 Time |
|---|---|---|---|---|---|---|---|---|
"""

for r in results:
    video_name = Path(r["video"]).name
    failures_str = ",".join(r["v2_failures"]) if r["v2_failures"] else "None"
    markdown_content += f"| {video_name} | {r['question']} | {r['legacy_ans'][:50]}... | {r['legacy_verified']} | {r['legacy_time']:.2f}s | {r['v2_badge']} | {r['v2_claims_count']} | {r['v2_attempts']} ({failures_str}) | {r['v2_time']:.2f}s |\n"

# Save the markdown report to scratch
model_name = str(getattr(aria.get_backend(), "text_model", "unknown")).replace(":", "_").replace("/", "_")
timestamp = time.strftime("%Y%m%d_%H%M%S")
report_path = Path(f"scratch/local_test_results_{model_name}_{timestamp}.md")
report_path.write_text(markdown_content, encoding='utf-8')



print("\n==================================================", flush=True)
print(f"TESTS COMPLETED. Report saved to: {report_path}", flush=True)
print("==================================================", flush=True)
print(markdown_content, flush=True)
