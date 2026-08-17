"""Why did F_shot1_action admit zero gold frames on Abuse028? Bug or genuine?"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\akash\Documents\Iris\scripts")
import r0_gate_full as r0
from t5_shotbucket_gate import shot_spans, shot_bucket_select

VID = "Abuse028"
d = r0.load_video(VID)
spans, meta = shot_spans(VID)
n, gt, score = d["n_frames"], d["gt"], d["score"]
k = r0.budget_k(n, 10.5)

print(f"{VID}: N={n} k={k} shots={len(spans)} gold windows={d['windows']} "
      f"gold frames={int(gt.sum())}")
print(f"score range [{score.min():.4f}, {score.max():.4f}]  unique={len(np.unique(score))}")

sel = shot_bucket_select(spans, score, k, 1, "action", "peak_action")
print(f"selected {len(sel)}, gold hits = {int(gt[sel].sum())}")

gs, ge = d["windows"][0]
print(f"\ngold window = [{gs}, {ge}]  (len {ge-gs+1})")
print("\nshots overlapping the gold window:")
for i, (s, e) in enumerate(spans):
    if e > gs and s <= ge:
        inshot = sel[(sel >= s) & (sel < e)]
        n_gold_in_shot = int(gt[s:e].sum())
        print(f"  shot {i:>3} [{s:>5},{e:>5}) len={e-s:>4} peak_score={score[s:e].max():.4f} "
              f"gold_frames_in_shot={n_gold_in_shot:>4} alloc={len(inshot):>3} "
              f"picked={sorted(inshot.tolist())[:12]}")

print("\nper-shot allocation vs peak score (all shots):")
for i, (s, e) in enumerate(spans):
    inshot = sel[(sel >= s) & (sel < e)]
    print(f"  shot {i:>3} [{s:>5},{e:>5}) len={e-s:>4} peak={score[s:e].max():>9.4f} "
          f"alloc={len(inshot):>4} gold_in_shot={int(gt[s:e].sum()):>4}")

# Where does the score put its mass relative to gold?
print(f"\nmean score inside gold  = {score[gs:ge+1].mean():.4f}")
mask = np.ones(n, bool); mask[gs:ge+1] = False
print(f"mean score outside gold = {score[mask].mean():.4f}")
top = r0.top_k_by_score(score, k)
print(f"action top-k gold hits  = {int(gt[top].sum())} / {k}")
uni = r0.arm_c_uniform(n, 10.5)
print(f"uniform gold hits       = {int(gt[uni].sum())} / {len(uni)}")
