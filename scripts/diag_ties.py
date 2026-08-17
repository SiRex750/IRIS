"""POST-HOC diagnostic: the action score is a hold-forward step function, so
'highest-action-in-shot' degenerates to 'earliest frame of the top plateau',
clustering picks into adjacent runs. Quantify across all 19."""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\akash\Documents\Iris\scripts")
import r0_gate_full as r0
from t5_shotbucket_gate import shot_spans, shot_bucket_select


def clustering(sel):
    """Fraction of selected frames that sit adjacent to another selected frame,
    and the mean length of consecutive runs."""
    sel = np.sort(sel)
    gaps = np.diff(sel)
    adj = int((gaps == 1).sum())
    runs, cur = [], 1
    for g in gaps:
        if g == 1:
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return 100.0 * (2 * adj) / max(len(sel), 1), float(np.mean(runs)), len(runs)


print(f"{'video':<12}{'N':>7}{'uniq':>7}{'plateau':>9}   "
      f"{'F adj%':>8}{'F run':>7}{'F blocks':>9}   {'C adj%':>8}{'C run':>7}   {'E adj%':>8}{'E run':>7}")
rows = []
for vid in r0.VIDEOS:
    d = r0.load_video(vid)
    spans, _ = shot_spans(vid)
    n, score = d["n_frames"], d["score"]
    k = r0.budget_k(n, 10.5)
    uniq = len(np.unique(score))
    F = shot_bucket_select(spans, score, k, 1, "action", "peak_action")
    C = r0.arm_c_uniform(n, 10.5)
    E = r0.top_k_by_score(score, k)
    fa, fr, fb = clustering(F)
    ca, cr, _ = clustering(C)
    ea, er, _ = clustering(E)
    rows.append((fa, fr, fb, ca, ea, er, n / uniq))
    print(f"{vid:<12}{n:>7}{uniq:>7}{n/uniq:>9.1f}   {fa:>8.1f}{fr:>7.2f}{fb:>9}   "
          f"{ca:>8.1f}{cr:>7.2f}   {ea:>8.1f}{er:>7.2f}")

a = np.array(rows)
print(f"\nmean plateau length (frames per distinct score) : {a[:,6].mean():.1f}")
print(f"mean adjacency%  F (shot 1/shot action) : {a[:,0].mean():.1f}%   mean run {a[:,1].mean():.2f}")
print(f"mean adjacency%  C (uniform)            : {a[:,3].mean():.1f}%")
print(f"mean adjacency%  E (action top-k)       : {a[:,4].mean():.1f}%   mean run {a[:,5].mean():.2f}")
print(f"\nF: mean distinct blocks {a[:,2].mean():.0f} vs mean budget "
      f"{np.mean([r0.budget_k(r0.load_video(v)['n_frames'],10.5) for v in r0.VIDEOS]):.0f} frames")
