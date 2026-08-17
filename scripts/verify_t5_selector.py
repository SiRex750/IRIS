"""Invariant tests for the T5 shot-bucketed selector."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r"C:\Users\akash\Documents\Iris\scripts")
from t5_shotbucket_gate import shot_bucket_select, apportion, SHOT_ARMS

rng = np.random.default_rng(7)
fails = 0


def check(cond, msg):
    global fails
    if not cond:
        fails += 1
        print("FAIL:", msg)


# ---- apportion: places exactly `extra` when capacity allows, never exceeds caps
for _ in range(2000):
    S = int(rng.integers(1, 40))
    caps = rng.integers(0, 12, size=S)
    extra = int(rng.integers(0, int(caps.sum()) + 1))
    w = rng.random(S) * rng.choice([0.0, 1.0, 100.0])
    a = apportion(w, extra, caps)
    check(a.sum() == extra, f"apportion sum {a.sum()} != {extra}")
    check((a <= caps).all(), "apportion exceeded caps")
    check((a >= 0).all(), "apportion negative")

# extra beyond total capacity -> fills capacity, no hang
a = apportion(np.ones(3), 100, np.array([2, 2, 2]))
check(a.sum() == 6, "over-capacity apportion should saturate")

# proportionality sanity: weight 9:1, 10 seats, ample caps -> 9/1
a = apportion(np.array([9.0, 1.0]), 10, np.array([100, 100]))
check(tuple(a) == (9, 1), f"proportionality broken: {a}")

# zero weights -> deterministic index order, still exact
a = apportion(np.zeros(5), 3, np.full(5, 10))
check(a.sum() == 3, "zero-weight apportion not exact")


# ---- selector: exactly k, in [0,N), for random span sets incl. the S>k branch
def rand_spans(n, s):
    cuts = sorted(rng.choice(np.arange(1, n), size=min(s - 1, n - 1), replace=False).tolist())
    edges = [0] + cuts + [n]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


for trial in range(3000):
    n = int(rng.integers(5, 800))
    S = int(rng.integers(1, min(n, 120)))
    spans = rand_spans(n, S)
    score = rng.random(n) * rng.choice([1.0, 1000.0])
    if rng.random() < 0.25:
        score[:] = 0.0                      # degenerate all-equal scores
    k = int(rng.integers(1, n + 1))         # includes k < S (the S>k branch) and k == n
    for key, per_shot, pick, weight in SHOT_ARMS:
        sel = shot_bucket_select(spans, score, k, per_shot, pick, weight)
        check(len(sel) == k, f"{key}: len {len(sel)} != k={k} (n={n},S={len(spans)})")
        check(len(np.unique(sel)) == len(sel), f"{key}: duplicates")
        check(sel.min() >= 0 and sel.max() < n, f"{key}: out of range")

# ---- coverage floor: when S <= k every shot gets >=1 frame (1/shot arms)
for trial in range(500):
    n = int(rng.integers(50, 900))
    S = int(rng.integers(1, 40))
    spans = rand_spans(n, S)
    k = int(rng.integers(len(spans), n + 1))
    for key, per_shot, pick, weight in (("F", 1, "action", "peak_action"),
                                        ("H", 1, "mid", "shot_len")):
        sel = set(shot_bucket_select(spans, rng.random(n), k, per_shot, pick, weight).tolist())
        for (s, e) in spans:
            check(any(i in sel for i in range(s, e)), f"{key}: shot ({s},{e}) empty despite S<=k")

# ---- mid variant with 1 frame/shot == pre-registered midpoint, and uses no score
spans = [(0, 10), (10, 25), (25, 30)]
sel = shot_bucket_select(spans, np.zeros(30), 3, 1, "mid", "shot_len")
check(list(sel) == [(0 + 10) // 2, (10 + 25) // 2, (25 + 30) // 2], f"midpoint rule broken: {sel}")
s1 = shot_bucket_select(spans, rng.random(30), 12, 1, "mid", "shot_len")
s2 = shot_bucket_select(spans, rng.random(30), 12, 1, "mid", "shot_len")
check(np.array_equal(s1, s2), "mid arm is not score-independent")

# ---- S>k: exactly k shots represented, chosen by peak action
spans = [(0, 10), (10, 20), (20, 30), (30, 40)]
score = np.zeros(40)
score[35] = 9.0   # shot 3 hottest
score[5] = 8.0    # shot 0 next
sel = shot_bucket_select(spans, score, 2, 1, "action", "peak_action")
check(sorted(sel.tolist()) == [5, 35], f"S>k ranking broken: {sel}")

print("FAILURES:", fails)
