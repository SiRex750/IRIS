"""S1 regression test: every call site of predicted_span_from_frames_peak that
has a per-question duration available must pass duration_s, so spans are
clamped to end-of-video (eval/metrics.py::predicted_span_from_frames_peak
clamps hi = min(duration_s, hi) only when duration_s is not None). Before the
S1 fix, scripts/val_confirm_e2e_eval.py, scripts/blind_ablation_eval.py,
scripts/query_reformulation_v2_ablation.py and scripts/part3_tune.py called it
without duration_s even though the question dicts each of them build
(load_val_confirm_questions / load_val_tune_questions) carry q["duration"].
This test greps the actual call-site source so it fails on the pre-fix code
and passes on the post-fix code -- it is not exercising eval/metrics.py's own
clamp logic, which is already covered by tests/test_span_methods.py's
test_method_d_clamps_to_duration.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (file, whether duration_s is genuinely available at that call site)
CALL_SITES_EXPECTING_DURATION = [
    REPO / "scripts" / "val_confirm_e2e_eval.py",
    REPO / "scripts" / "blind_ablation_eval.py",
    REPO / "scripts" / "query_reformulation_v2_ablation.py",
]

CALL_RE = re.compile(r"predicted_span_from_frames_peak\(([^)]*(?:\([^)]*\)[^)]*)*)\)", re.DOTALL)


def _calls_in(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    return CALL_RE.findall(source)


def test_val_confirm_e2e_eval_passes_duration_s():
    calls = _calls_in(REPO / "scripts" / "val_confirm_e2e_eval.py")
    assert calls, "no predicted_span_from_frames_peak call found"
    assert any("duration_s=" in c for c in calls), (
        "scripts/val_confirm_e2e_eval.py calls predicted_span_from_frames_peak "
        "without duration_s -- spans can extend past end-of-video"
    )


def test_blind_ablation_eval_passes_duration_s():
    calls = _calls_in(REPO / "scripts" / "blind_ablation_eval.py")
    assert calls, "no predicted_span_from_frames_peak call found"
    assert any("duration_s=" in c for c in calls), (
        "scripts/blind_ablation_eval.py calls predicted_span_from_frames_peak "
        "without duration_s -- spans can extend past end-of-video"
    )


def test_query_reformulation_v2_ablation_passes_duration_s():
    calls = _calls_in(REPO / "scripts" / "query_reformulation_v2_ablation.py")
    assert calls, "no predicted_span_from_frames_peak call found"
    assert any("duration_s=" in c for c in calls), (
        "scripts/query_reformulation_v2_ablation.py calls "
        "predicted_span_from_frames_peak without duration_s -- spans can "
        "extend past end-of-video"
    )


def test_part3_tune_default_predicted_span_forwards_duration_s():
    source = (REPO / "scripts" / "part3_tune.py").read_text(encoding="utf-8")
    m = re.search(
        r"def default_predicted_span\(.*?\n\):|def default_predicted_span\((.*?)\).*?\n(?:.*\n)*?\s*return span",
        source,
    )
    assert "def default_predicted_span(" in source
    calls = _calls_in(REPO / "scripts" / "part3_tune.py")
    assert calls, "no predicted_span_from_frames_peak call found in part3_tune.py"
    assert any("duration_s=" in c for c in calls), (
        "scripts/part3_tune.py::default_predicted_span does not forward "
        "duration_s to predicted_span_from_frames_peak"
    )
    # and the caller must actually supply a duration, not just accept the param
    assert "default_predicted_span(retrieved_frames, query_embedding, duration_s=" in source, (
        "part3_tune.py's default_predicted_span call site does not pass duration_s through"
    )
