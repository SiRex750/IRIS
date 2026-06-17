"""
IRIS ablation evaluation harness.

Runs all 4 ablation conditions against test videos
and outputs metrics for the paper.

Ablation table:
    Condition       | Codec gating | NLI verification
    Baseline        | None         | None
    Ablation 1      | KG only      | None
    Ablation 2      | None         | Uniform NLI
    Full IRIS       | Both jointly | Risk-proportional

Metrics: accuracy, compression_ratio, latency_ms

Owner: Track D
"""
from __future__ import annotations


ABLATION_CONDITIONS = ["baseline", "ablation_1", "ablation_2", "full_iris"]


def run_ablation(video_path: str, queries: list[str], condition: str) -> dict:
    """Run one ablation condition. Returns metrics dict."""
    # TODO: implement
    pass


def run_full_eval(video_path: str, queries: list[str]) -> dict:
    """Run all conditions and return comparative results table."""
    # TODO: implement
    pass
