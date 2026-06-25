"""
Unit tests for Cerberus-V NLI truth gate.

Owner: Track D
"""
from __future__ import annotations

from legacy.triple import KnowledgeTriple
from legacy.cache import L1Cache
from iris.cerberus_v import CerberusV
from iris.iris_config import IRISConfig


def test_cerberus_gating_and_ner_verification() -> None:
    config = IRISConfig()
    cache = L1Cache()
    cerberus = CerberusV()

    t = KnowledgeTriple(subject="Maverick", verb="flies", object="F-18")
    cache.add_fact(t, pagerank_score=0.9, source_residual=0.8)

    assert cerberus.get_verification_mode(0.8, config) == "full_nli"
    assert cerberus.get_verification_mode(0.5, config) == "filtered_nli"
    assert cerberus.get_verification_mode(0.2, config) == "ner_only"

    result = cerberus.verify(
        claims=["Maverick pilots a jet"],
        cache=cache,
        action_score=0.2,
        config=config,
    )
    assert result["mode"] == "ner_only"
    assert len(result["verified"]) + len(result["unverifiable"]) == 1
    print("Sanity assertions passed.")


def test_cerberus_full_nli_verification() -> None:
    config = IRISConfig()
    cache = L1Cache()
    cerberus = CerberusV()

    t = KnowledgeTriple(subject="Maverick", verb="flies", object="F-18")
    cache.add_fact(t, pagerank_score=0.9, source_residual=0.8)

    # Entailing claim
    result_ent = cerberus.verify(
        claims=["Maverick flies a jet"],
        cache=cache,
        action_score=0.8,
        config=config,
    )
    assert result_ent["mode"] == "full_nli"
    assert "Maverick flies a jet" in result_ent["verified"]

    # Contradicting claim
    result_contra = cerberus.verify(
        claims=["Maverick does not fly a jet"],
        cache=cache,
        action_score=0.8,
        config=config,
    )
    assert "Maverick does not fly a jet" in result_contra["rejected"]

