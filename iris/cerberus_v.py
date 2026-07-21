"""
Cerberus-V — NLI truth gate for IRIS.

Port of HADES Cerberus adapted for video context.
Verifies claims in L1 Elysium SCRATCH set against
facts grounded in PEAK and SALIENT frame data.

Backend: DeBERTa-v3-base
"""
from __future__ import annotations

import re
import json
import psutil
from typing import Any, Iterable

from iris.iris_config import IRISConfig, ConfigManager
from iris.triple import KnowledgeTriple


_SPACY_NLP = None
_NLI_TOKENIZER = None
_NLI_MODEL = None


class CerberusV:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        self.model_name = model_name
        self._tokenizer: Any = None
        self._model: Any = None
        self._nlp: Any = None

    def _get_spacy(self) -> Any:
        global _SPACY_NLP
        if _SPACY_NLP is None:
            import spacy
            try:
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                _SPACY_NLP = spacy.load("en_core_web_sm")
        return _SPACY_NLP

    def _get_nli_model(self) -> tuple[Any, Any]:
        global _NLI_TOKENIZER, _NLI_MODEL
        if _NLI_MODEL is None:
            import sys
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            _NLI_TOKENIZER = AutoTokenizer.from_pretrained(self.model_name)
            _NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            _NLI_MODEL.eval()
            # Force CPU on Windows to prevent DeBERTa build_relative_position CUDA access violations
            device = torch.device("cpu" if sys.platform == "win32" else ("cuda" if torch.cuda.is_available() else "cpu"))
            _NLI_MODEL.to(device)
        return _NLI_TOKENIZER, _NLI_MODEL

    def get_verification_mode(
        self,
        action_score: float,
        config: IRISConfig,
    ) -> str:
        """
        Returns verification intensity based on continuous action_score.
        action_score is a float in [0, 1] from action_score.py.
        Both thresholds are GEPA-tunable via IRISConfig.

        full_nli:     full DeBERTa forward pass on all claims
        filtered_nli: DeBERTa only on claims where confidence >= 0.6
        ner_only:     named entity overlap check, no DeBERTa forward pass
        """
        if getattr(config, "disable_nli", False):
            return "ner_only"
        if action_score >= config.cerberus_high_thresh:
            return "full_nli"
        elif action_score >= config.cerberus_low_thresh:
            return "filtered_nli"
        else:
            return "ner_only"

    def _parse_claims(self, claims: list[str]) -> list[str]:
        parsed_claims = []
        for claim in claims:
            # Clean think and tool blocks
            claim_clean = re.sub(r'<think>.*?</think>', '', claim, flags=re.DOTALL)
            claim_clean = re.sub(r'\{"tool":\s*".*?"\}', '', claim_clean)
            claim_clean = re.sub(r'\{"tool":\s*".*?",.*?\}', '', claim_clean, flags=re.DOTALL)
            
            claims_match = re.search(r'CLAIMS\s*:\s*(\[.*?\])', claim_clean, re.DOTALL | re.IGNORECASE)
            raw = None
            if claims_match:
                raw = claims_match.group(1)
            elif claim_clean.strip().startswith('[') and claim_clean.strip().endswith(']'):
                raw = claim_clean.strip()
                
            if raw:
                raw_claims = None
                try:
                    raw_claims = json.loads(raw)
                except Exception:
                    # Robust fallback: repaired trailing commas
                    repaired = re.sub(r',\s*([\]}])', r'\1', raw)
                    try:
                        raw_claims = json.loads(repaired)
                    except Exception:
                        repaired = repaired.replace("'", '"')
                        try:
                            raw_claims = json.loads(repaired)
                        except Exception:
                            pass
                
                if isinstance(raw_claims, list):
                    for c in raw_claims:
                        if isinstance(c, dict) and "s" in c and "v" in c and "o" in c:
                            parsed_claims.append(f"{c['s']} {c['v']} {c['o']}")
                        elif isinstance(c, str):
                            parsed_claims.append(c)
                    continue
            
            parsed_claims.append(claim)
            
        return parsed_claims

    def verify(
        self,
        claims: list[str],
        cache: object,
        action_score: float,
        config: IRISConfig,
    ) -> dict:
        """
        Verify ARIA's generated claims against facts in L1 Elysium.

        claims:       list of string claims from ARIA's output
        cache:        populated L1Cache instance
        action_score: continuous float [0,1] from action_score.py
        config:       IRISConfig instance for threshold access

        Returns:
        {
            "verified":     list[str],   # claims that passed
            "rejected":     list[str],   # claims contradicted by cache facts
            "unverifiable": list[str],   # claims with no relevant evidence
            "mode":         str,         # which verification mode was used
            "action_score": float        # echo back for logging
        }
        """
        import time
        t_load_start = time.time()
        self._get_spacy()
        mode = self.get_verification_mode(action_score, config)
        if mode in ("full_nli", "filtered_nli"):
            self._get_nli_model()
        t_load = time.time() - t_load_start

        t_inf_start = time.time()
        parsed_claims = self._parse_claims(claims)
        facts = [entry.text for entry in cache.set_facts.values()]

        if mode == "full_nli":
            res = self._full_nli(parsed_claims, facts, mode, action_score)
        elif mode == "filtered_nli":
            high_conf = [c for c in parsed_claims if self._confidence(c) >= 0.6]
            low_conf  = [c for c in parsed_claims if self._confidence(c) <  0.6]
            res = self._full_nli(high_conf, facts, mode, action_score)
            res["verified"] += low_conf   # low-conf claims pass unverified
        else:  # ner_only
            res = self._ner_overlap(parsed_claims, facts, mode, action_score)
        t_inf = time.time() - t_inf_start

        print(f"[DEBUG] CerberusV verification: model_load_time = {t_load:.4f}s | inference_time = {t_inf:.4f}s")
        return res

    def _confidence(self, claim: str) -> float:
        """
        Lightweight verification confidence scoring.
        Avoids DeBERTa pass to preserve budget on low-complexity or hedged statements.
        """
        if not claim or not claim.strip():
            return 0.0
        
        # Hedging words reduce confidence
        hedges = {"maybe", "probably", "possibly", "could", "might", "perhaps", "approximate", "approximately"}
        words = set(claim.lower().split())
        if words & hedges:
            return 0.4
            
        # Very short claims
        if len(words) < 3:
            return 0.3
            
        # Entity-rich factual claims require strict verification
        nlp = self._get_spacy()
        try:
            doc = nlp(claim)
            ent_count = len(doc.ents)
        except Exception:
            ent_count = 0
            
        if ent_count >= 1:
            return 0.85
            
        return 0.65

    def _full_nli(
        self,
        claims: list[str],
        facts: list[str],
        mode: str,
        action_score: float,
    ) -> dict:
        """
        Full verification using DeBERTa NLI.
        Batches all (claim, fact) pairs into a single forward pass.
        """
        if not claims:
            return {
                "verified": [],
                "rejected": [],
                "unverifiable": [],
                "mode": mode,
                "action_score": action_score,
            }

        if not facts:
            return {
                "verified": [],
                "rejected": [],
                "unverifiable": claims.copy(),
                "mode": mode,
                "action_score": action_score,
            }

        import torch
        import torch.nn.functional as F
        
        nlp = self._get_spacy()
        tokenizer, model = self._get_nli_model()
        device = model.device

        # Form all (claim, fact) pairs
        pairs = []
        for claim in claims:
            for fact in facts:
                pairs.append((claim, fact))

        # Check memory pressure and scale batch size
        base_batch_size = 64
        try:
            mem = psutil.virtual_memory()
            ram_percent = mem.percent
            if ram_percent >= 85:
                base_batch_size = 8
            elif ram_percent >= 75:
                base_batch_size = 16
            elif ram_percent >= 60:
                base_batch_size = 32
        except Exception:
            pass

        pair_results = {}
        
        # Batch inference
        for i in range(0, len(pairs), base_batch_size):
            batch = pairs[i:i + base_batch_size]
            batch_claims = [p[0] for p in batch]
            batch_facts = [p[1] for p in batch]

            inputs = tokenizer(
                batch_facts,
                batch_claims,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).cpu().tolist()
                probs = F.softmax(logits, dim=-1).cpu().tolist()

            for j, (claim, fact) in enumerate(batch):
                pred = predictions[j]
                prob_dist = probs[j]
                
                label = "neutral"
                id2label = getattr(model.config, "id2label", {}) or {}
                raw_label = str(id2label.get(pred, "")).lower()
                if "entail" in raw_label:
                    label = "entailment"
                elif "contrad" in raw_label:
                    label = "contradiction"
                elif "neutral" in raw_label:
                    label = "neutral"
                else:
                    if pred == 2:
                        label = "entailment"
                    elif pred == 0:
                        label = "contradiction"

                # Find entailment probability
                entail_idx = 2
                for idx, lbl in id2label.items():
                    if "entail" in str(lbl).lower():
                        entail_idx = idx
                        break
                entailment_score = prob_dist[entail_idx] if entail_idx < len(prob_dist) else prob_dist[-1]

                # Negation pre-check via spaCy parse
                claim_doc = nlp(claim)
                has_negation_claim = any(t.dep_ == "neg" or t.lower_ == "no" for t in claim_doc)
                fact_doc = nlp(fact)
                has_negation_fact = any(t.dep_ == "neg" or t.lower_ == "no" for t in fact_doc)
                negation_high_risk = has_negation_claim and not has_negation_fact

                threshold = 0.5 if negation_high_risk else 0.85
                if label == "entailment" and entailment_score <= threshold:
                    label = "neutral"

                # Geographic precision check
                if label == "entailment":
                    claim_gpes = {
                        ent.text.lower().strip()
                        for ent in claim_doc.ents
                        if ent.label_ in ("GPE", "LOC")
                    }
                    if claim_gpes:
                        if not any(gpe in fact.lower() for gpe in claim_gpes):
                            label = "neutral"

                # Topical overlap: a fact is "topically relevant" to a claim if they
                # share at least one non-stopword lemma or named entity.  Facts with
                # zero overlap are excluded from aggregation so an off-topic fact line
                # cannot veto an otherwise-supported claim.  claim_doc/fact_doc are
                # already parsed above for the negation check — reuse them.
                claim_lemmas = {
                    t.lemma_.lower()
                    for t in claim_doc
                    if not t.is_stop and t.is_alpha and len(t.text) > 2
                }
                fact_lemmas = {
                    t.lemma_.lower()
                    for t in fact_doc
                    if not t.is_stop and t.is_alpha and len(t.text) > 2
                }
                claim_ents_set = {e.text.lower() for e in claim_doc.ents}
                fact_ents_set  = {e.text.lower() for e in fact_doc.ents}
                has_topical_overlap = bool(
                    (claim_lemmas & fact_lemmas) or (claim_ents_set & fact_ents_set)
                )

                pair_results[(claim, fact)] = (label, entailment_score, has_topical_overlap)

        # Aggregate results
        # Fix 10: two-step aggregation.
        #
        # Step 1 — topical relevance filter:
        #   Only facts that share at least one non-stopword lemma or named entity
        #   with the claim are admitted to that claim's verdict pool.  A completely
        #   off-topic fact line cannot veto an otherwise-supported claim.
        #
        # Step 2 — majority vote over relevant facts:
        #   Contradiction wins only if contradicted_count > entailed_count.
        #   This prevents a single low-persistence frame (persistence=0.0000 in the
        #   fact text) from vetoing a claim that three other frames genuinely entail.
        #   Relevant negation/geo checks (applied per-pair above) are preserved.
        verified = []
        rejected = []
        unverifiable = []

        for claim in claims:
            all_results = [pair_results[(claim, fact)] for fact in facts]

            # Filter to topically relevant facts
            relevant = [
                (label, score)
                for label, score, has_overlap in all_results
                if has_overlap
            ]

            if not relevant:
                # No fact is topically related to this claim — cannot verify or reject
                unverifiable.append(claim)
                continue

            entailed_count    = sum(1 for label, _ in relevant if label == "entailment")
            contradicted_count = sum(1 for label, _ in relevant if label == "contradiction")

            if contradicted_count > entailed_count:
                # Contradictions outnumber entailments among relevant facts
                rejected.append(claim)
            elif entailed_count > 0:
                # At least one relevant fact entails, and entailments >= contradictions
                verified.append(claim)
            else:
                unverifiable.append(claim)

        return {
            "verified": verified,
            "rejected": rejected,
            "unverifiable": unverifiable,
            "mode": mode,
            "action_score": action_score,
        }

    def _ner_overlap(
        self,
        claims: list[str],
        facts: list[str],
        mode: str,
        action_score: float,
    ) -> dict:
        """
        Lightweight verification: named entity overlap between claims and facts.
        No DeBERTa forward pass. Used when action_score < cerberus_low_thresh.
        A claim is rejected only if a fact explicitly negates one of its entities.
        """
        nlp = self._get_spacy()

        verified, rejected, unverifiable = [], [], []
        fact_entities = set()
        for f in facts:
            fact_entities.update(ent.text.lower() for ent in nlp(f).ents)

        for claim in claims:
            claim_ents = {ent.text.lower() for ent in nlp(claim).ents}
            if not claim_ents:
                unverifiable.append(claim)
            elif claim_ents & fact_entities:
                verified.append(claim)
            else:
                unverifiable.append(claim)

        return {
            "verified":     verified,
            "rejected":     rejected,
            "unverifiable": unverifiable,
            "mode":         mode,
            "action_score": action_score,
        }

    def _extract_entity_claims(self, text: str, question_context: str = "") -> list[KnowledgeTriple]:
        """
        Fallback extraction for short factual answers, entities, and numeric patterns.
        Replicated from HADES shared/extractor.py.
        """
        if not text or "INSUFFICIENT DATA" in text.upper():
            return []
            
        nlp = self._get_spacy()
        doc = nlp(text)
        triples: list[KnowledgeTriple] = []
        
        entities = [ent for ent in doc.ents if ent.label_ in {"ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "DATE"}]
        for ent in entities:
            triples.append(KnowledgeTriple(
                subject=ent.text,
                verb="is",
                object=ent.label_.lower(),
                extraction_method="entity_fallback",
                is_deterministic=True
            ))
        if len(entities) >= 2:
            triples.append(KnowledgeTriple(
                subject=entities[0].text,
                verb="related to",
                object=entities[1].text,
                extraction_method="entity_fallback",
                is_deterministic=True
            ))
            
        numeric_regex = r'(\d+\.?\d*\s*%|\d+\.?\d*\s*(?:million|billion|thousand|tonnes|kg|km|years?|days?)|\b\d+\b)'
        for sent in doc.sents:
            for match in re.finditer(numeric_regex, sent.text, re.IGNORECASE):
                num_val = match.group()
                match_start_char = sent.start_char + match.start()
                
                nearest_chunk = None
                min_dist = float('inf')
                
                for chunk in doc.noun_chunks:
                    if chunk.start_char >= sent.start_char and chunk.end_char <= sent.end_char:
                        dist = min(abs(chunk.start_char - match_start_char), abs(chunk.end_char - match_start_char))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_chunk = chunk
                
                if nearest_chunk:
                    if nearest_chunk.text.strip().lower() == num_val.strip().lower():
                        continue
                        
                    triples.append(KnowledgeTriple(
                        subject=nearest_chunk.text,
                        verb="has value",
                        object=num_val,
                        extraction_method="entity_fallback",
                        is_deterministic=True
                    ))
                    
        return triples
