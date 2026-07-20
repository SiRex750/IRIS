"""
ARIA — LLM interface abstraction for IRIS.

Single entry point for all LLM calls in the pipeline.

Captioning:  BLIPCaptioner  — local Salesforce/blip-image-captioning-base
Text gen:    LlamaBackend   — local Ollama at localhost:11434 (default)
             OpenAIBackend  — OpenAI API (opt-in via set_backend)

No other file in IRIS should import openai, transformers, or call
any LLM/VLM API directly — all calls go through here.

Owner: Track B
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass

from iris.claim_contract import ANSWER_CLAIMS_WIRE_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class CaptionResult:
    success: bool
    caption: str | None
    error: str | None = None


class CaptionGenerationError(Exception):
    """Exception raised when caption generation fails."""
    pass


# ---------------------------------------------------------------------------
# Vision captioning — BLIP
# ---------------------------------------------------------------------------

class BLIPCaptioner:
    """Local image captioner using Salesforce BLIP (via HuggingFace transformers)."""

    DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._processor = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)

    def caption(self, pil_image, focus_hint: str | None = None) -> str:
        # NOTE: focus_hint is accepted for interface parity with
        # MoondreamCaptioner/MiniCPMCaptioner (see _ensure_captions in
        # query.py) but is not applied here -- BLIP's unconditional
        # captioning API in this codebase does not take a text prompt, and
        # conditional BLIP captioning (image + text prefix) has different
        # continuation semantics that were out of scope for this fix. BLIP is
        # a fallback-only captioner (see _clip.py::get_semantic_and_clip_caption),
        # so this only affects behavior when the primary captioner has failed.
        import torch
        self._load()
        inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(out[0], skip_special_tokens=True)




class MoondreamCaptioner:
    def __init__(self) -> None:
        self.model_name = 'vikhyatk/moondream2'
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        # P1-19: Respect CPU/GPU availability — do not hard-code device_map='cuda'.
        # On CPU-only machines the hard-coded value crashes with a CUDA-not-available
        # error.  The captioner still prefers GPU when one is available.
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._tokenizer = AutoTokenizer.from_pretrained('vikhyatk/moondream2', trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            'vikhyatk/moondream2',
            trust_remote_code=True,
            torch_dtype=torch.float16 if self._device == 'cuda' else torch.float32,
            device_map=self._device,
        )

    BASE_PROMPT = (
        'Describe only what is visually present in this single image. State objects, '
        'people, colors, and positions. Do not describe motion or changes.'
    )

    def caption(self, pil_image, focus_hint: str | None = None) -> str:
        import torch
        self._load()
        if str(self._model.device) == 'cpu':
            self._model = self._model.to(self._device)
        enc = self._model.encode_image(pil_image)
        prompt = self.BASE_PROMPT if not focus_hint else f"{self.BASE_PROMPT} {focus_hint}"
        return self._model.answer_question(enc, prompt, self._tokenizer).strip()

class MiniCPMCaptioner:
    """Local image captioner using MiniCPM via Ollama."""

    def __init__(self, model_name: str | None = None) -> None:
        self.endpoint = "http://localhost:11434"
        if model_name is None:
            try:
                import requests
                resp = requests.get(f"{self.endpoint}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    # Match any version of minicpm-v
                    if "minicpm-v4.6:latest" in models or "minicpm-v4.6" in models:
                        model_name = "minicpm-v4.6"
                    elif "minicpm-v:latest" in models or "minicpm-v" in models:
                        model_name = "minicpm-v"
                    else:
                        matched = next((m for m in models if "minicpm" in m.lower()), None)
                        if matched:
                            model_name = matched.split(":")[0]
            except Exception:
                pass
        self.model_name = model_name or "minicpm-v"
        # Seated production captioner is minicpm-v4.6 (confirmed via a separate
        # live infra-seating pass) -- log which variant this instance actually
        # resolved to so a misconfiguration (e.g. Ollama only has the older
        # minicpm-v tag, or minicpm-v4.6 was never pulled) is visible in logs
        # instead of silently answering with the wrong checkpoint.
        logger.info(
            "MiniCPMCaptioner resolved model_name=%r (endpoint=%r)",
            self.model_name, self.endpoint,
        )
        self.prompt = (
            "List everything visible in this image: every person, object, vehicle, "
            "and action. One short sentence per item. Only what is clearly visible."
        )
        # Was 250: raised to make truncation rare in practice rather than
        # discarding a real fraction of captions outright (an empty caption
        # gives CerberusV zero evidence, strictly worse than a longer-but-
        # complete one). Still not unbounded -- num_predict, not max_tokens,
        # is what actually bounds a single Ollama generate call.
        self.num_predict = 400
        self.retry_num_predict = 700  # one retry at a higher budget before giving up

    def caption(self, pil_image, focus_hint: str | None = None) -> str:
        import io
        import base64

        # 1. Convert image to base64 JPEG
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        prompt = self.prompt if not focus_hint else f"{self.prompt} {focus_hint}"

        _MINICPM_TRUNCATION_STATS["total_calls"] += 1

        data = self._generate(image_b64, prompt, self.num_predict)
        if data.get("done_reason") == "length":
            _MINICPM_TRUNCATION_STATS["truncated_first_attempt"] += 1
            # Retry once with a higher token budget before discarding --
            # matches the surrounding "truncated captions give zero evidence"
            # rationale: prefer a complete answer at higher cost over silently
            # dropping the caption.
            data = self._generate(image_b64, prompt, self.retry_num_predict)
            if data.get("done_reason") == "length":
                _MINICPM_TRUNCATION_STATS["truncated_after_retry"] += 1
                return ""

        raw = data.get("response", "").strip()

        # Check and strip <think>...</think> blocks if present
        import re
        think_tag_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        cleaned = think_tag_re.sub("", raw).strip()

        return cleaned

    def _generate(self, image_b64: str, prompt: str, num_predict: int) -> dict:
        import requests

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": num_predict,
            },
            "think": False,
            "stream": False,
        }
        resp = requests.post(f"{self.endpoint}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()


_MINICPM_TRUNCATION_STATS: dict[str, int] = {
    "total_calls": 0,
    "truncated_first_attempt": 0,
    "truncated_after_retry": 0,
}


def get_minicpm_truncation_stats() -> dict:
    """Truncation-rate instrumentation for MiniCPMCaptioner (item 5): was
    previously invisible in layer3_outputs.csv -- a caption that got
    discarded (done_reason=="length" on both the initial call and the retry)
    left no trace anywhere. truncated_after_retry / total_calls is the rate
    that should stay under 5% before minicpm-v4.6 is considered as a
    default-captioner candidate over moondream, per the task instruction."""
    stats = dict(_MINICPM_TRUNCATION_STATS)
    stats["truncation_rate_after_retry"] = (
        stats["truncated_after_retry"] / stats["total_calls"] if stats["total_calls"] else None
    )
    stats["first_attempt_truncation_rate"] = (
        stats["truncated_first_attempt"] / stats["total_calls"] if stats["total_calls"] else None
    )
    return stats


def reset_minicpm_truncation_stats() -> None:
    _MINICPM_TRUNCATION_STATS["total_calls"] = 0
    _MINICPM_TRUNCATION_STATS["truncated_first_attempt"] = 0
    _MINICPM_TRUNCATION_STATS["truncated_after_retry"] = 0


_ACTIVE_CAPTIONER: MiniCPMCaptioner | MoondreamCaptioner | BLIPCaptioner | None = None

def get_captioner(config: Any = None) -> MiniCPMCaptioner | MoondreamCaptioner | BLIPCaptioner:
    global _ACTIVE_CAPTIONER
    if _ACTIVE_CAPTIONER is None:
        try:
            if config is None:
                from iris.iris_config import ConfigManager
                config = ConfigManager().get_config()
            backend_type = getattr(config, 'captioner_backend', 'minicpm')
        except Exception:
            backend_type = 'minicpm'

        if backend_type == 'moondream':
            _ACTIVE_CAPTIONER = MoondreamCaptioner()
        elif backend_type == 'blip':
            _ACTIVE_CAPTIONER = BLIPCaptioner()
        else:
            _ACTIVE_CAPTIONER = MiniCPMCaptioner()
    return _ACTIVE_CAPTIONER

def set_captioner(captioner: MiniCPMCaptioner | MoondreamCaptioner | BLIPCaptioner) -> None:
    global _ACTIVE_CAPTIONER
    _ACTIVE_CAPTIONER = captioner

def unload_captioner() -> None:
    """Offload the active captioner's model weights and release the captioner.

    P1-18: The previous implementation moved the model to CPU but kept
    ``_ACTIVE_CAPTIONER`` alive.  A subsequent call to ``get_captioner()``
    returned the same object and, when ``caption()`` detected
    ``model.device == 'cpu'``, moved it back to GPU — but only for
    MoondreamCaptioner.  BLIPCaptioner has no such re-move logic, so it would
    silently run on CPU or crash with a device-mismatch on CUDA tensors.

    Fix: fully null out ``_ACTIVE_CAPTIONER`` after unloading so that the next
    ``get_captioner()`` constructs a fresh instance on the correct device.
    """
    global _ACTIVE_CAPTIONER
    import torch
    if _ACTIVE_CAPTIONER is not None:
        if hasattr(_ACTIVE_CAPTIONER, '_model') and _ACTIVE_CAPTIONER._model is not None:
            try:
                _ACTIVE_CAPTIONER._model.to('cpu')
            except Exception:
                pass  # ignore errors during offload (e.g. model already on CPU)
            torch.cuda.empty_cache()
        _ACTIVE_CAPTIONER = None


_SYSTEM_PROMPT = (
    "You are ARIA, a video understanding assistant.\n\n"
    "Use only the provided frame evidence and retrieval context.\n\n"
    "Answer in clear natural language.\n\n"
    "When describing events, reference timestamps and supporting frames.\n\n"
    "If evidence is insufficient, explicitly say so.\n\n"
    "Do not invent events that are not supported by the provided context.\n\n"
    "Prefer concise but human-readable explanations over raw metadata.\n\n"
)

# ── Cerberus v2 contract prompt ─────────────────────────────────────────────
#
# Schema and the four claim types are iris/claim_contract.py's contract --
# this prompt does not define its own shape, it instructs the model to
# produce that exact shape. The few-shot examples are lifted verbatim from
# claim_contract.py's module docstring so the two never drift apart.
#
# FLAT shape (task 3): matches iris.claim_contract.ANSWER_CLAIMS_WIRE_SCHEMA
# -- one claim object type discriminated by "claim_type", not the nested
# {"type": "visual", ...} shape from_dict/from_json use. Used for BOTH the
# json-mode path (response_format={"type":"json_object"}) and the
# schema-constrained path (format=ANSWER_CLAIMS_WIRE_SCHEMA) -- Ollama's own
# docs recommend prompt+grammar agree, since a grammar with a contradicting
# prompt yields garbage-in-valid-shape (structurally valid, semantically
# empty/wrong).
#
# SENTINEL SHAPE (task 4): every property is now REQUIRED on every claim
# object (ANSWER_CLAIMS_WIRE_SCHEMA), so the few-shot examples below fill
# every irrelevant property with its fixed sentinel (-1 / "" / "none") rather
# than omitting it -- the prompt must show the shape the grammar actually
# enforces, or a contradicting prompt just yields sentinel-omission errors.
#
# TASK-4 ROOT-CAUSE FIX -- report of prior state: before this task there was
# exactly ONE few-shot example, and while it did contain an AbsenceClaim,
# that claim was never is_core (the VisualClaim was core, absence was an
# afterthought last in the list) -- there was no demonstration of a NEGATIVE-
# style answer centered on an absence claim. That is the second hypothesized
# cause (alongside the ungrammar-enforceable field presence, task 4 Part A)
# of the task-3 constrained bake-off's zero AbsenceClaim production. Example
# 2 below is new: a NEGATIVE query answered with claim_type="absence" AS THE
# CORE claim.
_SYSTEM_PROMPT_V2 = (
    "You are ARIA, a video understanding assistant.\n\n"
    "Use only the provided frame evidence and retrieval context.\n\n"
    "Respond with ONLY a single JSON object (no prose, no markdown code fences) matching this schema:\n\n"
    "{\n"
    '  "query": "<the question you were asked>",\n'
    '  "claims": [\n'
    '    {"claim_type": "visual"|"metadata"|"absence"|"global", "frame_idx": <int>, "assertion": <str>, '
    '"is_core": <bool>, "field": "action_score"|"persistence"|"timestamp_sec"|"none", "stated_value": <float>, '
    '"source_text": <str>, "event": <str>, "text": <str>}\n'
    "  ]\n"
    "}\n\n"
    "Every claim is ONE flat object carrying ALL of these properties, every time -- claim_type says which "
    "of the four kinds it is; fill EVERY property NOT used by that kind with its sentinel value, never omit "
    "it:\n"
    "  frame_idx: -1   stated_value: -1   field: \"none\"   assertion / source_text / event / text: \"\"\n"
    "Which properties are REAL for each claim_type (everything else on that object must be the sentinel):\n"
    "  \"visual\":   frame_idx, assertion, is_core\n"
    "  \"metadata\": frame_idx, field, stated_value, source_text\n"
    "  \"absence\":  event, is_core\n"
    "  \"global\":   text\n"
    "is_core is required on every claim (never a sentinel) -- true/false is always a real answer: for "
    "\"metadata\"/\"global\" claims it is always false.\n\n"
    "Rules:\n"
    "- Exactly ONE claim among the \"visual\"/\"absence\" claims must have \"is_core\": true -- this is the "
    "claim the answer's badge is judged against. \"metadata\" and \"global\" claims always have \"is_core\": false.\n"
    "- Every \"visual\" claim must cite the frame_idx it is actually about. Its \"assertion\" text must be "
    "plain visual language only -- no frame numbers, timestamps, or metric numbers inside the assertion "
    "itself; those belong in a separate \"metadata\" claim.\n"
    "- Every \"absence\" claim's \"event\" MUST be phrased as the POSITIVE event that would need to be present "
    "to contradict the claim (e.g. \"a person running or fleeing\"), never as a negation (\"no person "
    "running\", \"nobody is running\") -- it will be checked by searching the evidence for that positive event.\n"
    "- If a query asks whether something is present/happening and the evidence does not show it, answer with "
    "an \"absence\" claim (is_core: true, event phrased positively) instead of omitting a core claim.\n"
    "- Do not invent events, frames, or numbers that are not supported by the provided context.\n\n"
    "EXAMPLE 1 -- for the question \"Is anyone loading a vehicle?\", evidence shows a parked car but no "
    "loading activity (core claim is visual, absence is supporting):\n"
    '{"query": "Is anyone loading a vehicle?", "claims": ['
    '{"claim_type": "visual", "frame_idx": 23760, "assertion": "a car is parked near the entrance", '
    '"is_core": true, "field": "none", "stated_value": -1, "source_text": "", "event": "", "text": ""}, '
    '{"claim_type": "metadata", "frame_idx": 23760, "field": "persistence", "stated_value": 0.0, '
    '"source_text": "frame 23760 shows a low persistence score of 0.00", "assertion": "", "is_core": false, '
    '"event": "", "text": ""}, '
    '{"claim_type": "absence", "event": "a person loading or unloading a vehicle", "is_core": false, '
    '"frame_idx": -1, "assertion": "", "field": "none", "stated_value": -1, "source_text": "", "text": ""}, '
    '{"claim_type": "global", "text": "Overall the parking lot appears static across the clip.", '
    '"frame_idx": -1, "assertion": "", "is_core": false, "field": "none", "stated_value": -1, "source_text": "", '
    '"event": ""}'
    "]}\n\n"
    "EXAMPLE 2 -- for the question \"Is anyone loading a vehicle?\", evidence shows NO loading activity at "
    "all (core claim is absence -- this is the shape a NEGATIVE answer should take):\n"
    '{"query": "Is anyone loading a vehicle?", "claims": ['
    '{"claim_type": "absence", "event": "a person loading a vehicle", "is_core": true, "frame_idx": -1, '
    '"assertion": "", "field": "none", "stated_value": -1, "source_text": "", "text": ""}, '
    '{"claim_type": "global", "text": "All retrieved frames show parked, stationary vehicles with no people '
    'interacting with them.", "frame_idx": -1, "assertion": "", "is_core": false, "field": "none", '
    '"stated_value": -1, "source_text": "", "event": ""}'
    "]}\n\n"
)


class LLMBackend:
    """Abstract base class for LLM backends."""
    def generate(self, prompt: str, context: str, model: str | None = None,
                 system_prompt: str | None = None, response_format: dict | None = None,
                 max_tokens: int | None = None, schema_format: bool = False,
                 seed: int | None = None, keep_alive: int | None = None) -> str:
        raise NotImplementedError("LLM backend must implement generate()")


class LlamaBackend(LLMBackend):
    """Local LLM via Ollama (OpenAI-compatible endpoint at localhost:11434)."""

    DEFAULT_TEXT_MODEL = "llama3.2:3b"

    def __init__(self, endpoint: str = "http://localhost:11434/v1",
                 text_model: str | None = None) -> None:
        self.endpoint = endpoint
        self.text_model = text_model or self.DEFAULT_TEXT_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.endpoint, api_key="ollama")
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None,
                 system_prompt: str | None = None, response_format: dict | None = None,
                 max_tokens: int | None = None, schema_format: bool = False,
                 seed: int | None = None, keep_alive: int | None = 0) -> str:
        model_name = model or self.text_model
        sys_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]
        if schema_format:
            # Schema-constrained decoding (grammar-guaranteed structure) needs
            # Ollama's NATIVE /api/chat endpoint with format=<json-schema
            # object> -- the OpenAI-compat /v1/chat/completions endpoint
            # (self.client, used by every other branch here) does not carry a
            # full schema object reliably, only the coarse
            # response_format={"type":"json_object"} used by the json-mode
            # path above. think:false is not needed here: the grammar forces
            # the '{' token first, so there is no room for a reasoning
            # preamble to consume the budget (see task 2's qwen3.5-2b finding).
            import requests
            native_base = self.endpoint[:-3] if self.endpoint.endswith("/v1") else self.endpoint
            options: dict = {"temperature": 0.0}
            if seed is not None:
                options["seed"] = seed
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            payload = {
                "model": model_name,
                "messages": messages,
                "format": ANSWER_CLAIMS_WIRE_SCHEMA,
                "stream": False,
                "options": options,
            }
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            # timeout=600 (not 300): reasoning-capable models under
            # schema-format have been observed taking 300-350s+ for a SINGLE
            # attempt (qwen3.5-4b, task 4) -- 300s crashed a live run with an
            # unhandled ReadTimeout mid-retry. 600s gives real headroom
            # without masking a genuinely hung request.
            resp = requests.post(f"{native_base}/api/chat", json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        if response_format is not None:
            # Best-effort: not every Ollama/openai-client version supports
            # response_format on the chat.completions endpoint. Fall back to
            # the plain call (same prompt, same low temperature) rather than
            # erroring the whole query -- v2's strict-parse retry downstream
            # is what actually enforces the contract either way.
            try:
                kwargs = dict(model=model_name, messages=messages, temperature=0.0,
                              response_format=response_format)
                if seed is not None:
                    kwargs["seed"] = seed
                if keep_alive is not None:
                    kwargs["extra_body"] = {"keep_alive": keep_alive}
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception:
                pass
        kwargs = dict(model=model_name, messages=messages, temperature=0.0)
        if seed is not None:
            kwargs["seed"] = seed
        if keep_alive is not None:
            kwargs["extra_body"] = {"keep_alive": keep_alive}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class LlamaServerBackend(LLMBackend):
    """Local LLM via llama-server OpenAI-compatible endpoint."""

    def __init__(self, endpoint: str = "http://localhost:8080/v1",
                 text_model: str | None = None,
                 timeout: float = 600.0) -> None:
        self.endpoint = endpoint
        self.text_model = text_model or "granite4:micro"
        self.timeout = timeout
        self._client = None
        # Guard against the exact misconfiguration a separate infra-seating
        # pass flagged: answerer_backend="llama_server" is meant to reach
        # llama-server (which this class relies on for per-request
        # cache_prompt=false -- required for the seed-based determinism fix),
        # NOT Ollama's default port 11434. Ollama's OpenAI-compat endpoint
        # would silently ignore the llama.cpp-specific cache_prompt field
        # rather than erroring, so a request that lands there instead breaks
        # the determinism guarantee with no visible failure. Warn, don't
        # raise -- this is a strong signal, not a certainty (a real
        # llama-server COULD be configured on 11434 by a non-default choice).
        if "11434" in endpoint:
            logger.warning(
                "LlamaServerBackend constructed with endpoint=%r, which uses Ollama's "
                "default port (11434). answerer_backend=\"llama_server\" requires an "
                "actual llama-server process (the seated production answerer runs on "
                "port 8091) -- llama-server's cache_prompt=false request parameter, "
                "required for determinism, is silently ignored if this endpoint is "
                "actually Ollama. If this is intentional (a non-default llama-server "
                "port choice), ignore this warning.",
                endpoint,
                stacklevel=2,
            )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.endpoint, api_key="llama-server")
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None,
                 system_prompt: str | None = None, response_format: dict | None = None,
                 max_tokens: int | None = None, schema_format: bool = False,
                 seed: int | None = None, keep_alive: int | None = None) -> str:
        # keep_alive is an Ollama-specific model-lifecycle parameter (see
        # LlamaBackend); llama-server has no such concept and already
        # disables its own analogous prompt-cache reuse via cache_prompt=False
        # below, so this argument is accepted for interface parity and ignored.
        model_name = model or self.text_model
        sys_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]

        # 2. Existing path (for non-schema or mock client unit tests)
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "timeout": self.timeout,
            "extra_body": {"cache_prompt": False}
        }
        if seed is not None:
            kwargs["seed"] = seed

        if schema_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_claims",
                    "schema": ANSWER_CLAIMS_WIRE_SCHEMA,
                    "strict": True,
                }
            }
        elif response_format is not None:
            kwargs["response_format"] = response_format

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            # 1. Production direct requests.post path for schema_format
            if schema_format and self._client is None:
                import requests
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0,
                    "seed": seed if seed is not None else 42,
                    "cache_prompt": False,
                    "max_tokens": max_tokens if max_tokens is not None else 1024,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer_claims",
                            "schema": ANSWER_CLAIMS_WIRE_SCHEMA,
                            "strict": True
                        }
                    }
                }
                resp = requests.post(f"{self.endpoint}/chat/completions", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # Fall back to native llama-server /completion endpoint
            import requests
            native_base = self.endpoint[:-3] if self.endpoint.endswith("/v1") else self.endpoint
            
            # Format the messages into a single prompt string for raw /completion
            formatted_prompt = ""
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted_prompt += "<|im_start|>assistant\n"
            
            payload = {
                "prompt": formatted_prompt,
                "temperature": 0.0,
                "seed": seed if seed is not None else 42,
                "cache_prompt": False,
            }
            if schema_format:
                payload["json_schema"] = ANSWER_CLAIMS_WIRE_SCHEMA
            if max_tokens is not None:
                # llama.cpp uses n_predict
                payload["n_predict"] = max_tokens
                
            try:
                resp = requests.post(f"{native_base}/completion", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["content"]
            except Exception:
                raise e



class OpenAIBackend(LLMBackend):
    """OpenAI API implementation."""

    DEFAULT_TEXT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.text_model = self.DEFAULT_TEXT_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set.\n"
                    "To run IRIS dynamically, please either:\n"
                    "  1. Set the OPENAI_API_KEY environment variable.\n"
                    "  2. Use the local LlamaBackend via Ollama (default)."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None,
                 system_prompt: str | None = None, response_format: dict | None = None,
                 max_tokens: int | None = None, schema_format: bool = False,
                 seed: int | None = None, keep_alive: int | None = None) -> str:
        # keep_alive is an Ollama-specific model-lifecycle parameter (see
        # LlamaBackend); the real OpenAI API has no such concept, so this
        # argument is accepted for interface parity and ignored.
        model_name = model or self.text_model
        sys_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]
        kwargs = dict(model=model_name, messages=messages, temperature=0.0)
        if seed is not None:
            kwargs["seed"] = seed
        if schema_format:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer_claims", "schema": ANSWER_CLAIMS_WIRE_SCHEMA, "strict": True},
            }
        elif response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


_ACTIVE_BACKEND: LLMBackend | None = None
_BACKEND_OVERRIDDEN: bool = False


def get_backend(config: Any = None) -> LLMBackend:
    """Returns the configured active LLM backend.

    P1-16 / P1-17: Accept and respect config settings dynamically rather than
    relying only on the static global configuration.
    """
    global _ACTIVE_BACKEND, _BACKEND_OVERRIDDEN
    if _ACTIVE_BACKEND is None or not _BACKEND_OVERRIDDEN:
        if config is None:
            try:
                from iris.iris_config import ConfigManager
                config = ConfigManager().get_config()
            except Exception:
                config = None

        cerberus_mode = getattr(config, "cerberus_mode", "legacy") if config is not None else "legacy"
        backend_type = getattr(config, "answerer_backend", "llama_server") if config is not None else "llama_server"

        if backend_type == "llama_server" and config is not None:
            endpoint = getattr(config, "answerer_endpoint", "http://localhost:8080/v1")
            model = getattr(config, "answerer_model", "granite4:micro")
            timeout = getattr(config, "answerer_timeout", 600.0)
            # Avoid recreating if it is already LlamaServerBackend with the correct parameters
            if not isinstance(_ACTIVE_BACKEND, LlamaServerBackend) or \
               _ACTIVE_BACKEND.endpoint != endpoint or \
               _ACTIVE_BACKEND.text_model != model or \
               _ACTIVE_BACKEND.timeout != timeout:
                _ACTIVE_BACKEND = LlamaServerBackend(
                    endpoint=endpoint,
                    text_model=model,
                    timeout=timeout
                )
        else:
            endpoint = getattr(config, "answerer_endpoint", "http://localhost:11434/v1") if config is not None else "http://localhost:11434/v1"
            model = getattr(config, "answerer_model", None) if config is not None else None
            # Default to LlamaBackend
            if not isinstance(_ACTIVE_BACKEND, LlamaBackend) or \
               _ACTIVE_BACKEND.endpoint != endpoint or \
               _ACTIVE_BACKEND.text_model != (model or LlamaBackend.DEFAULT_TEXT_MODEL):
                _ACTIVE_BACKEND = LlamaBackend(
                    endpoint=endpoint,
                    text_model=model
                )

    return _ACTIVE_BACKEND


def set_backend(backend: LLMBackend) -> None:
    """Explicitly override the active LLM backend (e.g. for testing)."""
    global _ACTIVE_BACKEND, _BACKEND_OVERRIDDEN
    _ACTIVE_BACKEND = backend
    _BACKEND_OVERRIDDEN = True


def generate(prompt: str, context: str, model: str | None = None, max_tokens: int | None = None, config: Any = None) -> str:
    """
    Generate a response from the active LLM backend.

    Args:
        prompt:  the user query or instruction
        context: formatted context string from L1 Elysium (as_context_text())
        model:   model identifier; if None, uses the backend's default
        max_tokens: max tokens limit for generation
        config: config snapshot to dynamically initialize the backend

    Returns:
        Raw string response from the model
    """
    seed = getattr(config, "answerer_seed", 42) if config is not None else 42
    keep_alive = getattr(config, "answerer_keep_alive", 0) if config is not None else 0
    return get_backend(config).generate(prompt, context, model=model, max_tokens=max_tokens,
                                         seed=seed, keep_alive=keep_alive)


def generate_v2(prompt: str, context: str, model: str | None = None,
                 max_tokens: int | None = None, schema_format: bool = False, config: Any = None) -> str:
    """Cerberus v2 contract generation: _SYSTEM_PROMPT_V2 (AnswerClaims JSON
    schema + few-shot example) plus a best-effort format=json request to the
    backend.
    """
    seed = getattr(config, "answerer_seed", 42) if config is not None else 42
    keep_alive = getattr(config, "answerer_keep_alive", 0) if config is not None else 0
    return get_backend(config).generate(
        prompt, context, model=model,
        system_prompt=_SYSTEM_PROMPT_V2,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        schema_format=schema_format,
        seed=seed,
        keep_alive=keep_alive,
    )


# ---------------------------------------------------------------------------
# Caption failures log
# ---------------------------------------------------------------------------

_CAPTION_FAILURES: list = []


def get_caption_failures() -> list:
    """Return all stored captioning failures."""
    return _CAPTION_FAILURES


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(config: Any = None) -> dict:
    """Run startup diagnostics; raises RuntimeError if a required backend is misconfigured."""
    import json

    backend = get_backend(config)
    backend_class = backend.__class__.__name__

    api_key = os.environ.get("OPENAI_API_KEY")
    api_key_present = bool(api_key and api_key.strip())

    if hasattr(backend, "text_model"):
        model = backend.text_model
    elif backend_class in ("MockLLMBackend", "MockBackend"):
        model = "mock-model"
    else:
        model = "unknown"

    captioner = get_captioner(config)
    captioner_class = captioner.__class__.__name__
    captioner_model = getattr(captioner, "model_name", "unknown")

    diag = {
        "backend": backend_class,
        "model": model,
        "captioner": captioner_class,
        "captioner_model": captioner_model,
        "api_key_present": api_key_present,
    }

    print(f"DIAGNOSTICS: {json.dumps(diag)}")

    if backend_class == "OpenAIBackend" and not api_key_present:
        raise RuntimeError("OpenAIBackend is active but OPENAI_API_KEY is not set.")

    return diag


# ---------------------------------------------------------------------------
# Frame captioning entry point
# ---------------------------------------------------------------------------

def generate_caption_for_frame(frame, frame_idx: int | None = None, config: Any = None,
                                focus_hint: str | None = None) -> CaptionResult:
    """Generate a semantic caption for a PyAV VideoFrame or PIL Image using the active captioner.

    focus_hint: optional question/choices text (built by iris.query._build_focus_hint) telling
    the captioner what detail matters for this query, so e.g. "a woman and two children eating
    ice cream" can become "a woman in grey holding a spoon..." when the question hinges on
    clothing color. None (the default) reproduces the old generic, question-blind prompt.
    """
    import time

    t_start = time.monotonic()

    if frame is None:
        err_msg = "No frame provided for captioning."
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        # P2-06: Record actual elapsed time even for early-exit paths.
        _CAPTION_FAILURES.append({
            "frame_idx": frame_idx,
            "latency": time.monotonic() - t_start,
            "error": err_msg,
        })
        return result

    # 1. Extract PIL image from PyAV frame if needed
    try:
        img = frame.to_image() if hasattr(frame, "to_image") else frame
    except Exception as e:
        err_msg = f"Failed to convert frame to image for captioning: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": time.monotonic() - t_start, "error": err_msg})
        return result

    # 2. Caption with active captioner
    try:
        captioner = get_captioner(config)
        try:
            caption = captioner.caption(img, focus_hint=focus_hint)
        except TypeError:
            # Backward compatibility with any captioner (e.g. a test double)
            # whose caption() doesn't accept focus_hint yet.
            caption = captioner.caption(img)
        latency = time.monotonic() - t_start
        # P2-06: Log empty captions as observable failures.
        if not caption or not caption.strip():
            err_msg = "Captioner returned empty string"
            result = CaptionResult(success=False, caption="[CAPTION_EMPTY]", error=err_msg)
            _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": latency, "error": err_msg})
            return result
        return CaptionResult(success=True, caption=caption)
    except Exception as e:
        err_msg = f"Captioning failed: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": time.monotonic() - t_start, "error": err_msg})
        return result
