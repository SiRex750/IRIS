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
import os
from dataclasses import dataclass

from iris.claim_contract import ANSWER_CLAIMS_WIRE_SCHEMA


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

    def caption(self, pil_image) -> str:
        import torch
        self._load()
        inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(out[0], skip_special_tokens=True)


class MiniCPMCaptioner:
    """Local image captioner using MiniCPM-V4.6 via Ollama."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or "minicpm-v4.6"
        self.endpoint = "http://localhost:11434"
        self.prompt = (
            "List everything visible in this image: every person, object, vehicle, "
            "and action. One short sentence per item. Only what is clearly visible."
        )
        self.num_predict = 250

    def caption(self, pil_image) -> str:
        import io
        import base64
        import requests
        import re

        # 1. Convert image to base64 JPEG
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # 2. Query Ollama vision endpoint
        payload = {
            "model": self.model_name,
            "prompt": self.prompt,
            "images": [image_b64],
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": self.num_predict
            },
            "think": False,
            "stream": False,
        }
        resp = requests.post(f"{self.endpoint}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()

        data = resp.json()
        raw = data["response"].strip()

        # Check and strip <think>...</think> blocks if present
        think_tag_re = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
        cleaned = think_tag_re.sub("", raw).strip()

        return cleaned


_ACTIVE_CAPTIONER: MiniCPMCaptioner | BLIPCaptioner | None = None


def get_captioner() -> MiniCPMCaptioner | BLIPCaptioner:
    """Returns the globally configured captioner (lazy-initialised)."""
    global _ACTIVE_CAPTIONER
    if _ACTIVE_CAPTIONER is None:
        _ACTIVE_CAPTIONER = MiniCPMCaptioner()
    return _ACTIVE_CAPTIONER


def set_captioner(captioner: MiniCPMCaptioner | BLIPCaptioner) -> None:
    """Override the active captioner (e.g. for testing)."""
    global _ACTIVE_CAPTIONER
    _ACTIVE_CAPTIONER = captioner


# ---------------------------------------------------------------------------
# Text generation — LLM backends
# ---------------------------------------------------------------------------

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
                 max_tokens: int | None = None, schema_format: bool = False) -> str:
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
                 max_tokens: int | None = None, schema_format: bool = False) -> str:
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
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            payload = {
                "model": model_name,
                "messages": messages,
                "format": ANSWER_CLAIMS_WIRE_SCHEMA,
                "stream": False,
                "options": options,
            }
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
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception:
                pass
        kwargs = dict(model=model_name, messages=messages, temperature=0.0)
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

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.endpoint, api_key="llama-server")
        return self._client

    def generate(self, prompt: str, context: str, model: str | None = None,
                 system_prompt: str | None = None, response_format: dict | None = None,
                 max_tokens: int | None = None, schema_format: bool = False) -> str:
        model_name = model or self.text_model
        sys_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]
        
        # Pin cache_prompt=False in extra_body (load-bearing config)
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "timeout": self.timeout,
            "extra_body": {"cache_prompt": False}
        }
        
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

        # PATH VERIFICATION COMMENT (Prompt 1.2):
        # We attempt to use the OpenAI-compatible endpoint (/v1/chat/completions)
        # with response_format carrying the json_schema and extra_body cache_prompt=false.
        # Since the live server was offline/unreachable during local verification, 
        # we prepare a try-except fallback to the native llama.cpp `/completion` 
        # endpoint (using raw requests) with `json_schema` and `cache_prompt` fields.
        try:
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
                 max_tokens: int | None = None, schema_format: bool = False) -> str:
        model_name = model or self.text_model
        sys_prompt = system_prompt if system_prompt is not None else _SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt + f"Provided Frame Evidence and Retrieval Context:\n{context}"},
            {"role": "user", "content": prompt},
        ]
        kwargs = dict(model=model_name, messages=messages, temperature=0.0)
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


def get_backend() -> LLMBackend:
    """Returns the globally configured active LLM backend."""
    global _ACTIVE_BACKEND, _BACKEND_OVERRIDDEN
    if _ACTIVE_BACKEND is None or not _BACKEND_OVERRIDDEN:
        try:
            from iris.iris_config import ConfigManager
            config = ConfigManager().get_config()
            cerberus_mode = getattr(config, "cerberus_mode", "legacy")
        except Exception:
            cerberus_mode = "legacy"
            config = None

        if cerberus_mode == "v2" and config is not None:
            backend_type = getattr(config, "answerer_backend", "llama_server")
            if backend_type == "llama_server":
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
                if not isinstance(_ACTIVE_BACKEND, LlamaBackend):
                    _ACTIVE_BACKEND = LlamaBackend()
        else:
            if not isinstance(_ACTIVE_BACKEND, LlamaBackend) and _ACTIVE_BACKEND is not None:
                _ACTIVE_BACKEND = LlamaBackend()
            elif _ACTIVE_BACKEND is None:
                _ACTIVE_BACKEND = LlamaBackend()

    return _ACTIVE_BACKEND


def set_backend(backend: LLMBackend) -> None:
    """Explicitly override the active LLM backend (e.g. for testing)."""
    global _ACTIVE_BACKEND, _BACKEND_OVERRIDDEN
    _ACTIVE_BACKEND = backend
    _BACKEND_OVERRIDDEN = True


def generate(prompt: str, context: str, model: str | None = None) -> str:
    """
    Generate a response from the active LLM backend.

    Args:
        prompt:  the user query or instruction
        context: formatted context string from L1 Elysium (as_context_text())
        model:   model identifier; if None, uses the backend's default

    Returns:
        Raw string response from the model
    """
    return get_backend().generate(prompt, context, model=model)


def generate_v2(prompt: str, context: str, model: str | None = None,
                 max_tokens: int | None = None, schema_format: bool = False) -> str:
    """Cerberus v2 contract generation: _SYSTEM_PROMPT_V2 (AnswerClaims JSON
    schema + few-shot example) plus a best-effort format=json request to the
    backend. Returns the RAW string response -- same raw-string contract as
    generate(); parsing into AnswerClaims, the strict-parse retry, and
    compliance-failure tracking are the caller's job
    (iris.query._generate_answer_claims_v2), not this function's.

    max_tokens is additive and defaults to None (unbounded, today's
    production behavior unchanged); callers that want to cap runaway
    generation (e.g. scripts/answerer_bakeoff.py) pass it explicitly.

    schema_format is additive and defaults to False (byte-identical to
    today: json-mode via response_format={"type":"json_object"}). When True,
    routes to grammar-guaranteed schema-constrained decoding
    (iris.claim_contract.ANSWER_CLAIMS_WIRE_SCHEMA) instead -- see
    LlamaBackend.generate. Structure is then guaranteed by the grammar; the
    caller parses via AnswerClaims.from_wire, not from_json.
    """
    return get_backend().generate(
        prompt, context, model=model,
        system_prompt=_SYSTEM_PROMPT_V2,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        schema_format=schema_format,
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

def run_diagnostics() -> dict:
    """Run startup diagnostics; raises RuntimeError if a required backend is misconfigured."""
    import json

    backend = get_backend()
    backend_class = backend.__class__.__name__

    api_key = os.environ.get("OPENAI_API_KEY")
    api_key_present = bool(api_key and api_key.strip())

    if hasattr(backend, "text_model"):
        model = backend.text_model
    elif backend_class in ("MockLLMBackend", "MockBackend"):
        model = "mock-model"
    else:
        model = "unknown"

    captioner = get_captioner()
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

def generate_caption_for_frame(frame, frame_idx: int | None = None) -> CaptionResult:
    """Generate a semantic caption for a PyAV VideoFrame or PIL Image using the active captioner."""
    import time

    t_start = time.time()

    if frame is None:
        err_msg = "No frame provided for captioning."
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": 0.0, "error": err_msg})
        return result

    # 1. Extract PIL image from PyAV frame if needed
    try:
        img = frame.to_image() if hasattr(frame, "to_image") else frame
    except Exception as e:
        err_msg = f"Failed to convert frame to image for captioning: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": time.time() - t_start, "error": err_msg})
        return result

    # 2. Caption with active captioner
    try:
        captioner = get_captioner()
        caption = captioner.caption(img)
        return CaptionResult(success=True, caption=caption)
    except Exception as e:
        err_msg = f"Captioning failed: {e}"
        result = CaptionResult(success=False, caption="[CAPTION_FAILED]", error=err_msg)
        _CAPTION_FAILURES.append({"frame_idx": frame_idx, "latency": time.time() - t_start, "error": err_msg})
        return result
