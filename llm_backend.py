"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes

_ollama_cache = None
_ollama_cache_time = 0.0

def detect_local_ollama() -> dict:
    """Checks host.docker.internal:11434 and localhost:11434 for Ollama instance."""
    global _ollama_cache, _ollama_cache_time
    now = time.time()
    if _ollama_cache is not None and (now - _ollama_cache_time) < 15.0:
        return _ollama_cache

    candidates = ["http://host.docker.internal:11434", "http://127.0.0.1:11434", "http://localhost:11434"]
    for host in candidates:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{host}/api/tags")
                if r.status_code == 200:
                    models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
                    _ollama_cache = {
                        "detected": True,
                        "host": host,
                        "baseUrl": f"{host}/v1",
                        "models": models,
                        "defaultModel": models[0] if models else DEFAULT_MODEL,
                    }
                    _ollama_cache_time = now
                    return _ollama_cache
        except Exception:
            continue
    _ollama_cache = {"detected": False, "models": []}
    _ollama_cache_time = now
    return _ollama_cache


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("AI_PROVIDER") or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    url = (os.environ.get("AI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "").strip()
    if url:
        return "openai"
    # Auto-detect Ollama on host
    det = detect_local_ollama()
    if det.get("detected"):
        return "openai"
    return "gemini"


def base_url() -> str:
    url = (os.environ.get("AI_BASE_URL") or os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
    if url:
        return url
    explicit = (os.environ.get("AI_PROVIDER") or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local") or detect_local_ollama().get("detected"):
        det = detect_local_ollama()
        if det.get("detected"):
            return det.get("baseUrl")
    return ""


def model_name() -> str:
    explicit = (os.environ.get("AI_MODEL") or os.environ.get("LLM_MODEL") or "").strip()
    if explicit:
        return explicit
    det = detect_local_ollama()
    if det.get("detected") and det.get("models"):
        for pref in ("llama3:8b", "llama3.1:8b", "mistral:7b", "gemma:7b"):
            if pref in det["models"]:
                return pref
        return det["models"][0]
    return DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def validate_connection(test_base_url: Optional[str] = None,
                        test_model: Optional[str] = None,
                        test_key: Optional[str] = None) -> Tuple[bool, str]:
    """Test connection to the OpenAI/Ollama endpoint."""
    b_url = (test_base_url or base_url()).rstrip("/")
    if not b_url:
        return False, "No base URL configured for OpenAI/Ollama provider."
    m_name = test_model or model_name()
    headers = {"Content-Type": "application/json"}
    k = (test_key or os.environ.get("LLM_API_KEY") or "ollama").strip()
    if k:
        headers["Authorization"] = f"Bearer {k}"
    try:
        with httpx.Client(timeout=60.0) as client:
            body = {"model": m_name, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}
            c_resp = client.post(f"{b_url}/chat/completions", json=body, headers=headers)
            if c_resp.status_code == 200:
                return True, f"Connected to {b_url} ({m_name})."
            # Try native ollama api if /v1 was omitted
            if c_resp.status_code == 404 and not b_url.endswith("/v1"):
                v1_resp = client.post(f"{b_url}/v1/chat/completions", json=body, headers=headers)
                if v1_resp.status_code == 200:
                    return True, f"Connected to {b_url}/v1 ({m_name})."
            return False, f"Server returned {c_resp.status_code}: {c_resp.text[:150]}"
    except Exception as e:
        return False, f"Connection to {b_url} failed: {type(e).__name__}: {str(e)[:150]}"


def validate_gemini(api_key: Optional[str] = None,
                    model: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """Test Gemini API key and model availability.
    Returns (valid, message_or_error, active_model).
    """
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        return False, "No Gemini API key provided.", None
    
    try:
        from google import genai
        client = genai.Client(api_key=key)
        candidates = [model] if model else ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        last_err = ""
        for m in candidates:
            if not m:
                continue
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents="Hi",
                    config={"max_output_tokens": 5}
                )
                return True, f"Verified Gemini access with model '{m}'.", m
            except Exception as e:
                err_msg = str(e)
                last_err = err_msg
                if "free tier" in err_msg.lower() or "403" in err_msg or "PERMISSION_DENIED" in err_msg:
                    continue  # try next candidate
                elif "API key not valid" in err_msg or "INVALID_ARGUMENT" in err_msg or "400" in err_msg:
                    return False, "Invalid Gemini API key. Please check your API key from Google AI Studio.", None
                else:
                    return False, f"Gemini error: {err_msg[:200]}", None
        
        # If all candidates failed due to free tier:
        if "free tier" in last_err.lower() or "403" in last_err or "PERMISSION_DENIED" in last_err:
            return False, "Gemini API Free Tier is not available for this project. Switch to Local AI (Ollama).", None
        return False, f"Gemini verification failed: {last_err[:200]}", None
    except Exception as e:
        return False, f"Failed to test Gemini client: {e}", None


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost
