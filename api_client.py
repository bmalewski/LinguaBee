"""Shared HTTP client utilities for OpenRouter and Gemini API calls.

Centralizes retry logic, backoff, and response parsing to avoid duplication
across correction_adapters, summarization_manager, and translation_manager.

Uses a module-level persistent httpx.Client per API (OpenRouter / Gemini) so
that TCP connections are reused across batch calls, avoiding repeated TLS
handshakes on every segment/chunk.
"""
import time
import httpx

_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_HEADERS_BASE = {
    "Content-Type": "application/json",
    "HTTP-Referer": "https://linguabee.local",
    "X-Title": "LinguaBee",
}
_DEFAULT_OPENROUTER_MODEL = "google/gemini-3.5-flash"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Persistent clients — created on first use and reused for the lifetime of the
# process.  This avoids a new TLS handshake for every API call.
_openrouter_client: httpx.Client | None = None
_gemini_client: httpx.Client | None = None


def _get_openrouter_client(timeout: float) -> httpx.Client:
    global _openrouter_client
    if _openrouter_client is None or _openrouter_client.is_closed:
        _openrouter_client = httpx.Client(timeout=timeout)
    return _openrouter_client


def _get_gemini_client(timeout: float) -> httpx.Client:
    global _gemini_client
    if _gemini_client is None or _gemini_client.is_closed:
        _gemini_client = httpx.Client(timeout=timeout)
    return _gemini_client


def call_openrouter(api_key: str, model: str, messages: list, timeout: float = 120) -> str:
    """Send messages to OpenRouter chat completions endpoint.

    Returns the assistant's response text, or raises on unrecoverable error.
    """
    if not api_key:
        return ""
    normalized_model = model.strip() if isinstance(model, str) and model.strip() else _DEFAULT_OPENROUTER_MODEL
    payload = {
        "model": normalized_model,
        "messages": messages,
        "temperature": 0.0,
    }
    headers = {**_OPENROUTER_HEADERS_BASE, "Authorization": f"Bearer {api_key}"}
    last_error = None
    c = _get_openrouter_client(timeout)
    for attempt in range(4):
        try:
            r = c.post(_OPENROUTER_ENDPOINT, json=payload, headers=headers)
            if r.status_code in (429, 503) and attempt < 3:
                retry_after = r.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after else float(2 ** attempt)
                except Exception:
                    wait_s = float(2 ** attempt)
                time.sleep(min(wait_s, 10.0))
                continue
            r.raise_for_status()
            j = r.json()
            choices = j.get("choices") if isinstance(j, dict) else None
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                txt = msg.get("content", "")
                if isinstance(txt, str):
                    return txt.strip()
            return ""
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise last_error
    return ""


def call_gemini(api_key: str, model: str, prompt_text: str, timeout: float = 120) -> str:
    """Send a prompt to the Gemini Generative Language REST API.

    Returns the generated text, or raises on unrecoverable error.
    """
    if not api_key:
        return ""
    normalized_model = model.strip() if isinstance(model, str) and model.strip() else _DEFAULT_GEMINI_MODEL
    if normalized_model.startswith("models/"):
        normalized_model = normalized_model.split("/", 1)[1]
    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent"
    last_error = None
    c = _get_gemini_client(timeout)
    for attempt in range(4):
        try:
            r = c.post(
                endpoint + f"?key={api_key}",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code in (429, 503) and attempt < 3:
                retry_after = r.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after else float(2 ** attempt)
                except Exception:
                    wait_s = float(2 ** attempt)
                time.sleep(min(wait_s, 10.0))
                continue
            r.raise_for_status()
            j = r.json()
            text_parts = []

            def _collect(node):
                if isinstance(node, dict):
                    t = node.get("text")
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t.strip())
                    for v in node.values():
                        _collect(v)
                elif isinstance(node, list):
                    for it in node:
                        _collect(it)

            _collect(j.get("candidates") if isinstance(j, dict) else j)
            if text_parts:
                return "\n".join(text_parts).strip()
            return ""
        except Exception as e:
            last_error = e
    if last_error is not None:
        raise last_error
    return ""
