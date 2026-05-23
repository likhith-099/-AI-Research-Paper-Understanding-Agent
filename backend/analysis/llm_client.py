import os
import re
import time

import requests


API_KEY_ERROR_MESSAGE = (
    "Analysis features require a Groq API key. "
    "Set GROQ_API_KEY (or groq_api_key)."
)

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "400"))
GROQ_MAX_PROMPT_CHARS = int(os.getenv("GROQ_MAX_PROMPT_CHARS", "12000"))
GROQ_RETRIES = int(os.getenv("GROQ_RETRIES", "2"))
GROQ_RETRY_FALLBACK_SECONDS = float(os.getenv("GROQ_RETRY_FALLBACK_SECONDS", "8"))


def _resolve_api_key():
    return os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")


def _build_headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _truncate_prompt(prompt: str) -> str:
    if len(prompt) <= GROQ_MAX_PROMPT_CHARS:
        return prompt
    return prompt[:GROQ_MAX_PROMPT_CHARS] + "\n\n[Context truncated to fit API limits.]"


def _extract_retry_seconds(error_text: str) -> float:
    match = re.search(r"try again in ([0-9.]+)s", error_text.lower())
    if match:
        try:
            return max(float(match.group(1)), 1.0)
        except ValueError:
            return GROQ_RETRY_FALLBACK_SECONDS
    return GROQ_RETRY_FALLBACK_SECONDS


def generate_with_claude(prompt, max_tokens=None):
    api_key = _resolve_api_key()
    if not api_key:
        raise ValueError(API_KEY_ERROR_MESSAGE)

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Provide concise, complete answers. Do not end mid-sentence.",
            },
            {"role": "user", "content": _truncate_prompt(prompt)},
        ],
        "max_tokens": max_tokens or GROQ_MAX_TOKENS,
    }

    attempts = GROQ_RETRIES + 1
    response = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                f"{GROQ_BASE_URL.rstrip('/')}/chat/completions",
                headers=_build_headers(api_key),
                json=payload,
                timeout=120,
            )
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise RuntimeError(f"Groq API request failed: {exc}") from exc
            time.sleep(GROQ_RETRY_FALLBACK_SECONDS)
            continue

        if response.status_code == 429 and attempt < attempts - 1:
            wait_seconds = _extract_retry_seconds(response.text)
            time.sleep(wait_seconds)
            continue
        break

    if response.status_code >= 400:
        detail = response.text
        if "credit" in detail.lower() or "insufficient" in detail.lower() or "quota" in detail.lower():
            raise RuntimeError("Groq API credits/quota are insufficient. Please check your Groq billing.")
        if response.status_code == 429:
            raise RuntimeError(
                "Groq rate limit exceeded. Please retry shortly or reduce request size."
            )
        raise RuntimeError(f"Groq API request failed ({response.status_code}): {detail}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Groq API returned no choices.")

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if not content:
        raise RuntimeError("Groq API returned an empty response.")

    return content.strip()
