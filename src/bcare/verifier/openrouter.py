"""Minimal OpenRouter chat client for Fixed-Llama PBCE / PACE / EDCV."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = "https://openrouter.ai/api/v1/chat/completions"


def chat_completion(
    *,
    model: str,
    user: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None = None,
    base_url: str | None = None,
    system: str | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 6,
) -> dict[str, Any]:
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example → .env and add a key, "
            "or export OPENROUTER_API_KEY."
        )
    url = base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_URL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return {
                "content": content,
                "raw": body,
                "model": model,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
            }
        except urllib.error.HTTPError as e:
            last_err = e
            code = getattr(e, "code", None)
            if code in (429, 500, 502, 503, 504):
                time.sleep(min(2**attempt, 30))
                continue
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")
