"""Shared OpenAI-compatible chat request, used by the harness (run.py) and the smoke test.

Works against Ollama's /v1 and llama.cpp's llama-server. One request = one prompt, no history.
"""

import json
import time

import httpx

from .prompts import ANSWER_SCHEMA, LETTERS


def parse_answer(raw: str) -> str | None:
    """The one valid letter in `{"answer": "<letter>"}`, else None (invalid; counts as wrong)."""
    try:
        answer = json.loads(raw)["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return answer if isinstance(answer, str) and len(answer) == 1 and answer in LETTERS else None


def ask(client: httpx.Client, base_url: str, model: str, prompt: str, seed: int,
        max_tokens: int = 32, reasoning_effort: str | None = None) -> dict:
    t0 = time.perf_counter()
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    r = client.post(f"{base_url}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        # Thinking off for every model (where the model allows it). Templates that support it (Sarvam, Qwen3) read this;
        # llama-server must also run with --reasoning-budget 0, since Sarvam keeps thinking otherwise.
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA}},
        **extra,
    })
    r.raise_for_status()
    body = r.json()
    raw = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage") or {}
    return {"raw": raw, "answer": parse_answer(raw), "seconds": round(time.perf_counter() - t0, 2),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
