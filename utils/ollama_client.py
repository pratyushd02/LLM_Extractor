"""Thin HTTP client for the Ollama chat API."""

from __future__ import annotations

import json
import re
import sys
import ast
from typing import Any

import requests

from config import OllamaConfig


def chat(
    prompt: str,
    ollama: OllamaConfig,
    system: str | None = None,
    *,
    verbose: bool = False,
) -> str:
    base = ollama.host.rstrip("/")
    url = f"{base}/api/chat"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": ollama.model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": ollama.temperature},
    }
    r = requests.post(url, json=payload, timeout=ollama.timeout_seconds)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content", "") or ""
    if verbose:
        print(content, file=sys.stderr)
    return content


def parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON from an LLM response with tolerant fallbacks."""
    text = raw.strip()

    # Common case: fenced markdown JSON.
    fence = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()

    # Some models emit bare unicode symbols for `symbol` values (e.g. "symbol": ‡).
    text = re.sub(
        r'("symbol"\s*:\s*)([^"\[\{\],\n][^,\n]*)(\s*[,}])',
        lambda m: f'{m.group(1)}{json.dumps(m.group(2).strip())}{m.group(3)}',
        text,
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If the model added commentary, extract the first JSON-looking container.
    candidates = re.findall(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Final fallback for Python-style dict literals from some models.
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise json.JSONDecodeError("Unable to parse LLM response as JSON", text, 0) from exc

    if isinstance(parsed, dict):
        return parsed
    raise json.JSONDecodeError("Parsed value is not a JSON object", text, 0)
