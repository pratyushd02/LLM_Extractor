"""Thin HTTP client for the Ollama chat API."""

from __future__ import annotations

import json
import re
import sys
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
    """Parse JSON from an LLM response, stripping optional markdown fences."""
    text = raw.strip()
    m = re.match(
        r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        text = m.group(1).strip()
    return json.loads(text)
