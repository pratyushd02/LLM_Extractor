from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_structured(text: str, model: type[T]) -> T:
    """
    Parse `text` into an instance of `model` using an LLM.

    Implement this with your provider (structured outputs, tool calls, or JSON mode).
    """
    raise NotImplementedError(
        "Wire extract_structured to your LLM client; `model` is the Pydantic schema for output."
    )
