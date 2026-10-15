"""Thin wrapper around the LLM so the rest of the code never depends on a provider.

Every call returns the text plus what the evaluation needs: tokens and latency.
To switch to an API model later, only this file changes.
"""

from dataclasses import dataclass, field
import os
import time

import ollama

MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)  # [{"name": ..., "arguments": {...}}]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0) -> LLMResponse:
    """Send a conversation to the LLM. temperature=0 makes the evaluation reproducible."""
    start = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=tools,
        options={"temperature": temperature},
    )
    latency = time.perf_counter() - start

    message = response["message"]
    tool_calls = [
        {"name": call["function"]["name"], "arguments": dict(call["function"]["arguments"])}
        for call in (message.get("tool_calls") or [])
    ]
    return LLMResponse(
        text=message.get("content") or "",
        tool_calls=tool_calls,
        input_tokens=response.get("prompt_eval_count") or 0,
        output_tokens=response.get("eval_count") or 0,
        latency_s=latency,
    )
