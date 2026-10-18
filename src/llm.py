"""Thin wrapper around the LLMs so the rest of the code never depends on a provider.

Two providers, chosen by the model name: Ollama (local, e.g. "qwen2.5:7b") and the Claude API
(names starting with "claude-", e.g. "claude-haiku-5-5"; needs ANTHROPIC_API_KEY in .env).
The rest of the code always speaks the Ollama message format; this file translates for Claude.
Every call returns the text plus what the evaluation needs: tokens and latency.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import os
import time

import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)  # [{"name": ..., "arguments": {...}}]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    # Claude only: the exact content blocks, thinking included. They must go back unchanged in the
    # conversation (as "raw" in the assistant message), or the API may reject the next request.
    raw: list | None = None


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.0,
         model: str | None = None) -> LLMResponse:
    """Send a conversation to the LLM. temperature=0 makes the local evaluation reproducible.
    model: to use another model than MODEL for this call (e.g. a different model for SQL)."""
    model = model or MODEL
    start = time.perf_counter()
    if model.startswith("claude-"):
        response = _chat_claude(model, messages, tools)
    else:
        response = _chat_ollama(model, messages, tools, temperature)
    response.latency_s = time.perf_counter() - start
    return response


def _chat_ollama(model: str, messages: list[dict], tools: list[dict] | None, temperature: float) -> LLMResponse:
    messages = [{k: v for k, v in m.items() if k != "raw"} for m in messages]  # Ollama does not know "raw"
    response = ollama.chat(model=model, messages=messages, tools=tools, options={"temperature": temperature})
    message = response["message"]
    return LLMResponse(
        text=message.get("content") or "",
        tool_calls=[{"name": c["function"]["name"], "arguments": dict(c["function"]["arguments"])}
                    for c in (message.get("tool_calls") or [])],
        input_tokens=response.get("prompt_eval_count") or 0,
        output_tokens=response.get("eval_count") or 0,
    )


@lru_cache(maxsize=1)
def _claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def _chat_claude(model: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    system, history = _to_claude_messages(messages)
    options = {}
    if system:
        options["system"] = system
    if tools:
        options["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                             "input_schema": t["function"]["parameters"]} for t in tools]
    # No temperature: Claude Haiku 5.5 rejects non-default sampling parameters.
    # Thinking is left on (adaptive, the default): the model thinks before hard queries.
    response = _claude_client().messages.create(model=model, max_tokens=16000, messages=history, **options)
    return LLMResponse(
        text="".join(b.text for b in response.content if b.type == "text"),
        tool_calls=[{"name": b.name, "arguments": b.input} for b in response.content if b.type == "tool_use"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        raw=response.content,
    )


def _to_claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Ollama-format conversation -> (system text, messages in the Claude format).

    Differences: the system prompt is a separate parameter; Claude's own turns go back as their raw
    blocks; tool results are "tool_result" blocks in a user message, linked to the call by its id.
    """
    system, history, call_ids = "", [], []
    for m in messages:
        if m["role"] == "system":
            system += m["content"] + "\n"
        elif m["role"] == "tool":
            result = {"type": "tool_result", "tool_use_id": call_ids.pop(0), "content": m["content"]}
            if history[-1]["role"] == "user" and isinstance(history[-1]["content"], list):
                history[-1]["content"].append(result)  # all the results of one turn in one message
            else:
                history.append({"role": "user", "content": [result]})
        elif m["role"] == "assistant" and m.get("raw"):
            history.append({"role": "assistant", "content": m["raw"]})
            call_ids += [b.id for b in m["raw"] if b.type == "tool_use"]
        else:
            history.append({"role": m["role"], "content": m["content"]})
    return system.strip(), history
