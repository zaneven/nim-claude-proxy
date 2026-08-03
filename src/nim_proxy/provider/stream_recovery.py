"""Stream recovery helpers for midstream error handling.

Implements midstream text continuation and tool JSON repair body builders
— adapted from freeclaudecode's recovery architecture.

Note: the SSE holdback buffer lives in ``nim_proxy.api.sse.SseHoldback``.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from loguru import logger

# ── Continuation / tool repair body builders ──────────────────────────


def clone_body_without_stream(body: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy body without the stream flag."""
    cloned = copy.deepcopy(body)
    cloned.pop("stream", None)
    return cloned


def make_text_continuation_body(original_body: dict[str, Any], partial_text: str) -> dict[str, Any] | None:
    """Build a continuation body that asks the model to continue from *partial_text*.

    Appends an assistant message with the partial text so the model continues
    naturally from that point.
    """
    if not partial_text.strip():
        return None

    body = clone_body_without_stream(original_body)
    messages = body["messages"]

    # Append the partial text as the last assistant message so the model can
    # continue from where it left off
    messages.append({"role": "assistant", "content": partial_text})

    # Strip any existing tool definitons — we want text only
    body.pop("tools", None)
    body.pop("tool_choice", None)
    body["max_tokens"] = body.get("max_tokens", 4096)

    return body


def make_tool_repair_body(
    original_body: dict[str, Any],
    tool_name: str,
    prefix: str,
    input_schema: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a repair request to complete a partially streamed tool call.

    Appends an assistant message with a partial tool_call + the JSON prefix.
    The model should output the rest of the JSON.
    """
    if not tool_name or not prefix:
        return None

    body = clone_body_without_stream(original_body)
    messages = body["messages"]

    # Append assistant message with the partial tool call
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_recovery",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": prefix,
                },
            }
        ],
    })

    body.pop("tools", None)
    body.pop("tool_choice", None)
    body["max_tokens"] = body.get("max_tokens", 4096)

    return body


def continuation_suffix(partial: str, full: str) -> str:
    """Return the portion of ``full`` that extends ``partial``.

    Used to extract only the new text from a continuation response.
    """
    if not partial:
        return full
    if full.startswith(partial):
        return full[len(partial):]
    # Best-effort: return everything
    return full


def accept_tool_json_repair(
    prefix: str,
    completion: str,
    tool_name: str,
    schemas: dict[str, Any] | None = None,
) -> str | None:
    """Try to extract a valid JSON suffix from a tool recovery completion.

    Attempts to parse prefix + completion as JSON. On success,
    returns the suffix (non-prefix portion) of the completion.
    Returns None if no valid JSON could be extracted.
    """
    candidate = prefix + completion
    try:
        json.loads(candidate)
        # Full parse succeeded: the suffix is just completion
        return completion
    except json.JSONDecodeError:
        pass

    # Try truncating from the right to find valid JSON
    for i in range(len(candidate) - 1, len(prefix) - 1, -1):
        try:
            json.loads(candidate[:i + 1])
            # Valid JSON found up to position i
            accepted_suffix = candidate[:i + 1][len(prefix):]
            if accepted_suffix:
                logger.debug(
                    "Recovered tool JSON for '{}' by trimming trailing content",
                    tool_name,
                )
                return accepted_suffix
            return None
        except json.JSONDecodeError:
            continue

    return None