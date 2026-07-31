"""OpenAI Responses API <-> Chat Completions conversion helpers.

The proxy exposes ``/v1/responses`` for tools such as Codex. We convert the
Responses request shape into a Chat Completions body, execute it against NIM,
and convert the result back into a minimal Responses-shaped payload.
"""

import json
import time
import uuid
from typing import Any


def _text_of(content: Any) -> str:
    """Flatten Responses content (str or list of typed blocks) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("input_text", "output_text"):
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def _response_input_item_to_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a single Responses input item to one or more Chat Completions messages."""
    if item.get("type") == "function_call_output":
        return [{
            "role": "tool",
            "tool_call_id": item.get("call_id"),
            "content": item.get("output", ""),
        }]

    role = item.get("role")
    content = item.get("content")

    if role == "tool":
        return [{
            "role": "tool",
            "tool_call_id": item.get("tool_call_id") or item.get("call_id"),
            "content": _text_of(content),
        }]

    if role in ("user", "assistant", "system"):
        if role == "assistant" and isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "output_text":
                        text_parts.append(c.get("text", ""))
                    elif c.get("type") == "function_call":
                        args = c.get("arguments", {})
                        if isinstance(args, dict):
                            args = json.dumps(args, ensure_ascii=False)
                        tool_calls.append({
                            "id": c.get("call_id"),
                            "type": "function",
                            "function": {"name": c.get("name"), "arguments": args or ""},
                        })
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return [msg]
        return [{"role": role, "content": _text_of(content)}]

    return []


def responses_request_to_openai_body(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Responses API request into a Chat Completions payload."""
    model = body.get("model", "")
    messages: list[dict[str, Any]] = []

    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                messages.extend(_response_input_item_to_messages(item))

    out: dict[str, Any] = {"model": model, "messages": messages}

    raw_tools = body.get("tools")
    if isinstance(raw_tools, list):
        # 仅转发可被 NIM 理解的 function 工具；预览类工具（web_search_preview 等）忽略
        fn_tools = [t for t in raw_tools if isinstance(t, dict) and t.get("type") == "function"]
        if fn_tools:
            out["tools"] = fn_tools

    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "top_p" in body:
        out["top_p"] = body["top_p"]
    if "max_output_tokens" in body:
        out["max_tokens"] = body["max_output_tokens"]
    if body.get("stream"):
        out["stream"] = True

    return out


def openai_response_to_responses(oai_response: Any, model: str) -> dict[str, Any]:
    """Convert a Chat Completions response into a minimal Responses-shaped payload."""
    choice = oai_response.choices[0]
    msg = choice.message

    output: list[dict[str, Any]] = []
    text = getattr(msg, "content", None) or ""
    if text:
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })

    for tc in (getattr(msg, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        output.append({
            "type": "function_call",
            "call_id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}"),
            "name": getattr(fn, "name", "") if fn else "",
            "arguments": getattr(fn, "arguments", "") if fn else "",
        })

    usage = getattr(oai_response, "usage", None)
    out_usage = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }

    return {
        "id": f"resp_{uuid.uuid4().hex[:16]}",
        "object": "response",
        "created": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": out_usage,
    }
