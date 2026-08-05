"""Server-side agentic loop that transparently resolves local web tools.

When the upstream NIM model emits a ``web_search`` / ``web_fetch`` tool call,
the proxy executes it locally (DuckDuckGo / Tavily / Doubao / fetch) and feeds
the result back into the conversation before returning to the client. Non-web
tool calls are always surfaced to the client untouched.
"""

import json
import uuid
from typing import Any

from loguru import logger

from nim_proxy.config.settings import settings
from nim_proxy.fcc_port.core.anthropic.thinking import ContentType, ThinkTagParser
from nim_proxy.provider.schema import unalias_tool_call_arguments
from nim_proxy.web_tools.executor import handle_local_web_tool, is_web_tool

_MAX_WEB_TOOL_ROUNDS = 5


def _parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        logger.debug(f"Failed to parse tool call arguments: {exc}")
        return {}


def _split_tool_calls(tool_calls: list[Any]) -> tuple[list[Any], list[Any]]:
    """Partition tool calls into locally-resolvable web calls and everything else."""
    web_calls: list[Any] = []
    non_web: list[Any] = []
    for tc in tool_calls:
        name = getattr(getattr(tc, "function", None), "name", None) or ""
        if is_web_tool(name):
            enabled = (
                name in ("web_search", "search") and settings.enable_web_search
            ) or (name in ("web_fetch", "fetch") and settings.enable_web_fetch)
            (web_calls if enabled else non_web).append(tc)
        else:
            non_web.append(tc)
    return web_calls, non_web


async def run_with_web_tools(execute_chat: Any, body: dict[str, Any], max_rounds: int = _MAX_WEB_TOOL_ROUNDS) -> Any:
    """Execute chat, resolving web tools server-side, capped at ``max_rounds``.

    ``execute_chat`` is a callable that takes a body dict and returns the
    upstream OpenAI-format response. This accepts either the fcc_port provider's
    ``execute_chat_non_streaming`` (with admission + retry) or a thin NimClient wrapper.

    Returns the final upstream response (text, or non-web tool calls for the client).
    """
    web_available = settings.enable_web_search or settings.enable_web_fetch
    if not web_available:
        return await execute_chat(body)

    response = await execute_chat(body)

    for _ in range(max_rounds):
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            break

        web_calls, non_web = _split_tool_calls(tool_calls)
        # 含无法本地处理的工具调用 -> 交还客户端，停止循环
        if non_web:
            break
        if not web_calls:
            break

        # 回填助手消息（含 tool_calls）与本地执行结果，继续对话
        body["messages"].append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })

        for tc in web_calls:
            tool_input = _parse_args(tc.function.arguments)
            try:
                result = await handle_local_web_tool(tc.function.name, tool_input)
            except Exception as err:
                logger.warning(f"Local web tool '{tc.function.name}' failed: {err}")
                result = f"[web tool error] {err}"
            body["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result or "",
            })

        response = await execute_chat(body)

    return response


def _append_content_chunk(blocks: list[dict[str, Any]], chunk_type: ContentType, content: str) -> None:
    """Append a parsed thinking/text chunk, merging adjacent same-type blocks."""
    if not content:
        return
    block_type = "thinking" if chunk_type is ContentType.THINKING else "text"
    key = "thinking" if block_type == "thinking" else "text"
    if blocks and blocks[-1].get("type") == block_type:
        blocks[-1][key] += content
    else:
        blocks.append({"type": block_type, key: content})


def openai_final_to_anthropic_message(response: Any, model: str) -> dict[str, Any]:
    """Convert a final (non-streaming) OpenAI chat response into an Anthropic Message.

    Used when local web tools were resolved server-side: the upstream response
    arrives as a complete OpenAI payload instead of the engine's SSE stream, so
    it is converted here with the same semantics as the streaming engine
    (native reasoning fields, ``<think>`` tag parsing, tool argument
    un-aliasing, stop-reason mapping, usage passthrough).
    """
    choice = response.choices[0]
    msg = choice.message

    content_blocks: list[dict[str, Any]] = []

    # 1. Native reasoning fields emitted by the provider
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})

    # 2. Parse <think> tags from text content (same parser as the engine)
    if msg.content:
        parser = ThinkTagParser()
        for chunk in parser.feed(msg.content):
            _append_content_chunk(content_blocks, chunk.type, chunk.content)
        remaining = parser.flush()
        if remaining is not None:
            _append_content_chunk(content_blocks, remaining.type, remaining.content)

    # 3. Tool calls (with NIM argument-name un-aliasing)
    for tc in getattr(msg, "tool_calls", None) or []:
        args = _parse_args(getattr(tc.function, "arguments", None))
        clean_args = unalias_tool_call_arguments(args)
        tc_id = getattr(tc, "id", None) or f"call_nim_{uuid.uuid4().hex[:10]}"
        content_blocks.append({
            "type": "tool_use",
            "id": str(tc_id),
            "name": tc.function.name,
            "input": clean_args if isinstance(clean_args, dict) else args,
        })

    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_message_as_anthropic_sse(message: dict[str, Any]):
    """Replay a complete Anthropic Message as a standard SSE event sequence.

    Used by the streaming path when web tools were resolved server-side (the
    upstream stream cannot be intercepted mid-flight, so the final aggregated
    message is replayed). Event shapes mirror the streaming engine's ledger
    exactly: message_start -> content_block_* -> message_delta -> message_stop.
    """
    usage = message.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": message.get("model"),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1},
        },
    })

    for index, block in enumerate(message.get("content") or []):
        block_type = block.get("type")
        if block_type == "thinking":
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            })
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")},
            })
        elif block_type == "text":
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            })
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": block.get("text", "")},
            })
        elif block_type == "tool_use":
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": {},
                },
            })
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })
        else:
            continue
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        })

    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": message.get("stop_reason") or "end_turn",
            "stop_sequence": None,
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    })
    yield _sse_event("message_stop", {"type": "message_stop"})
