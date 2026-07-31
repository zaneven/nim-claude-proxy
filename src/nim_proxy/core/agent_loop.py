"""Server-side agentic loop that transparently resolves local web tools.

When the upstream NIM model emits a ``web_search`` / ``web_fetch`` tool call,
the proxy executes it locally (DuckDuckGo / Tavily / Doubao / fetch) and feeds
the result back into the conversation before returning to the client. Non-web
tool calls are always surfaced to the client untouched.
"""

import json
from typing import Any
from types import SimpleNamespace

from loguru import logger

from nim_proxy.config.settings import settings
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


async def run_with_web_tools(client: Any, body: dict[str, Any], max_rounds: int = _MAX_WEB_TOOL_ROUNDS) -> Any:
    """Execute chat, resolving web tools server-side, capped at ``max_rounds``.

    Returns the final upstream response (text, or non-web tool calls for the client).
    """
    web_available = settings.enable_web_search or settings.enable_web_fetch
    if not web_available:
        return await client.execute_chat(body)

    response = await client.execute_chat(body)

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

        response = await client.execute_chat(body)

    return response


async def stream_text_as_anthropic_sse(text: str, model: str):
    """Yield a minimal Anthropic SSE sequence for a single final text blob.

    Used by the streaming path when web tools must be resolved server-side
    (the upstream stream cannot be intercepted mid-flight).
    """
    fake_chunk = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=text,
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason="stop",
            )
        ],
    )
    from nim_proxy.api.sse import format_anthropic_sse_stream

    async for event in format_anthropic_sse_stream(_single(fake_chunk), model):
        yield event


async def _single(chunk):
    yield chunk
