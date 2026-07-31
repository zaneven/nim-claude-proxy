"""Stream Transformer converting OpenAI SSE chunks into Anthropic Messages SSE events."""

import json
import uuid
from collections.abc import AsyncIterator


async def format_anthropic_sse_stream(raw_stream: AsyncIterator[Any], model: str) -> AsyncIterator[str]:
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"
    
    # 1. 发送 message_start
    message_start_event = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start_event)}\n\n"

    # 2. 发送 content_block_start (text)
    block_start_event = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    }
    yield f"event: content_block_start\ndata: {json.dumps(block_start_event)}\n\n"

    has_tool_calls = False
    stop_reason = "end_turn"

    async for chunk in raw_stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason:
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
                has_tool_calls = True
            elif finish_reason == "length":
                stop_reason = "max_tokens"

        delta = getattr(choice, "delta", None)
        if not delta:
            continue

        # 处理文本增量
        content = getattr(delta, "content", None)
        if content:
            delta_event = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": content}
            }
            yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

        # 处理 Tool Call 增量
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            has_tool_calls = True
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn and getattr(fn, "name", None):
                    # 开始 Tool Use block
                    tool_start_event = {
                        "type": "content_block_start",
                        "index": tc.index + 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": getattr(tc, "id", f"call_{uuid.uuid4().hex[:8]}"),
                            "name": fn.name,
                            "input": {}
                        }
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(tool_start_event)}\n\n"

                args = getattr(fn, "arguments", None) if fn else None
                if args:
                    tool_delta_event = {
                        "type": "content_block_delta",
                        "index": getattr(tc, "index", 0) + 1,
                        "delta": {"type": "input_json_delta", "partial_json": args}
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(tool_delta_event)}\n\n"

    # 3. 发送 content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # 4. 发送 message_delta 与 message_stop
    msg_delta_event = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 100}
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta_event)}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
