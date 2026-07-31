"""Normalize native MiniMax/Nemotron tool markup exposed by NVIDIA NIM."""

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any


_NAMESPACE = "]<]minimax[>["
_TOOL_BLOCK_START = f"{_NAMESPACE}<tool_call>"
_TOOL_BLOCK_END = f"{_NAMESPACE}</tool_call>"
_INVOKE_START = f"{_NAMESPACE}<invoke"
_INVOKE_END = f"{_NAMESPACE}</invoke>"


class FramingMode(StrEnum):
    TEXT = "text"
    TOOL_BLOCK = "tool_block"
    AFTER_TOOL_BLOCK = "after_tool_block"
    FINISHED = "finished"


class MiniMaxM3ToolFramer:
    """Separate visible text from namespace-delimited native tool blocks."""

    def __init__(self) -> None:
        self._mode = FramingMode.TEXT
        self._buffer = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        self._buffer += chunk

        prefix_out = ""
        block_out = ""

        if self._mode == FramingMode.TEXT:
            if _TOOL_BLOCK_START in self._buffer:
                prefix_out, _, rest = self._buffer.partition(_TOOL_BLOCK_START)
                self._buffer = rest
                self._mode = FramingMode.TOOL_BLOCK
            else:
                if not _TOOL_BLOCK_START.startswith(self._buffer) and len(self._buffer) >= len(_TOOL_BLOCK_START):
                    prefix_out = self._buffer
                    self._buffer = ""
                return prefix_out, ""

        if self._mode == FramingMode.TOOL_BLOCK:
            if _TOOL_BLOCK_END in self._buffer:
                block_out, _, rest = self._buffer.partition(_TOOL_BLOCK_END)
                self._buffer = rest
                self._mode = FramingMode.AFTER_TOOL_BLOCK
                return prefix_out, block_out
            return prefix_out, ""

        if self._mode == FramingMode.AFTER_TOOL_BLOCK:
            prefix_out += self._buffer
            self._buffer = ""
            return prefix_out, ""

        return prefix_out, block_out

    def flush(self) -> str:
        out = self._buffer
        self._buffer = ""
        return out


def parse_minimax_tool_calls(block_text: str) -> list[dict[str, Any]]:
    """Parse XML-like invoke blocks into standard OpenAI tool calls structure."""
    tool_calls = []
    current = block_text
    index = 0

    while _INVOKE_START in current:
        _, _, rest = current.partition(_INVOKE_START)
        if 'name="' not in rest:
            break
        
        _, _, rest = rest.partition('name="')
        name, _, rest = rest.partition('"')
        
        if _INVOKE_END not in rest:
            break

        args_text, _, current = rest.partition(_INVOKE_END)
        args_dict = {}

        param_parts = args_text.split('<parameter name="')
        for part in param_parts[1:]:
            p_name, _, p_rest = part.partition('">')
            p_val, _, _ = p_rest.partition("</parameter>")
            try:
                args_dict[p_name] = json.loads(p_val)
            except Exception:
                args_dict[p_name] = p_val

        tool_calls.append({
            "index": index,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args_dict, ensure_ascii=False),
            }
        })
        index += 1

    return tool_calls


async def normalize_nim_native_tool_stream(raw_stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Async generator wrapping raw OpenAI stream to repair native tool leaks."""
    framer = MiniMaxM3ToolFramer()
    
    async for chunk in raw_stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            yield chunk
            continue

        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None

        if not content:
            yield chunk
            continue

        text_part, tool_block = framer.feed(content)

        if text_part:
            delta.content = text_part
            yield chunk
        elif tool_block:
            tool_calls = parse_minimax_tool_calls(tool_block)
            if tool_calls:
                delta.content = None
                delta.tool_calls = [
                    SimpleNamespace(
                        index=tc["index"],
                        id=tc["id"],
                        type=tc["type"],
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        )
                    )
                    for tc in tool_calls
                ]
                yield chunk
        else:
            pass

    remaining = framer.flush()
    if remaining:
        dummy_chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=remaining, tool_calls=None)
                )
            ]
        )
        yield dummy_chunk
