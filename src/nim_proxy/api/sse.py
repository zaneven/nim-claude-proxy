from __future__ import annotations

"""Stream Transformer: OpenAI SSE chunks → Anthropic Messages SSE events.

Implements proper Anthropic SSE protocol including:
- Block lifecycle management (start → delta → stop) for text, thinking, and tool_use
- thinking tag streaming parser for models that emit reasoning in content
- reasoning_content / reasoning field support for native OpenAI-style reasoning
- Proper stop_reason inference (tool_use when tool blocks are emitted)
- Incremental block index allocation (thinking → text → tool_use)
- Stream recovery: early transparent retry holdback + midstream error injection
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format one Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── SSE Holdback Buffer for Early Transparent Retries ──────────────────


class SseHoldback:
    """Buffers SSE events until the first ``content_block_start`` is emitted.

    Events held before a content_block_start are silently discardable, enabling
    **early transparent retries**: if the upstream stream fails before any
    content block has started, we can retry without the client seeing anything.

    Once the first ``content_block_start`` passes through, the holdback is
    "committed" and all subsequent events pass through immediately.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._committed = False

    def push(self, event: str) -> list[str]:
        """Feed one SSE event. Returns events that should now be yielded."""
        if self._committed:
            return [event]
        if '"content_block_start"' in event:
            self._committed = True
            committed = list(self._buffer)
            committed.append(event)
            self._buffer.clear()
            return committed
        self._buffer.append(event)
        return []

    def flush(self) -> list[str]:
        """Flush all remaining buffered events on stream success."""
        committed = list(self._buffer)
        self._buffer.clear()
        return committed

    def discard(self) -> None:
        """Discard all buffered events on early retry."""
        self._buffer.clear()

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def has_buffered(self) -> bool:
        return len(self._buffer) > 0


# ── Inline error event helpers ─────────────────────────────────────────


def sse_error_event(message: str) -> str:
    """Anthropic-compatible SSE error event for midstream injection."""
    return _sse_event("error", {
        "type": "error",
        "error": {"type": "api_error", "message": message},
    })


def _error_cleanup_events(
    ledger: "_StreamLedger",
    error_message: str,
) -> list[str]:
    """Close open blocks + inject error SSE event for midstream failures."""
    events: list[str] = []
    events.extend(ledger.close_all_open_blocks())
    events.append(sse_error_event(error_message))
    return events


# ── Stop reason mapping ────────────────────────────────────────────────

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


# ── Think tag parser ───────────────────────────────────────────────────


class ThinkTagParser:
    """Streaming parser for `` thinking... response`` tags embedded in content deltas.

    Models like DeepSeek emit reasoning wrapped in thinking tags. This parser
    separates thinking content from regular text at chunk boundaries.
    """

    OPEN_TAG = " thinking"
    CLOSE_TAG = " response"

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_think: bool = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Feed a content chunk and return list of (content_type, content) pairs.

        content_type is either "thinking" or "text".
        """
        self._buffer += text
        results: list[tuple[str, str]] = []

        while self._buffer:
            prev_len = len(self._buffer)
            if not self._in_think:
                chunk = self._parse_outside()
            else:
                chunk = self._parse_inside()

            if chunk:
                results.append(chunk)
            elif len(self._buffer) == prev_len:
                break

        return results

    def flush(self) -> tuple[str, str] | None:
        """Flush remaining buffer."""
        if not self._buffer:
            return None
        content_type = "thinking" if self._in_think else "text"
        out = self._buffer
        self._buffer = ""
        return (content_type, out)

    def _parse_outside(self) -> tuple[str, str] | None:
        think_start = self._buffer.find(self.OPEN_TAG)
        orphan_close = self._buffer.find(self.CLOSE_TAG)

        # Handle orphan close tag before open tag
        if orphan_close != -1 and (think_start == -1 or orphan_close < think_start):
            pre = self._buffer[:orphan_close]
            self._buffer = self._buffer[orphan_close + len(self.CLOSE_TAG):]
            if pre:
                return ("text", pre)
            return None

        if think_start == -1:
            # Check for partial tag at end
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1:
                potential = self._buffer[last_bracket:]
                if (len(potential) < len(self.OPEN_TAG) and self.OPEN_TAG.startswith(potential)) or \
                   (len(potential) < len(self.CLOSE_TAG) and self.CLOSE_TAG.startswith(potential)):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ("text", emit)
                    return None

            emit = self._buffer
            self._buffer = ""
            if emit:
                return ("text", emit)
            return None

        pre = self._buffer[:think_start]
        self._buffer = self._buffer[think_start + len(self.OPEN_TAG):]
        self._in_think = True
        if pre:
            return ("text", pre)
        return None

    def _parse_inside(self) -> tuple[str, str] | None:
        close_pos = self._buffer.find(self.CLOSE_TAG)
        if close_pos == -1:
            # Check for partial close tag at end
            last_bracket = self._buffer.rfind("<")
            if last_bracket != -1:
                potential = self._buffer[last_bracket:]
                if len(potential) < len(self.CLOSE_TAG) and self.CLOSE_TAG.startswith(potential):
                    emit = self._buffer[:last_bracket]
                    self._buffer = self._buffer[last_bracket:]
                    if emit:
                        return ("thinking", emit)
                    return None

            emit = self._buffer
            self._buffer = ""
            if emit:
                return ("thinking", emit)
            return None

        thinking_content = self._buffer[:close_pos]
        self._buffer = self._buffer[close_pos + len(self.CLOSE_TAG):]
        self._in_think = False
        if thinking_content:
            return ("thinking", thinking_content)
        return None


# ── Tool block state ───────────────────────────────────────────────────


class _ToolBlockState:
    """Track one streamed tool call's lifecycle."""

    def __init__(self, tool_index: int, block_index: int, tool_id: str, name: str) -> None:
        self.tool_index = tool_index
        self.block_index = block_index
        self.tool_id = tool_id
        self.name = name
        self.contents: list[str] = []  # accumulated argument deltas
        self.started = True  # content_block_start has been emitted
        self.stopped = False  # content_block_stop has been emitted


# ── Stream Ledger ──────────────────────────────────────────────────────


class _StreamLedger:
    """Manage Anthropic SSE state: block indexes, open/close tracking."""

    def __init__(self, message_id: str, model: str) -> None:
        self.message_id = message_id
        self.model = model
        self._next_index = 0

        # Accumulated text/reasoning for recovery
        self.accumulated_text = ""
        self.accumulated_reasoning = ""

        # Thinking block state
        self.thinking_index = -1
        self.thinking_started = False

        # Text block state
        self.text_index = -1
        self.text_started = False

        # Tool blocks: keyed by OpenAI tool_call index
        self.tool_states: dict[int, _ToolBlockState] = {}

        self.has_tool_blocks = False

    def allocate_index(self) -> int:
        idx = self._next_index
        self._next_index += 1
        return idx

    # ── Message lifecycle ──

    def message_start(self) -> str:
        return _sse_event("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 1},
            },
        })

    def message_delta(self, stop_reason: str, output_tokens: int) -> str:
        return _sse_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

    def message_stop(self) -> str:
        return _sse_event("message_stop", {"type": "message_stop"})

    # ── Thinking block ──

    def ensure_thinking_block(self) -> list[str]:
        """Ensure thinking block is open, closing text if needed."""
        events: list[str] = []
        if self.text_started:
            events.extend(self.stop_text_block())
        if not self.thinking_started:
            self.thinking_index = self.allocate_index()
            self.thinking_started = True
            events.append(_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": self.thinking_index,
                "content_block": {"type": "thinking", "thinking": ""},
            }))
        return events

    def emit_thinking_delta(self, content: str) -> str:
        self.accumulated_reasoning += content
        return _sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.thinking_index,
            "delta": {"type": "thinking_delta", "thinking": content},
        })

    def stop_think_block(self) -> list[str]:
        if not self.thinking_started:
            return []
        self.thinking_started = False
        return [_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": self.thinking_index,
        })]

    # ── Text block ──

    def ensure_text_block(self) -> list[str]:
        """Ensure text block is open, closing thinking if needed."""
        events: list[str] = []
        if self.thinking_started:
            events.extend(self.stop_think_block())
        if not self.text_started:
            self.text_index = self.allocate_index()
            self.text_started = True
            events.append(_sse_event("content_block_start", {
                "type": "content_block_start",
                "index": self.text_index,
                "content_block": {"type": "text", "text": ""},
            }))
        return events

    def emit_text_delta(self, content: str) -> str:
        self.accumulated_text += content
        return _sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self.text_index,
            "delta": {"type": "text_delta", "text": content},
        })

    def stop_text_block(self) -> list[str]:
        if not self.text_started:
            return []
        self.text_started = False
        return [_sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": self.text_index,
        })]

    # ── Tool blocks ──

    def start_tool_block(self, tc_index: int, tool_id: str, name: str) -> str:
        block_idx = self.allocate_index()
        state = _ToolBlockState(tc_index, block_idx, tool_id, name)
        self.tool_states[tc_index] = state
        self.has_tool_blocks = True
        return _sse_event("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {
                "type": "tool_use",
                "id": tool_id,
                "name": name,
                "input": {},
            },
        })

    def emit_tool_delta(self, tc_index: int, partial_json: str) -> str:
        state = self.tool_states[tc_index]
        state.contents.append(partial_json)
        return _sse_event("content_block_delta", {
            "type": "content_block_delta",
            "index": state.block_index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })

    def stop_tool_block(self, tc_index: int) -> str:
        state = self.tool_states[tc_index]
        state.stopped = True
        return _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": state.block_index,
        })

    # ── Close all open blocks ──

    def close_content_blocks(self) -> list[str]:
        """Close thinking and text blocks (before emitting tool blocks)."""
        events: list[str] = []
        if self.thinking_started:
            events.extend(self.stop_think_block())
        if self.text_started:
            events.extend(self.stop_text_block())
        return events

    def close_all_open_blocks(self) -> list[str]:
        """Close all open blocks at stream end."""
        events: list[str] = []
        events.extend(self.close_content_blocks())
        for tc_index, state in self.tool_states.items():
            if state.started and not state.stopped:
                events.append(self.stop_tool_block(tc_index))
        return events

    def final_stop_reason(self, finish_reason: str | None) -> str:
        """Determine final stop_reason, aware of tool_use blocks."""
        if self.has_tool_blocks:
            return "tool_use"
        return STOP_REASON_MAP.get(finish_reason or "", "end_turn")

    def has_committed_output(self) -> bool:
        """Whether any content block has been emitted (text/thinking/tool)."""
        return self.text_index != -1 or self.thinking_index != -1 or self.has_tool_blocks

    def tool_block_snapshot(self) -> list[dict[str, Any]]:
        """Return a list of started tool blocks with their names and accumulated JSON."""
        result: list[dict[str, Any]] = []
        for tc_index, state in self.tool_states.items():
            if state.started:
                result.append({
                    "index": tc_index,
                    "tool_id": state.tool_id,
                    "name": state.name,
                    "content": "".join(state.contents),
                    "stopped": state.stopped,
                })
        return result

    def estimate_output_tokens(self) -> int:
        """Estimate output tokens from accumulated text and reasoning."""
        total_chars = len(self.accumulated_text) + len(self.accumulated_reasoning)
        for tc_index, state in self.tool_states.items():
            total_chars += sum(len(c) for c in state.contents)
        return max(1, total_chars // 4)  # rough heuristic: ~4 chars per token


# ── SSE Stream Formatter ───────────────────────────────────────────────


async def format_anthropic_sse_stream(
    raw_stream: AsyncIterator[Any],
    model: str,
    *,
    holdback: SseHoldback | None = None,
    _ledger_ref: list[Any] | None = None,
) -> AsyncIterator[str]:
    """Convert OpenAI streaming chunks into Anthropic Messages SSE events.

    Implements the full Anthropic SSE protocol with proper block lifecycle
    management for text, thinking, and tool_use content blocks.

    If ``holdback`` is provided, events pass through it before being yielded.
    If ``_ledger_ref`` is provided, the internal ``_StreamLedger`` is stored in
    ``_ledger_ref[0]`` so the caller can inspect ``accumulated_text``,
    ``accumulated_reasoning``, and ``tool_block_snapshot()`` for midstream recovery.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"
    ledger = _StreamLedger(msg_id, model)
    if _ledger_ref is not None:
        _ledger_ref.append(ledger)
    think_parser = ThinkTagParser()

    # Pending tool call rate (name can arrive across multiple deltas)
    pending_tool_names: dict[int, str] = {}
    pending_tool_ids: dict[int, str] = {}
    pending_tool_args: dict[int, list[str]] = {}

    finish_reason: str | None = None
    total_output_tokens = 0

    def _yield(event: str) -> list[str]:
        """Push an event through the holdback (if present) and return committed list."""
        if holdback is not None:
            return holdback.push(event)
        return [event]

    def _yield_multi(events: list[str]) -> list[str]:
        """Push multiple events through the holdback and return committed list."""
        result: list[str] = []
        for event in events:
            result.extend(_yield(event))
        return result

    # 1. message_start
    for event in _yield(ledger.message_start()):
        yield event

    async for chunk in raw_stream:
        # Track usage if available
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            total_output_tokens = getattr(chunk_usage, "completion_tokens", 0) or 0

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        # ── Handle reasoning_content / reasoning field (native OpenAI reasoning) ──
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            for event in _yield_multi(ledger.ensure_thinking_block()):
                yield event
            for event in _yield(ledger.emit_thinking_delta(reasoning)):
                yield event

        # ── Handle text content (with thinking tag parsing) ──
        content = getattr(delta, "content", None)
        if content:
            for content_type, text in think_parser.feed(content):
                if content_type == "thinking":
                    for event in _yield_multi(ledger.ensure_thinking_block()):
                        yield event
                    for event in _yield(ledger.emit_thinking_delta(text)):
                        yield event
                else:  # "text"
                    for event in _yield_multi(ledger.ensure_text_block()):
                        yield event
                    for event in _yield(ledger.emit_text_delta(text)):
                        yield event

        # ── Handle tool calls ──
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            # Close thinking/text blocks before tool blocks
            for event in _yield_multi(ledger.close_content_blocks()):
                yield event

            for tc in tool_calls:
                raw_index = getattr(tc, "index", 0)
                tc_index = raw_index if isinstance(raw_index, int) and raw_index >= 0 else 0

                tool_id = getattr(tc, "id", None)
                if tool_id:
                    pending_tool_ids[tc_index] = str(tool_id)
                elif tc_index not in pending_tool_ids:
                    # NIM/some models may omit the ID; generate fallback
                    pending_tool_ids[tc_index] = f"call_nim_{uuid.uuid4().hex[:10]}"

                fn = getattr(tc, "function", None)
                incoming_name = getattr(fn, "name", None) if fn else None
                arguments = getattr(fn, "arguments", None) if fn else None

                if incoming_name:
                    prev_name = pending_tool_names.get(tc_index, "")
                    if not prev_name or incoming_name.startswith(prev_name):
                        pending_tool_names[tc_index] = incoming_name
                    elif not prev_name.startswith(incoming_name):
                        pending_tool_names[tc_index] = prev_name + incoming_name

                # If tool block not yet been but we have name + id, start it
                if tc_index not in ledger.tool_states:
                    resolved_name = pending_tool_names.get(tc_index, "")
                    resolved_id = pending_tool_ids.get(tc_index)
                    if resolved_name.strip() and resolved_id:
                        for event in _yield(ledger.start_tool_block(
                            tc_index,
                            resolved_id,
                            resolved_name.strip(),
                        )):
                            yield event
                        # Flush any pre-buffered arguments
                        if tc_index in pending_tool_args:
                            for buffered_arg in pending_tool_args[tc_index]:
                                for event in _yield(ledger.emit_tool_delta(tc_index, buffered_arg)):
                                    yield event
                            del pending_tool_args[tc_index]

                # Emit argument delta
                if arguments:
                    if tc_index in ledger.tool_states:
                        for event in _yield(ledger.emit_tool_delta(tc_index, arguments)):
                            yield event
                    else:
                        # Buffer arguments until we have name + id to start the block
                        pending_tool_args.setdefault(tc_index, []).append(arguments)

    # ── Flush think parser ──
    remaining = think_parser.flush()
    if remaining:
        content_type, text = remaining
        if content_type == "thinking":
            for event in _yield_multi(ledger.ensure_thinking_block()):
                yield event
            for event in _yield(ledger.emit_thinking_delta(text)):
                yield event
        else:
            for event in _yield_multi(ledger.ensure_text_block()):
                yield event
            for event in _yield(ledger.emit_text_delta(text)):
                yield event

    # ── Ensure at least one content block ──
    has_any_block = (
        ledger.text_index != -1
        or ledger.thinking_index != -1
        or ledger.has_tool_blocks
    )
    if not has_any_block:
        for event in _yield_multi(ledger.ensure_text_block()):
            yield event
        for event in _yield(ledger.emit_text_delta(" ")):
            yield event

    # ── Close all open blocks ──
    for event in _yield_multi(ledger.close_all_open_blocks()):
        yield event

    # ── message_delta + message_stop ──
    stop_reason = ledger.final_stop_reason(finish_reason)
    if total_output_tokens == 0:
        total_output_tokens = max(1, ledger.estimate_output_tokens())
    for event in _yield(ledger.message_delta(stop_reason, total_output_tokens)):
        yield event
    for event in _yield(ledger.message_stop()):
        yield event

    # Flush any deferred content from the holdback at the very end
    if holdback is not None:
        for event in holdback.flush():
            yield event