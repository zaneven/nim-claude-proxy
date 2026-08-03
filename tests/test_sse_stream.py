"""Tests for the rewritten SSE streaming module with proper Anthropic protocol."""

import json
import pytest
from types import SimpleNamespace
from nim_proxy.api.sse import (
    SseHoldback,
    format_anthropic_sse_stream,
    ThinkTagParser,
    _StreamLedger,
    _sse_event,
    sse_error_event,
)


# ── Helpers ──

def _make_chunk(
    content=None,
    tool_calls=None,
    finish_reason=None,
    reasoning_content=None,
    reasoning=None,
    usage=None,
):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
    )
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
    )


def _make_tool_call_delta(index=0, tc_id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=fn)


async def _chunks_to_list(chunks):
    return [c async for c in chunks]


async def _aiter(items):
    for item in items:
        yield item


def _parse_sse_events(events: list[str]) -> list[dict]:
    """Parse SSE event strings into dicts."""
    parsed = []
    for event_str in events:
        for block in event_str.strip().split("\n\n"):
            lines = block.strip().split("\n")
            data_line = [l for l in lines if l.startswith("data: ")]
            if data_line:
                parsed.append(json.loads(data_line[0][len("data: "):]))
    return parsed


# ── ThinkTagParser Tests ──

class TestThinkTagParser:
    def test_plain_text(self):
        parser = ThinkTagParser()
        result = parser.feed("hello world")
        assert result == [("text", "hello world")]

    def test_think_tag(self):
        parser = ThinkTagParser()
        result = parser.feed(" thinkingreasoning here responsevisible text")
        assert ("thinking", "reasoning here") in result
        assert ("text", "visible text") in result

    def test_streaming_think_tag(self):
        parser = ThinkTagParser()
        # Feed tag in parts
        r1 = parser.feed(" thinkingpart1")
        assert ("thinking", "part1") in r1

        r2 = parser.feed(" more responseout")
        assert ("thinking", " more") in r2
        assert ("text", "out") in r2

    def test_flush(self):
        parser = ThinkTagParser()
        # Start thinking mode
        r = parser.feed(" thinkingstuff")
        assert ("thinking", "stuff") in r
        # Feed content without close tag
        r2 = parser.feed("more data")
        assert ("thinking", "more data") in r2
        # Buffer is empty after feed, so flush returns None
        result = parser.flush()
        assert result is None


# ── SseHoldback Tests ──

class TestSseHoldback:
    def test_holds_until_content_block_start(self):
        hb = SseHoldback()
        assert hb.push("event: message_start\ndata: {}\n\n") == []
        assert hb.committed is False
        assert hb.has_buffered is True

    def test_commits_on_content_block_start(self):
        hb = SseHoldback()
        hb.push('event: message_start\ndata: {"type":"message_start"}\n\n')
        result = hb.push('event: content_block_start\ndata: {"type":"content_block_start"}\n\n')
        assert len(result) == 2
        assert hb.committed is True

    def test_pass_through_after_committed(self):
        hb = SseHoldback()
        hb.push('event: message_start\ndata: {}\n\n')
        hb.push('event: content_block_start\ndata: {"type":"content_block_start","content_block":{"type":"text","text":""}}\n\n')
        result = hb.push('event: content_block_delta\ndata: {}\n\n')
        assert len(result) == 1

    def test_flush_returns_buffered(self):
        hb = SseHoldback()
        hb.push('event: message_start\ndata: {}\n\n')
        flushed = hb.flush()
        assert len(flushed) == 1

    def test_discard_clears(self):
        hb = SseHoldback()
        hb.push('event: message_start\ndata: {}\n\n')
        hb.discard()
        assert len(hb.flush()) == 0
        assert hb.has_buffered is False


# ── _StreamLedger Tests ──

class TestStreamLedger:
    def test_message_lifecycle(self):
        ledger = _StreamLedger("msg_1", "test-model")
        start = ledger.message_start()
        assert "message_start" in start

        delta = ledger.message_delta("end_turn", 10)
        assert "message_delta" in delta

        stop = ledger.message_stop()
        assert "message_stop" in stop

    def test_text_block_lifecycle(self):
        ledger = _StreamLedger("msg_1", "test-model")
        events = ledger.ensure_text_block()
        assert len(events) == 1  # content_block_start for text
        assert ledger.text_started is True
        assert ledger.text_index == 0

        delta = ledger.emit_text_delta("hello")
        assert "hello" in delta
        assert ledger.accumulated_text == "hello"

        events = ledger.stop_text_block()
        assert len(events) == 1

    def test_thinking_to_text_transition(self):
        ledger = _StreamLedger("msg_1", "test-model")
        # Start thinking
        events = ledger.ensure_thinking_block()
        assert len(events) == 1
        assert ledger.thinking_started is True
        assert ledger.thinking_index == 0

        # Transition to text should close thinking first
        events = ledger.ensure_text_block()
        assert len(events) == 2  # stop thinking + start text
        assert ledger.thinking_started is False
        assert ledger.text_started is True
        assert ledger.text_index == 1

    def test_tool_block_lifecycle(self):
        ledger = _StreamLedger("msg_1", "test-model")
        event = ledger.start_tool_block(0, "tool_1", "test_tool")
        assert "tool_use" in event
        assert ledger.has_tool_blocks is True
        assert 0 in ledger.tool_states

        delta = ledger.emit_tool_delta(0, '{"key":')
        assert "input_json_delta" in delta
        assert ledger.tool_states[0].contents == ['{"key":']

        event = ledger.stop_tool_block(0)
        assert ledger.tool_states[0].stopped is True

    def test_final_stop_reason_with_tools(self):
        ledger = _StreamLedger("msg_1", "test-model")
        ledger.start_tool_block(0, "tc1", "test_tool")
        assert ledger.final_stop_reason("stop") == "tool_use"

    def test_final_stop_reason_without_tools(self):
        ledger = _StreamLedger("msg_1", "test-model")
        assert ledger.final_stop_reason("stop") == "end_turn"
        assert ledger.final_stop_reason("length") == "max_tokens"
        assert ledger.final_stop_reason(None) == "end_turn"

    def test_close_all_open_blocks(self):
        ledger = _StreamLedger("msg_1", "test-model")
        ledger.ensure_text_block()
        ledger.start_tool_block(0, "tool_1", "test_tool")
        events = ledger.close_all_open_blocks()
        assert len(events) >= 1  # text stop + tool stop


# ── Integration Tests ──

@pytest.mark.asyncio
async def test_text_only_stream():
    chunks = [
        _make_chunk(content="Hello "),
        _make_chunk(content="world"),
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)
    assert len(parsed) >= 4  # message_start, content_block_start, content_block_stop, message_delta, message_stop


@pytest.mark.asyncio
async def test_tool_call_stream():
    chunks = [
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, tc_id="call_1", name="test_tool", arguments='{"k')
        ]),
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, arguments='ey": "v"}')
        ]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)
    tool_starts = [e for e in parsed if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1


@pytest.mark.asyncio
async def test_think_tag_in_stream():
    """Test thinking tag parsing produces thinking blocks."""
    chunks = [
        _make_chunk(content=" thinkingLet me reason"),
        _make_chunk(content=" about this"),
        _make_chunk(content=" responseHere is the answer"),
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)
    thinking_starts = [e for e in parsed if e.get("type") == "content_block_start"
                       and e.get("content_block", {}).get("type") == "thinking"]
    assert len(thinking_starts) == 1
    text_starts = [e for e in parsed if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "text"]
    assert len(text_starts) == 1


@pytest.mark.asyncio
async def test_reasoning_content_field():
    chunks = [
        _make_chunk(content="thinking content", reasoning_content="CoT reasoning"),
        _make_chunk(content=" direct text"),
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)
    thinking_starts = [e for e in parsed if e.get("type") == "content_block_start"
                       and e.get("content_block", {}).get("type") == "thinking"]
    assert len(thinking_starts) == 1


@pytest.mark.asyncio
async def test_empty_stream_produces_minimal_response():
    chunks = [
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)
    assert len(parsed) >= 1  # at least message_start + content


@pytest.mark.asyncio
async def test_multiple_tool_calls():
    chunks = [
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, tc_id="tc_1", name="tool_a",
                                  arguments={"a": "1"}),
        ]),
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=1, tc_id="tc_2", name="tool_b",
                                  arguments={"b": "2"}),
        ]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)
    tool_starts = [e for e in parsed if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 2


# ── SseHoldback Integration Test ──

@pytest.mark.asyncio
async def test_holdback_early_retry_scenario():
    """Test that holdback buffers events until content_block_start."""
    chunks = [
        _make_chunk(content="hello"),
        _make_chunk(finish_reason="stop"),
    ]
    holdback = SseHoldback()
    raw_stream = _aiter(chunks)
    events = await _chunks_to_list(
        format_anthropic_sse_stream(raw_stream, "test-model", holdback=holdback)
    )
    # After stream completes, flush should return remaining events
    assert holdback.committed is True


@pytest.mark.asyncio
async def test_holdback_allows_recovery_on_early_failure():
    """SseHoldback returns nothing committed if stream fails before content_block_start."""
    hb = SseHoldback()
    # Simulate message_start only (no content_block_start)
    assert hb.push("event: message_start\ndata: {}\n\n") == []
    assert hb.committed is False
    # On error, discard and retry
    hb.discard()
    assert hb.has_buffered is False