"""Tests for the rewritten SSE streaming module with proper Anthropic protocol."""

import json
import pytest
from types import SimpleNamespace
from nim_proxy.api.sse import (
    format_anthropic_sse_stream,
    ThinkTagParser,
    _StreamLedger,
    _sse_event,
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
        result = parser.feed("<think>reasoning here</think>visible text")
        assert ("thinking", "reasoning here") in result
        assert ("text", "visible text") in result

    def test_streaming_think_tag(self):
        parser = ThinkTagParser()
        # Feed tag in parts
        r1 = parser.feed("<think>part1")
        assert ("thinking", "part1") in r1

        r2 = parser.feed(" part2</think>text")
        assert ("thinking", " part2") in r2
        assert ("text", "text") in r2

    def test_flush(self):
        parser = ThinkTagParser()
        # Feed content that leaves a partial tag in the buffer
        r = parser.feed("<think>start")
        # "start" is already emitted as thinking since no partial close tag
        assert ("thinking", "start") in r
        # Feed partial close tag that gets buffered
        r2 = parser.feed("more<")
        # "more" can't be emitted because "<" might be start of "</think>"
        # So it's buffered. Flush should emit it.
        result = parser.flush()
        assert result is not None
        assert result[0] == "thinking"
        assert "<" in result[1] or "more" in result[1]


# ── StreamLedger Tests ──

class TestStreamLedger:
    def test_message_lifecycle(self):
        ledger = _StreamLedger("msg_test", "test-model")
        start = json.loads(ledger.message_start().split("data: ")[1].split("\n")[0])
        assert start["type"] == "message_start"
        assert start["message"]["id"] == "msg_test"
        assert start["message"]["model"] == "test-model"

    def test_text_block_lifecycle(self):
        ledger = _StreamLedger("msg_test", "test-model")
        events = ledger.ensure_text_block()
        assert len(events) == 1
        start = json.loads(events[0].split("data: ")[1].split("\n")[0])
        assert start["content_block"]["type"] == "text"
        assert start["index"] == 0

        delta = json.loads(ledger.emit_text_delta("hello").split("data: ")[1].split("\n")[0])
        assert delta["delta"]["text"] == "hello"

        stop_events = ledger.stop_text_block()
        assert len(stop_events) == 1
        stop = json.loads(stop_events[0].split("data: ")[1].split("\n")[0])
        assert stop["type"] == "content_block_stop"
        assert stop["index"] == 0

    def test_thinking_to_text_transition(self):
        ledger = _StreamLedger("msg_test", "test-model")
        # Start thinking block
        events = ledger.ensure_thinking_block()
        assert len(events) == 1
        start = json.loads(events[0].split("data: ")[1].split("\n")[0])
        assert start["content_block"]["type"] == "thinking"
        assert start["index"] == 0

        # Transition to text should close thinking first
        events = ledger.ensure_text_block()
        assert len(events) == 2  # thinking stop + text start
        thinking_stop = json.loads(events[0].split("data: ")[1].split("\n")[0])
        assert thinking_stop["type"] == "content_block_stop"
        assert thinking_stop["index"] == 0

        text_start = json.loads(events[1].split("data: ")[1].split("\n")[0])
        assert text_start["content_block"]["type"] == "text"
        assert text_start["index"] == 1  # incremented

    def test_tool_block_lifecycle(self):
        ledger = _StreamLedger("msg_test", "test-model")
        start = ledger.start_tool_block(0, "call_abc", "read_file")
        start_data = json.loads(start.split("data: ")[1].split("\n")[0])
        assert start_data["content_block"]["type"] == "tool_use"
        assert start_data["content_block"]["id"] == "call_abc"
        assert start_data["content_block"]["name"] == "read_file"

        delta = ledger.emit_tool_delta(0, '{"path": "/tmp"}')
        delta_data = json.loads(delta.split("data: ")[1].split("\n")[0])
        assert delta_data["delta"]["partial_json"] == '{"path": "/tmp"}'

        stop = ledger.stop_tool_block(0)
        stop_data = json.loads(stop.split("data: ")[1].split("\n")[0])
        assert stop_data["type"] == "content_block_stop"

    def test_final_stop_reason_with_tools(self):
        ledger = _StreamLedger("msg_test", "test-model")
        ledger.start_tool_block(0, "call_1", "write_file")
        assert ledger.final_stop_reason("stop") == "tool_use"

    def test_final_stop_reason_without_tools(self):
        ledger = _StreamLedger("msg_test", "test-model")
        assert ledger.final_stop_reason("stop") == "end_turn"
        assert ledger.final_stop_reason("length") == "max_tokens"

    def test_close_all_open_blocks(self):
        ledger = _StreamLedger("msg_test", "test-model")
        ledger.ensure_text_block()
        ledger.start_tool_block(0, "call_1", "test")
        events = ledger.close_all_open_blocks()
        # Should close text block + tool block
        assert len(events) == 2


# ── Full SSE Stream Integration Tests ──

@pytest.mark.asyncio
async def test_text_only_stream():
    chunks = [
        _make_chunk(content="Hello "),
        _make_chunk(content="world!"),
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)

    types = [e["type"] for e in parsed]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"

    # Check stop_reason
    msg_delta = [e for e in parsed if e["type"] == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_tool_call_stream():
    """Test that tool call blocks have proper start → delta → stop lifecycle."""
    chunks = [
        _make_chunk(content="Let me check that."),
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, tc_id="call_abc123", name="read_file"),
        ]),
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, arguments='{"path":'),
        ]),
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, arguments=' "/tmp/test.py"}'),
        ]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)

    # Find tool_use content_block_start
    tool_starts = [e for e in parsed if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["content_block"]["id"] == "call_abc123"
    assert tool_starts[0]["content_block"]["name"] == "read_file"
    tool_index = tool_starts[0]["index"]

    # Find tool_use argument deltas
    tool_deltas = [e for e in parsed if e.get("type") == "content_block_delta"
                   and e.get("delta", {}).get("type") == "input_json_delta"]
    assert len(tool_deltas) == 2
    assert tool_deltas[0]["delta"]["partial_json"] == '{"path":'
    assert tool_deltas[1]["delta"]["partial_json"] == ' "/tmp/test.py"}'

    # Find content_block_stop for the tool block
    tool_stops = [e for e in parsed if e.get("type") == "content_block_stop"
                  and e.get("index") == tool_index]
    assert len(tool_stops) == 1

    # Check stop_reason is tool_use
    msg_delta = [e for e in parsed if e["type"] == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_think_tag_in_stream():
    """Test <think> tag parsing produces thinking blocks."""
    chunks = [
        _make_chunk(content="<think>Let me reason about this"),
        _make_chunk(content="</think>Here is the answer"),
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)

    # Should have a thinking block
    thinking_starts = [e for e in parsed if e.get("type") == "content_block_start"
                       and e.get("content_block", {}).get("type") == "thinking"]
    assert len(thinking_starts) == 1

    thinking_deltas = [e for e in parsed if e.get("type") == "content_block_delta"
                       and e.get("delta", {}).get("type") == "thinking_delta"]
    assert len(thinking_deltas) >= 1

    # Should also have a text block
    text_starts = [e for e in parsed if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "text"]
    assert len(text_starts) == 1


@pytest.mark.asyncio
async def test_reasoning_content_field():
    """Test reasoning_content field produces thinking blocks."""
    chunks = [
        _make_chunk(reasoning_content="Step 1: analyze the problem"),
        _make_chunk(content="The answer is 42"),
        _make_chunk(finish_reason="stop"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)

    thinking_starts = [e for e in parsed if e.get("type") == "content_block_start"
                       and e.get("content_block", {}).get("type") == "thinking"]
    assert len(thinking_starts) == 1

    thinking_deltas = [e for e in parsed if e.get("type") == "content_block_delta"
                       and e.get("delta", {}).get("type") == "thinking_delta"]
    assert len(thinking_deltas) == 1
    assert thinking_deltas[0]["delta"]["thinking"] == "Step 1: analyze the problem"


@pytest.mark.asyncio
async def test_empty_stream_produces_minimal_response():
    """Ensure even an empty stream produces a valid response."""
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter([]), "test-model"))
    parsed = _parse_sse_events(events)

    types = [e["type"] for e in parsed]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"


@pytest.mark.asyncio
async def test_multiple_tool_calls():
    """Test multiple simultaneous tool calls each get proper lifecycle."""
    chunks = [
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, tc_id="call_1", name="read_file"),
            _make_tool_call_delta(index=1, tc_id="call_2", name="write_file"),
        ]),
        _make_chunk(tool_calls=[
            _make_tool_call_delta(index=0, arguments='{"path": "/a"}'),
            _make_tool_call_delta(index=1, arguments='{"path": "/b"}'),
        ]),
        _make_chunk(finish_reason="tool_calls"),
    ]
    events = await _chunks_to_list(format_anthropic_sse_stream(_aiter(chunks), "test-model"))
    parsed = _parse_sse_events(events)

    tool_starts = [e for e in parsed if e.get("type") == "content_block_start"
                   and e.get("content_block", {}).get("type") == "tool_use"]
    assert len(tool_starts) == 2

    # Each tool block should have a stop
    for ts in tool_starts:
        idx = ts["index"]
        stops = [e for e in parsed if e.get("type") == "content_block_stop" and e.get("index") == idx]
        assert len(stops) == 1

    msg_delta = [e for e in parsed if e["type"] == "message_delta"][0]
    assert msg_delta["delta"]["stop_reason"] == "tool_use"
