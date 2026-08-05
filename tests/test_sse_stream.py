"""Tests for the ported Anthropic SSE streaming engine (fcc_port).

Replaces the legacy ``nim_proxy.api.sse`` tests: the SSE engine is now the
1:1 port of the original free-claude-code implementation living in
``nim_proxy.fcc_port.core.anthropic.streaming``.
"""

import json

import pytest

from nim_proxy.fcc_port.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from nim_proxy.fcc_port.core.anthropic.streaming import (
    AnthropicStreamLedger,
    map_stop_reason,
)
from nim_proxy.fcc_port.core.anthropic.thinking import ContentType, ThinkTagParser


def _parse_event(raw: str) -> tuple[str, dict]:
    event = None
    data = None
    for line in raw.strip().splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data = json.loads(line[len("data: "):])
    return event, data


async def _aiter(items):
    for item in items:
        yield item


# ── ThinkTagParser ──


def test_think_tag_parser_splits_thinking_and_text():
    parser = ThinkTagParser()
    chunks = list(parser.feed("<think>internal thought</think>final answer"))
    tail = parser.flush()
    if tail is not None:
        chunks.append(tail)
    kinds = [c.type for c in chunks]
    assert ContentType.THINKING in kinds
    assert ContentType.TEXT in kinds
    thinking = "".join(c.content for c in chunks if c.type is ContentType.THINKING)
    text = "".join(c.content for c in chunks if c.type is ContentType.TEXT)
    assert thinking == "internal thought"
    assert text == "final answer"


def test_think_tag_parser_handles_partial_tags_across_feeds():
    parser = ThinkTagParser()
    out = []
    for piece in ("<thi", "nk>deep", "</think>", "done"):
        out.extend(parser.feed(piece))
    tail = parser.flush()
    if tail is not None:
        out.append(tail)
    thinking = "".join(c.content for c in out if c.type is ContentType.THINKING)
    text = "".join(c.content for c in out if c.type is ContentType.TEXT)
    assert thinking == "deep"
    assert text == "done"


# ── Stream ledger event shapes ──


def test_ledger_message_start_shape():
    ledger = AnthropicStreamLedger("msg_test", "claude-test", input_tokens=42)
    event, data = _parse_event(ledger.message_start())
    assert event == "message_start"
    assert data["message"]["id"] == "msg_test"
    assert data["message"]["model"] == "claude-test"
    assert data["message"]["usage"]["input_tokens"] == 42


def test_ledger_text_block_sequence():
    ledger = AnthropicStreamLedger(None, "claude-test", input_tokens=1)
    start = ledger.content_block_start(0, "text")
    delta = ledger.content_block_delta(0, "text_delta", "hello")
    stop = ledger.content_block_stop(0)
    assert _parse_event(start)[1]["content_block"] == {"type": "text", "text": ""}
    assert _parse_event(delta)[1]["delta"] == {"type": "text_delta", "text": "hello"}
    assert _parse_event(stop)[1]["index"] == 0


def test_map_stop_reason():
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason("length") == "max_tokens"


# ── Aggregation round-trip (non-streaming parity path) ──


@pytest.mark.asyncio
async def test_aggregate_sse_to_message_round_trip():
    ledger = AnthropicStreamLedger("msg_rt", "claude-test", input_tokens=10)
    events = [
        ledger.message_start(),
        ledger.content_block_start(0, "thinking"),
        ledger.content_block_delta(0, "thinking_delta", "reasoning here"),
        ledger.content_block_stop(0),
        ledger.content_block_start(1, "text"),
        ledger.content_block_delta(1, "text_delta", "final text"),
        ledger.content_block_stop(1),
        ledger.message_delta("end_turn", 7),
        ledger.message_stop(),
    ]
    message, error = await aggregate_anthropic_sse_to_message(_aiter(events))
    assert error is None
    assert message["id"] == "msg_rt"
    assert message["content"][0]["type"] == "thinking"
    assert message["content"][0]["thinking"] == "reasoning here"
    assert message["content"][1] == {"type": "text", "text": "final text"}
    assert message["stop_reason"] == "end_turn"
    assert message["usage"]["output_tokens"] == 7


@pytest.mark.asyncio
async def test_aggregate_surfaces_stream_error():
    error_frame = (
        "event: error\n"
        'data: {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}\n\n'
    )
    message, error = await aggregate_anthropic_sse_to_message(_aiter([error_frame]))
    assert error is not None
    assert error["type"] == "overloaded_error"
