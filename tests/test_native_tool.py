"""Tests for the ported MiniMax-M3 native tool stream cleaning engine.

Covers the same behaviors as the legacy ``nim_proxy.provider.native_tool``
module, now living in ``fcc_port.providers.nvidia_nim.native_tool_stream``
(1:1 port of the original free-claude-code engine).
"""

import pytest

from nim_proxy.fcc_port.providers.nvidia_nim.native_tool_stream import (
    NimNativeToolProtocolError,
    _MiniMaxM3ToolFramer,
    _parse_tool_block,
)

NS = "]<]minimax[>["

_TOOL_SCHEMAS = {
    "bash": {
        "type": "object",
        "properties": {"cmd": {"type": "string"}},
    },
    "get_weather": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    },
}


def _element(name: str, value: str) -> str:
    return f"{NS}<{name}>{value}{NS}</{name}>"


def _invoke(name: str, arguments: str) -> str:
    return f'{NS}<invoke name="{name}">{arguments}{NS}</invoke>'


def _tool_block(*invokes: str) -> str:
    return f"{NS}<tool_call>{''.join(invokes)}{NS}</tool_call>"


def test_framer_passes_plain_text_through():
    framer = _MiniMaxM3ToolFramer()
    visible = framer.feed("Hello World from NVIDIA NIM test string!")
    assert visible == "Hello World from NVIDIA NIM test string!"
    assert framer.tool_block is None


def test_framer_separates_text_and_tool_block():
    framer = _MiniMaxM3ToolFramer()
    raw = _tool_block(_invoke("bash", _element("cmd", "ls")))
    visible = framer.feed(raw)
    assert visible == ""
    assert framer.tool_block is not None
    assert '<invoke name="bash">' in framer.tool_block


def test_framer_keeps_visible_text_before_tool_block():
    framer = _MiniMaxM3ToolFramer()
    raw = "Let me check." + _tool_block(_invoke("bash", _element("cmd", "ls")))
    visible = framer.feed(raw)
    assert visible == "Let me check."
    assert framer.tool_block is not None


def test_framer_streams_tool_block_across_chunks():
    framer = _MiniMaxM3ToolFramer()
    raw = _tool_block(_invoke("bash", _element("cmd", "ls")))
    mid = len(raw) // 2
    assert framer.feed(raw[:mid]) == ""
    assert framer.tool_block is None
    assert framer.feed(raw[mid:]) == ""
    assert framer.tool_block is not None
    assert "bash" in framer.tool_block


def test_parse_tool_block_extracts_call():
    block = _invoke("get_weather", _element("city", "Beijing"))
    calls = _parse_tool_block(block, _TOOL_SCHEMAS)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Beijing"}


def test_parse_tool_block_rejects_undeclared_tool():
    block = _invoke("unknown_tool", _element("x", "1"))
    with pytest.raises(NimNativeToolProtocolError):
        _parse_tool_block(block, _TOOL_SCHEMAS)


def test_framer_rejects_trailing_text_after_tool_block():
    framer = _MiniMaxM3ToolFramer()
    raw = _tool_block(_invoke("bash", _element("cmd", "ls"))) + "trailing"
    with pytest.raises(NimNativeToolProtocolError):
        framer.feed(raw)
