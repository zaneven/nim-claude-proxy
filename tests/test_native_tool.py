"""Tests for MiniMaxM3ToolFramer (Native Tool Cleaning)."""

from nim_proxy.provider.native_tool import MiniMaxM3ToolFramer, parse_minimax_tool_calls


def test_framer_filters_text_and_block():
    framer = MiniMaxM3ToolFramer()
    
    text1, block1 = framer.feed("Hello World from NVIDIA NIM test string!")
    assert text1 == "Hello World from NVIDIA NIM test string!"
    assert block1 == ""

    # 输入触发开头的标签
    raw = "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"bash\"><parameter name=\"cmd\">\"ls\"</parameter>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
    text2, block2 = framer.feed(raw)
    assert text2 == ""
    assert block2 == "]<]minimax[>[<invoke name=\"bash\"><parameter name=\"cmd\">\"ls\"</parameter>]<]minimax[>[</invoke>"


def test_parse_minimax_tool_calls():
    raw_block = ']<]minimax[>[<invoke name="get_weather"><parameter name="city">"Beijing"</parameter>]<]minimax[>[</invoke>'
    tool_calls = parse_minimax_tool_calls(raw_block)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert "Beijing" in tool_calls[0]["function"]["arguments"]
