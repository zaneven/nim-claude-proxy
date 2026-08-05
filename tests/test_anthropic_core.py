"""Tests for Anthropic Messages -> OpenAI Chat conversion logic (fcc_port engine).

Covers the same behaviors as the legacy ``nim_proxy.core.anthropic`` module,
now exercised against the 1:1 port of the original free-claude-code engine.
"""

from nim_proxy.fcc_port.core.anthropic import (
    AnthropicToOpenAIConverter,
    ContentBlockRedactedThinking,
    ContentBlockText,
    ContentBlockThinking,
    Message,
    MessagesRequest,
    build_base_request_body,
)


def test_request_with_thinking_blocks():
    req = MessagesRequest(
        model="test-model",
        messages=[
            Message(
                role="assistant",
                content=[
                    ContentBlockThinking(type="thinking", thinking="This is my thought process."),
                    ContentBlockText(type="text", text="And here is my answer."),
                    ContentBlockRedactedThinking(type="redacted_thinking", data="opaque-data"),
                ],
            )
        ],
    )

    body = build_base_request_body(req)

    assistant = [m for m in body["messages"] if m["role"] == "assistant"]
    assert assistant
    content = assistant[0]["content"]

    assert "<think>" in content
    assert "This is my thought process." in content
    assert "And here is my answer." in content
    assert "opaque-data" not in content  # redacted thinking is never replayed


def test_tool_choice_conversion():
    req = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        tool_choice={"type": "tool", "name": "Bash"},
    )
    body = build_base_request_body(req)
    assert body["tool_choice"] == {"type": "function", "function": {"name": "Bash"}}

    req2 = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        tool_choice={"type": "any"},
    )
    body2 = build_base_request_body(req2)
    assert body2["tool_choice"] == "required"


def test_stop_sequences_conversion():
    req = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stop_sequences=["stop_1", "stop_2"],
    )
    body = build_base_request_body(req)
    assert body["stop"] == ["stop_1", "stop_2"]


def test_image_block_conversion():
    req = MessagesRequest(
        model="test-model",
        messages=[
            Message(
                role="user",
                content=[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGVsbG8=",
                        },
                    },
                    {"type": "text", "text": "what is this?"},
                ],
            )
        ],
    )
    body = build_base_request_body(req)
    user = body["messages"][0]
    assert user["role"] == "user"
    parts = user["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts, f"expected an image_url part, got: {parts}"
    assert "aGVsbG8=" in image_parts[0]["image_url"]["url"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert text_parts and text_parts[0]["text"] == "what is this?"


def test_convert_tools_basic_shape():
    req = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
    )
    body = build_base_request_body(req)
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "get_weather"


def test_converter_exported_from_engine():
    assert AnthropicToOpenAIConverter is not None
