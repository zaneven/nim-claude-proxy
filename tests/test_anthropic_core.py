"""Tests for Anthropic Messages -> OpenAI Chat conversion logic."""

from nim_proxy.core.anthropic import (
    anthropic_request_to_openai_body,
    MessagesRequest,
    Message,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
)


def test_request_with_thinking_blocks():
    req = MessagesRequest(
        model="test-model",
        messages=[
            Message(
                role="assistant",
                content=[
                    ContentBlockThinking(thinking="This is my thought process."),
                    ContentBlockText(text="And here is my answer."),
                    ContentBlockRedactedThinking(data="opaque-data"),
                ]
            )
        ]
    )

    body = anthropic_request_to_openai_body(req)
    
    assert len(body["messages"]) == 1
    content = body["messages"][0]["content"]
    
    assert "<think>\nThis is my thought process.\n</think>" in content
    assert "And here is my answer." in content
    assert "opaque-data" not in content  # redacted thinking should be ignored

def test_tool_choice_conversion():
    req = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        tool_choice={"type": "tool", "name": "Bash"}
    )
    body = anthropic_request_to_openai_body(req)
    assert body["tool_choice"] == {"type": "function", "function": {"name": "Bash"}}

    req2 = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        tool_choice={"type": "any"}
    )
    body2 = anthropic_request_to_openai_body(req2)
    assert body2["tool_choice"] == "required"

def test_stop_sequences_conversion():
    req = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stop_sequences=["stop_1", "stop_2"]
    )
    body = anthropic_request_to_openai_body(req)
    assert body["stop"] == ["stop_1", "stop_2"]

def test_image_block_conversion():
    from nim_proxy.core.anthropic import ContentBlockImage, ContentBlockImageSource
    req = MessagesRequest(
        model="test-model",
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(text="What is this?"),
                    ContentBlockImage(
                        source=ContentBlockImageSource(media_type="image/png", data="base64_str")
                    )
                ]
            )
        ]
    )
    body = anthropic_request_to_openai_body(req)
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,base64_str"}}
