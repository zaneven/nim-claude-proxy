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
