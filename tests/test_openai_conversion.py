"""Tests for OpenAI Responses API <-> Chat Completions conversion."""

from types import SimpleNamespace

from nim_proxy.core.openai import (
    openai_response_to_responses,
    responses_request_to_openai_body,
)


def test_responses_string_input():
    body = responses_request_to_openai_body({"model": "x", "input": "hello"})
    assert body["model"] == "x"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_responses_list_input_with_instructions_and_function_tool():
    body = responses_request_to_openai_body({
        "model": "x",
        "instructions": "be nice",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    })
    assert body["messages"][0] == {"role": "system", "content": "be nice"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["messages"][2] == {"role": "assistant", "content": "ok"}
    assert body["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_responses_preview_tool_dropped():
    body = responses_request_to_openai_body({
        "model": "x",
        "input": "q",
        "tools": [{"type": "web_search_preview"}],
    })
    assert "tools" not in body


def test_openai_response_to_responses():
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7),
    )
    out = openai_response_to_responses(resp, "x")
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"][0]["text"] == "hello"
    assert out["usage"]["input_tokens"] == 5
    assert out["usage"]["output_tokens"] == 7
