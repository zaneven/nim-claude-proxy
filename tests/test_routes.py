"""Route-level tests using FastAPI TestClient with a mocked NimClient (no network)."""

import pytest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from nim_proxy.api import routes as routes_mod
from nim_proxy.api.app import app


def _fake_execute(content="ok", tool_calls=None):
    async def _exec(body):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=None,
            )],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
        )
    routes_mod.nim_provider.execute_chat_non_streaming = _exec


def _sse(event: str, data: dict) -> str:
    import json as _json
    return f"event: {event}\ndata: {_json.dumps(data)}\n\n"


def _message_start_event() -> str:
    return _sse("message_start", {"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant",
        "content": [], "model": "claude-3-5-sonnet-20241022",
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }})


def _block_events(index: int, start_block: dict, delta: dict) -> list[str]:
    return [
        _sse("content_block_start", {"type": "content_block_start", "index": index,
                                     "content_block": start_block}),
        _sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                     "delta": delta}),
        _sse("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _message_end_events() -> list[str]:
    return [
        _sse("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"input_tokens": 1, "output_tokens": 2}}),
        _sse("message_stop", {"type": "message_stop"}),
    ]


def _text_message_sse(text: str) -> list[str]:
    return (
        [_message_start_event()]
        + _block_events(0, {"type": "text", "text": ""},
                        {"type": "text_delta", "text": text})
        + _message_end_events()
    )


def _mock_provider_stream(monkeypatch, sse_events: list[str]):
    """Mock the provider seam used by the original engine pipeline."""
    monkeypatch.setattr(routes_mod.nim_provider, "preflight_stream", lambda *a, **k: None)

    async def _gen(*a, **k):
        for event in sse_events:
            yield event

    monkeypatch.setattr(routes_mod.nim_provider, "stream_response", _gen)


def test_health(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_no_crash(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    c = TestClient(app)
    r = c.get("/v1/models")
    assert r.status_code == 200
    assert "data" in r.json()


def test_anthropic_messages_non_stream_mock(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    _mock_provider_stream(monkeypatch, _text_message_sse("hi there"))
    c = TestClient(app)
    r = c.post("/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
        "stream": False,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "message"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "hi there"


def test_openai_chat_completions_mock(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    _fake_execute(content="olá")
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "olá"


def test_openai_responses_mock(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    _fake_execute(content="resp-text")
    c = TestClient(app)
    r = c.post("/v1/responses", json={
        "model": "x",
        "input": "hi",
        "stream": False,
    })
    assert r.status_code == 200
    out = r.json()
    assert out["object"] == "response"
    assert out["output"][0]["content"][0]["text"] == "resp-text"


def test_admin_api_requires_token_when_configured(monkeypatch):
    # 模拟已配置 PROXY_API_TOKEN
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "secret-token")
    c = TestClient(app)
    # 无 Token -> 401
    assert c.get("/admin/api/config").status_code == 401
    # 带 Token -> 200
    assert c.get("/admin/api/config", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_anthropic_messages_non_stream_thinking(monkeypatch):
    """Non-streaming requests aggregate the engine's SSE stream (original parity)."""
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    sse_events = (
        [_message_start_event()]
        + _block_events(0, {"type": "thinking", "thinking": ""},
                        {"type": "thinking_delta", "thinking": "native reasoning field"})
        + _block_events(1, {"type": "thinking", "thinking": ""},
                        {"type": "thinking_delta", "thinking": "internal thought process"})
        + _block_events(2, {"type": "text", "text": ""},
                        {"type": "text_delta", "text": "final answer"})
        + _message_end_events()
    )
    _mock_provider_stream(monkeypatch, sse_events)

    c = TestClient(app)
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stream": False,
    }

    resp = c.post("/v1/messages", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["content"]) == 3

    # First block: thinking from native reasoning field
    assert data["content"][0]["type"] == "thinking"
    assert data["content"][0]["thinking"] == "native reasoning field"

    # Remaining blocks: parsed thinking + text
    block_types = [b["type"] for b in data["content"][1:]]
    assert block_types == ["thinking", "text"]
    assert data["content"][2]["text"] == "final answer"


def test_anthropic_messages_rejects_empty_messages(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    c = TestClient(app)
    r = c.post("/v1/messages", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [],
        "max_tokens": 100,
        "stream": False,
    })
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_anthropic_messages_count_tokens(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    c = TestClient(app)
    r = c.post("/v1/messages/count_tokens", json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hello world"}],
    })
    assert r.status_code == 200
    assert r.json()["input_tokens"] > 0
