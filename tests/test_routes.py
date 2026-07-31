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
    routes_mod._client.execute_chat = _exec


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
    _fake_execute(content="hi there")
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


@pytest.mark.asyncio
async def test_anthropic_messages_non_stream_thinking(monkeypatch):
    monkeypatch.setattr(routes_mod.settings, "proxy_api_token", "")
    
    async def _exec(body):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="<think>internal thought process</think>final answer",
                    reasoning_content="native reasoning field",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )
    routes_mod._client.execute_chat = _exec

    c = TestClient(app)
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stream": False,
    }

    resp = c.post(
        "/v1/messages",
        json=payload,
    )
    assert resp.status_code == 200
    data = resp.json()
    
    # 验证解析出两个 thinking 块和一个 text 块
    assert len(data["content"]) == 3
    
    # 验证第一个块: 从 reasoning_content 提取的 thinking
    assert data["content"][0]["type"] == "thinking"
    assert data["content"][0]["thinking"] == "native reasoning field"

    # 验证第二个块: 从 <think> 提取的 thinking
    assert data["content"][1]["type"] == "thinking"
    assert data["content"][1]["thinking"] == "internal thought process"

    # 验证第三个块: 纯文本
    assert data["content"][2]["type"] == "text"
    assert data["content"][2]["text"] == "final answer"
