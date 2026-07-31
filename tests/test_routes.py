"""Route-level tests using FastAPI TestClient with a mocked NimClient (no network)."""

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
