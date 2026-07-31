"""Tests for GET /v1/models and GET /models endpoint discovery."""

from fastapi.testclient import TestClient
from nim_proxy.api.app import app
from nim_proxy.config.settings import settings


def test_list_models_endpoint():
    settings.proxy_api_token = ""
    client = TestClient(app)

    # 1. 校验 GET /v1/models
    r1 = client.get("/v1/models")
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1.get("object") == "list"
    models1 = [m["id"] for m in data1.get("data", [])]
    assert "claude-3-5-sonnet-20241022" in models1
    assert "claude-3-opus-20240229" in models1

    # 2. 校验 GET /models (兼容别名)
    r2 = client.get("/models")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2.get("object") == "list"
    models2 = [m["id"] for m in data2.get("data", [])]
    assert "claude-3-5-sonnet-20241022" in models2
