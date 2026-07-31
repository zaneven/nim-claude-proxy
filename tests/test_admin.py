"""Tests for Admin API and Configuration Persistence."""

import pytest
from fastapi.testclient import TestClient
from nim_proxy.api.app import app
from nim_proxy.config.settings import settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_auth(monkeypatch):
    # 这些测试验证 Admin API 逻辑本身，不依赖 PROXY_API_TOKEN 鉴权
    monkeypatch.setattr(settings, "proxy_api_token", "")


def test_get_admin_config():
    response = client.get("/admin/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "settings" in data
    assert "key_pool_stats" in data
    assert "NIM_DEFAULT_MODEL" in data["settings"]


def test_discover_nim_models():
    response = client.get("/admin/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0


def test_save_admin_config():
    orig_model = settings.nim_default_model
    orig_cooldown = settings.key_cooldown_seconds
    try:
        new_model = "meta/llama-3.3-70b-instruct"
        response = client.post(
            "/admin/api/config",
            json={"nim_default_model": new_model, "key_cooldown_seconds": 90.0}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert settings.nim_default_model == new_model
    finally:
        # 还原，避免污染真实 .env
        client.post(
            "/admin/api/config",
            json={"nim_default_model": orig_model, "key_cooldown_seconds": orig_cooldown},
        )


def test_reset_keypool():
    response = client.post("/admin/api/keypool/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_serve_admin_ui():
    response = client.get("/admin")
    assert response.status_code == 200
    assert "NIM Proxy Admin" in response.text
