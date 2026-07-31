"""Tests for verifying config persistence roundtrip for boolean web tool flags."""

from fastapi.testclient import TestClient
from nim_proxy.api.app import app
from nim_proxy.config.settings import settings


def test_boolean_web_tool_flags_persistence_roundtrip(monkeypatch):
    # 关闭鉴权以便测试配置持久化逻辑本身（不依赖 PROXY_API_TOKEN）
    monkeypatch.setattr(settings, "proxy_api_token", "")
    client = TestClient(app)

    # 1. 设为 False
    res_false = client.post("/admin/api/config", json={
        "enable_web_search": False,
        "enable_web_fetch": False,
    })
    assert res_false.status_code == 200
    assert settings.enable_web_search is False
    assert settings.enable_web_fetch is False

    # GET 校验
    g_false = client.get("/admin/api/config").json()
    assert g_false["settings"]["ENABLE_WEB_SEARCH"] is False
    assert g_false["settings"]["ENABLE_WEB_FETCH"] is False

    # 2. 设回 True
    res_true = client.post("/admin/api/config", json={
        "enable_web_search": True,
        "enable_web_fetch": True,
    })
    assert res_true.status_code == 200
    assert settings.enable_web_search is True
    assert settings.enable_web_fetch is True

    # GET 校验
    g_true = client.get("/admin/api/config").json()
    assert g_true["settings"]["ENABLE_WEB_SEARCH"] is True
    assert g_true["settings"]["ENABLE_WEB_FETCH"] is True
