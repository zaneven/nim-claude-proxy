"""Tests for Reasoning Policy and Route Overrides."""

from nim_proxy.config.settings import settings
from nim_proxy.core.reasoning import resolve_reasoning_budget, ReasoningPreference


def test_reasoning_resolution_inherit_and_off():
    # 当子路由为 inherit 且全局为 off
    res = resolve_reasoning_budget(
        preference_str="inherit",
        global_preference_str="off",
    )
    assert res.enabled is False
    assert res.budget_tokens is None


def test_reasoning_resolution_route_override():
    # 当全局为 client，但 sonnet 路由覆盖为 high (2048)
    res = resolve_reasoning_budget(
        preference_str="high",
        global_preference_str="client",
    )
    assert res.enabled is True
    assert res.budget_tokens == 2048


def test_settings_route_reasoning():
    settings.reasoning_preference = "client"
    settings.reasoning_sonnet = "xhigh"

    res = settings.get_reasoning_for_model("claude-3-5-sonnet-20241022")
    assert res.enabled is True
    assert res.budget_tokens == 4096
