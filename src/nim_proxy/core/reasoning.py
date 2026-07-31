"""Reasoning policy and route-specific overrides module for NIM Proxy."""

from enum import StrEnum
from dataclasses import dataclass


class ReasoningPreference(StrEnum):
    INHERIT = "inherit"
    OFF = "off"
    CLIENT = "client"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


_EFFORT_BUDGET_TOKENS = {
    ReasoningPreference.LOW: 512,
    ReasoningPreference.MEDIUM: 1024,
    ReasoningPreference.HIGH: 2048,
    ReasoningPreference.XHIGH: 4096,
    ReasoningPreference.MAX: 8192,
}


@dataclass(frozen=True, slots=True)
class ResolvedReasoning:
    enabled: bool
    budget_tokens: int | None = None


def resolve_reasoning_budget(
    preference_str: str,
    global_preference_str: str,
    default_budget: int = 4096,
    client_budget: int | None = None,
) -> ResolvedReasoning:
    """Resolve reasoning preference with route-specific override and fallback."""
    # 1. 如果当前路由的偏好是 inherit，继承全局偏好
    pref = preference_str.lower()
    if pref == ReasoningPreference.INHERIT or not pref:
        pref = global_preference_str.lower()

    # 2. 如果最终偏好为 off
    if pref == ReasoningPreference.OFF:
        return ResolvedReasoning(enabled=False, budget_tokens=None)

    # 3. 如果偏好为 client，优先采用客户端传入的 budget
    if pref == ReasoningPreference.CLIENT:
        if client_budget is not None and client_budget > 0:
            return ResolvedReasoning(enabled=True, budget_tokens=client_budget)
        return ResolvedReasoning(enabled=True, budget_tokens=default_budget)

    # 4. 根据预设数值计算
    try:
        pref_enum = ReasoningPreference(pref)
        if pref_enum in _EFFORT_BUDGET_TOKENS:
            return ResolvedReasoning(enabled=True, budget_tokens=_EFFORT_BUDGET_TOKENS[pref_enum])
    except ValueError:
        pass

    return ResolvedReasoning(enabled=True, budget_tokens=default_budget)
