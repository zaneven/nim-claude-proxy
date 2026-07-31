"""Retry helpers for stripping rejected attributes when NVIDIA NIM returns 400 errors."""

import copy

def clone_body_without_reasoning_budget(body: dict) -> dict | None:
    """Strip reasoning_budget or extra_body thinking options if rejected by NIM."""
    cloned = copy.deepcopy(body)
    modified = False

    if "reasoning_budget" in cloned:
        del cloned["reasoning_budget"]
        modified = True

    extra = cloned.get("extra_body")
    if isinstance(extra, dict):
        if "reasoning_budget" in extra:
            del extra["reasoning_budget"]
            modified = True
        if "thinking" in extra:
            del extra["thinking"]
            modified = True
        if "reasoning_config" in extra:
            del extra["reasoning_config"]
            modified = True

    return cloned if modified else None


def clone_body_without_chat_template(body: dict) -> dict | None:
    """Strip chat_template option if rejected by NIM."""
    cloned = copy.deepcopy(body)
    modified = False

    if "chat_template" in cloned:
        del cloned["chat_template"]
        modified = True

    extra = cloned.get("extra_body")
    if isinstance(extra, dict) and "chat_template" in extra:
        del extra["chat_template"]
        modified = True

    return cloned if modified else None
