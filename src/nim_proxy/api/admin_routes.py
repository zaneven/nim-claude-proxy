"""Admin API Endpoints for Dashboard Control."""

import httpx
from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nim_proxy.config.persistence import apply_settings_update
from nim_proxy.config.settings import settings
from nim_proxy.provider.client import NimClient
from nim_proxy.provider.key_pool import KeyPoolManager

admin_router = APIRouter(prefix="/admin/api")

_key_pool: KeyPoolManager | None = None
_client: NimClient | None = None

FALLBACK_NIM_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "mistralai/mistral-7b-instruct-v0.3",
    "nvidia/nemotron-4-340b-instruct",
]


def init_admin_globals(key_pool: KeyPoolManager, client: NimClient):
    global _key_pool, _client
    _key_pool = key_pool
    _client = client


class ConfigSaveRequest(BaseModel):
    nvidia_nim_api_keys: str | None = None
    nvidia_nim_base_url: str | None = None
    nvidia_nim_proxy: str | None = None
    key_cooldown_seconds: float | None = None
    nim_default_model: str | None = None
    model_sonnet: str | None = None
    model_opus: str | None = None
    model_haiku: str | None = None
    model_fable: str | None = None
    reasoning_preference: str | None = None
    reasoning_sonnet: str | None = None
    reasoning_opus: str | None = None
    reasoning_haiku: str | None = None
    reasoning_fable: str | None = None
    default_reasoning_budget: int | None = None
    enable_web_search: bool | None = None
    enable_web_fetch: bool | None = None
    web_search_provider: str | None = None
    duckduckgo_max_results: int | None = None
    tavily_api_key: str | None = None
    doubao_search_api_key: str | None = None
    port: int | None = None
    host: str | None = None
    proxy_api_token: str | None = None
    http_timeout: float | None = None
    log_level: str | None = None
    rate_limit_rpm: int | None = None


class TestChatRequest(BaseModel):
    prompt: str = "Hello! Please send a 1-sentence status check."
    model: str | None = None


@admin_router.get("/config")
async def get_admin_config():
    """Return all proxy settings and current KeyPool status."""
    keys_masked = []
    if _key_pool:
        for k in _key_pool._keys:
            keys_masked.append({
                "key": k.masked_key,
                "is_disabled": k.is_disabled,
                "cooldown_until": k.cooldown_until,
                "failure_count": k.failure_count,
            })

    return {
        "settings": {
            "PORT": settings.port,
            "HOST": settings.host,
            "PROXY_API_TOKEN": settings.proxy_api_token,
            "HTTP_TIMEOUT": settings.http_timeout,
            "NVIDIA_NIM_API_KEYS": settings.nvidia_nim_api_keys,
            "NVIDIA_NIM_BASE_URL": settings.nvidia_nim_base_url,
            "NVIDIA_NIM_PROXY": settings.nvidia_nim_proxy,
            "KEY_COOLDOWN_SECONDS": settings.key_cooldown_seconds,
            "NIM_DEFAULT_MODEL": settings.nim_default_model,
            "MODEL_SONNET": settings.model_sonnet,
            "MODEL_OPUS": settings.model_opus,
            "MODEL_HAIKU": settings.model_haiku,
            "MODEL_FABLE": settings.model_fable,
            "REASONING_PREFERENCE": settings.reasoning_preference,
            "REASONING_SONNET": settings.reasoning_sonnet,
            "REASONING_OPUS": settings.reasoning_opus,
            "REASONING_HAIKU": settings.reasoning_haiku,
            "REASONING_FABLE": settings.reasoning_fable,
            "DEFAULT_REASONING_BUDGET": settings.default_reasoning_budget,
            "RATE_LIMIT_RPM": settings.rate_limit_rpm,
            "ENABLE_WEB_SEARCH": settings.enable_web_search,
            "ENABLE_WEB_FETCH": settings.enable_web_fetch,
            "WEB_SEARCH_PROVIDER": settings.web_search_provider,
            "DUCKDUCKGO_MAX_RESULTS": settings.duckduckgo_max_results,
            "TAVILY_API_KEY": settings.tavily_api_key,
            "DOUBAO_SEARCH_API_KEY": settings.doubao_search_api_key,
            "LOG_LEVEL": settings.log_level,
            "ENABLE_NATIVE_TOOL_CLEANING": settings.enable_native_tool_cleaning,
        },
        "key_pool_stats": _key_pool.get_stats() if _key_pool else {},
        "key_details": keys_masked,
    }


@admin_router.get("/models")
async def discover_nim_models():
    """Fetch discovered online NVIDIA NIM models from upstream API using configured KeyPool."""
    if not _key_pool:
        return {"models": FALLBACK_NIM_MODELS, "source": "fallback"}

    try:
        api_key = await _key_pool.get_next_key()
    except Exception:
        return {"models": FALLBACK_NIM_MODELS, "source": "fallback"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    url = f"{settings.nvidia_nim_base_url.rstrip('/')}/models"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
                models.sort()
                if models:
                    return {"models": models, "source": "live"}
    except Exception:
        pass

    return {"models": FALLBACK_NIM_MODELS, "source": "fallback"}


@admin_router.post("/config")
async def save_admin_config(req: ConfigSaveRequest):
    """Save configuration to .env and hot-reload."""
    to_save = {}
    for k, v in req.model_dump(exclude_none=True).items():
        if isinstance(v, bool):
            to_save[k.upper()] = "true" if v else "false"
        else:
            to_save[k.upper()] = str(v)

    await apply_settings_update(to_save, _key_pool)

    return {"status": "success", "message": "Settings updated and saved to .env"}


@admin_router.post("/keypool/reset")
async def reset_keypool():
    """Reset cooldown timers and disabled states for all keys."""
    if _key_pool:
        for k in _key_pool._keys:
            k.is_disabled = False
            k.cooldown_until = 0.0
            k.failure_count = 0
        return {"status": "success", "message": "All keys reset to healthy state."}
    raise HTTPException(status_code=500, detail="KeyPool uninitialized.")


@admin_router.post("/test_chat")
async def test_chat(req: TestChatRequest):
    """Run a test prompt against NIM using current KeyPool."""
    if not _client:
        raise HTTPException(status_code=500, detail="Client uninitialized.")

    model_target = req.model or settings.nim_default_model
    body = {
        "model": model_target,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": 100,
    }

    try:
        res = await _client.execute_chat(body)
        content = res.choices[0].message.content
        return {
            "status": "success",
            "model_used": model_target,
            "response": content,
        }
    except Exception as err:
        return {
            "status": "error",
            "message": str(err),
        }
