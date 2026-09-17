"""Admin API Endpoints for Dashboard Control."""

import os
import sys
import time
import httpx
from typing import Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger
from pydantic import BaseModel

from nim_proxy.config.persistence import _sync_keys_line, apply_settings_update
from nim_proxy.config.settings import settings
from nim_proxy.core.metrics import get_metrics_store
from nim_proxy.provider.key_pool import KeyPoolManager

admin_router = APIRouter(prefix="/admin/api")

_key_pool: KeyPoolManager | None = None
_provider: Any = None  # NvidiaNimProvider for test-chat

FALLBACK_NIM_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.3-70b-instruct",
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "mistralai/mistral-7b-instruct-v0.3",
    "nvidia/nemotron-4-340b-instruct",
]

# ── Model cache with 60s TTL ──
_model_cache: dict[str, Any] = {"models": [], "source": "fallback", "timestamp": 0.0}
_MODEL_CACHE_TTL = 60.0


def init_admin_globals(key_pool: KeyPoolManager, provider: Any = None):
    global _key_pool, _provider
    _key_pool = key_pool
    _provider = provider


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
    fast_prefix_detection: bool | None = None
    enable_network_probe_mock: bool | None = None
    enable_title_generation_skip: bool | None = None
    enable_suggestion_mode_skip: bool | None = None
    enable_filepath_extraction_mock: bool | None = None


class TestChatRequest(BaseModel):
    prompt: str = "Hello! Please send a 1-sentence status check."
    model: str | None = None


class KeyAddRequest(BaseModel):
    api_key: str


class KeyEditRequest(BaseModel):
    old_key: str
    new_key: str


class KeyDeleteRequest(BaseModel):
    api_key: str


@admin_router.get("/config")
async def get_admin_config():
    """Return all proxy settings and current KeyPool status."""
    keys_masked = []
    if _key_pool:
        for k in _key_pool._keys:
            keys_masked.append({
                "key": k.masked_key,
                "key_masked": k.masked_key,
                "raw_key": k.api_key,
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
            "FAST_PREFIX_DETECTION": settings.fast_prefix_detection,
            "ENABLE_NETWORK_PROBE_MOCK": settings.enable_network_probe_mock,
            "ENABLE_TITLE_GENERATION_SKIP": settings.enable_title_generation_skip,
            "ENABLE_SUGGESTION_MODE_SKIP": settings.enable_suggestion_mode_skip,
            "ENABLE_FILEPATH_EXTRACTION_MOCK": settings.enable_filepath_extraction_mock,
        },
        "key_pool_stats": _key_pool.get_stats() if _key_pool else {},
        "key_details": keys_masked,
    }


@admin_router.get("/models")
async def discover_nim_models(force_refresh: bool = False):
    """Fetch discovered online NVIDIA NIM models from upstream API using configured KeyPool."""
    if not force_refresh and _model_cache["timestamp"] and (time.monotonic() - _model_cache["timestamp"]) < _MODEL_CACHE_TTL:
        return {"models": _model_cache["models"], "source": _model_cache["source"] + " (cached)"}

    if not _key_pool:
        return {"models": FALLBACK_NIM_MODELS, "source": "fallback"}

    try:
        api_key = await _key_pool.get_next_key()
    except Exception as exc:
        logger.debug(f"Failed to get next key for model discovery: {exc}")
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
                    _model_cache["models"] = models
                    _model_cache["source"] = "live"
                    _model_cache["timestamp"] = time.monotonic()
                    return {"models": models, "source": "live"}
    except Exception:
        pass

    # Populate cache with fallback so we don't keep hitting the API on failure
    _model_cache["models"] = FALLBACK_NIM_MODELS
    _model_cache["source"] = "fallback"
    _model_cache["timestamp"] = time.monotonic()
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


@admin_router.get("/stats")
async def get_keypool_stats():
    """Lightweight endpoint for dashboard polling — key stats only, no settings or raw keys."""
    keys_light = []
    now = time.monotonic()
    if _key_pool:
        for k in _key_pool._keys:
            remaining = max(0.0, k.cooldown_until - now) if k.cooldown_until > now else 0.0
            keys_light.append({
                "key": k.masked_key,
                "key_masked": k.masked_key,
                "is_disabled": k.is_disabled,
                "cooldown_remaining": int(remaining + 0.5),
                "failure_count": k.failure_count,
            })

    return {
        "key_pool_stats": _key_pool.get_stats() if _key_pool else {},
        "key_details": keys_light,
    }


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


@admin_router.get("/usage/summary")
async def get_usage_summary():
    """Aggregated token / request statistics for the dashboard."""
    return get_metrics_store().summary()


@admin_router.get("/usage/records")
async def get_usage_records(
    page: int = 1,
    page_size: int = 50,
    status: str | None = None,
    endpoint: str | None = None,
    model: str | None = None,
):
    """Paginated request-forwarding records."""
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 500:
        page_size = 50
    return get_metrics_store().records(
        page=page,
        page_size=page_size,
        status=status or None,
        endpoint=endpoint or None,
        model=model or None,
    )


@admin_router.post("/usage/reset")
async def reset_usage_metrics():
    """Clear all recorded usage metrics and forwarding records."""
    get_metrics_store().reset()
    return {"status": "success", "message": "Usage metrics cleared."}


@admin_router.post("/keys")
async def add_key(req: KeyAddRequest):
    """新增单个 NIM API key。"""
    if not _key_pool:
        raise HTTPException(status_code=500, detail="KeyPool uninitialized.")
    clean = req.api_key.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Key cannot be empty.")
    # 防重复
    for k in _key_pool._keys:
        if k.api_key == clean:
            raise HTTPException(status_code=409, detail="Key already exists.")
    # 加入 pool
    await _key_pool.add_keys([clean])
    # 同步 .env
    all_keys = [k.api_key for k in _key_pool._keys]
    _sync_keys_line(all_keys)
    # 更新 settings 内存
    settings.nvidia_nim_api_keys = ",".join(all_keys)
    return {"status": "success", "message": "Key added."}


@admin_router.put("/keys")
async def edit_key(req: KeyEditRequest):
    """编辑（替换）一个 NIM API key。"""
    if not _key_pool:
        raise HTTPException(status_code=500, detail="KeyPool uninitialized.")
    old = req.old_key.strip()
    new = req.new_key.strip()
    if not old or not new:
        raise HTTPException(status_code=400, detail="Both old_key and new_key are required.")
    if old == new:
        raise HTTPException(status_code=400, detail="old_key and new_key are identical.")
    # 检查 old_key 是否存在
    found = any(k.api_key == old for k in _key_pool._keys)
    if not found:
        raise HTTPException(status_code=404, detail="Key not found.")
    # 检查 new_key 没有重复
    for k in _key_pool._keys:
        if k.api_key == new:
            raise HTTPException(status_code=409, detail="New key already exists.")
    # 替换
    ok = await _key_pool.replace_key(old, new)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to replace key.")
    # 同步 .env
    all_keys = [k.api_key for k in _key_pool._keys]
    _sync_keys_line(all_keys)
    settings.nvidia_nim_api_keys = ",".join(all_keys)
    return {"status": "success", "message": "Key updated."}


@admin_router.delete("/keys")
async def delete_key(req: KeyDeleteRequest):
    """删除一个 NIM API key。"""
    if not _key_pool:
        raise HTTPException(status_code=500, detail="KeyPool uninitialized.")
    target = req.api_key.strip()
    if not target:
        raise HTTPException(status_code=400, detail="api_key is required.")
    ok = await _key_pool.remove_key(target)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found.")
    # 同步 .env
    all_keys = [k.api_key for k in _key_pool._keys]
    _sync_keys_line(all_keys)
    settings.nvidia_nim_api_keys = ",".join(all_keys)
    return {"status": "success", "message": "Key deleted."}


@admin_router.post("/test_chat_stream")
async def test_chat_stream(req: TestChatRequest):
    """Run a streaming test prompt against NIM — returns simple text/plain SSE chunks."""
    from fastapi.responses import StreamingResponse

    if not _provider:
        raise HTTPException(status_code=500, detail="Provider uninitialized.")

    model_target = req.model or settings.nim_default_model
    body = {
        "model": model_target,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": 256,
        "stream": True,
    }

    try:
        stream = await _provider.execute_chat(body)
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

    async def generate():
        try:
            async for chunk in stream:
                # chunk is an openai ChatCompletionChunk; extract delta content
                content = None
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as err:
            yield f"\n[STREAM ERROR: {err}]"

    return StreamingResponse(generate(), media_type="text/plain")


@admin_router.post("/test_chat")
async def test_chat(req: TestChatRequest):
    """Run a test prompt against NIM through the admission-controlled provider."""
    if not _provider:
        raise HTTPException(status_code=500, detail="Provider uninitialized.")

    model_target = req.model or settings.nim_default_model
    body = {
        "model": model_target,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": 100,
        "stream": False,
    }

    try:
        res = await _provider.execute_chat_non_streaming(body)
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

def _do_restart():
    logger.warning("Executing os.execv to restart the service...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

@admin_router.post("/restart")
async def restart_service(background_tasks: BackgroundTasks):
    """Restart the proxy service by replacing the current process."""
    logger.info("Received restart request via Admin API.")
    # We schedule the actual restart as a background task so the response can be returned to the client first.
    background_tasks.add_task(_do_restart)
    return {"status": "success", "message": "Service is restarting..."}

