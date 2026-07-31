"""API Routes for Anthropic & OpenAI compatible endpoints."""

import json
import time
import uuid
import httpx
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from nim_proxy.api.sse import format_anthropic_sse_stream
from nim_proxy.config.settings import settings
from nim_proxy.core.anthropic import MessagesRequest, anthropic_request_to_openai_body
from nim_proxy.core.agent_loop import run_with_web_tools, stream_text_as_anthropic_sse
from nim_proxy.web_tools.executor import is_web_tool
from nim_proxy.provider.client import NimClient
from nim_proxy.provider.key_pool import KeyPoolManager
from nim_proxy.provider.schema import unalias_tool_call_arguments

router = APIRouter()

# 共享 KeyPoolManager 与 NimClient 实例
_key_pool = KeyPoolManager(
    api_keys=settings.parsed_api_keys(),
    cooldown_seconds=settings.key_cooldown_seconds,
)
_client = NimClient(key_pool=_key_pool)

# 在线模型发现结果缓存（TTL）
_ONLINE_MODELS_TTL = 60.0
_online_models_cache: dict[str, Any] = {"ts": 0.0, "models": []}


async def _fetch_online_models() -> list[str]:
    """Fetch the live model list from the upstream NIM endpoint (best-effort)."""
    try:
        api_key = await _key_pool.get_next_key()
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        url = f"{settings.nvidia_nim_base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                online_models = resp.json().get("data", [])
                return [m.get("id") for m in online_models if isinstance(m, dict) and m.get("id")]
    except Exception:
        pass
    return []


def _extract_text(response) -> str:
    """Pull assistant text out of a (non-streaming) OpenAI chat response."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


@router.get("/health")
async def health_check():
    stats = _key_pool.get_stats()
    return {
        "status": "ok",
        "service": "nim-claude-proxy",
        "keys": stats,
        "default_model": settings.nim_default_model,
    }


@router.get("/v1/models")
@router.get("/models")
async def list_models():
    """Return model list compliant with OpenAI and Anthropic client discovery specifications."""
    known_models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-latest",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-3-7-sonnet-20250219",
        settings.nim_default_model,
    ]

    if settings.model_sonnet:
        known_models.append(settings.model_sonnet)
    if settings.model_opus:
        known_models.append(settings.model_opus)
    if settings.model_haiku:
        known_models.append(settings.model_haiku)

    # 在线模型列表带 TTL 缓存，避免每次请求都实时拉取上游
    now = time.time()
    if now - _online_models_cache["ts"] >= _ONLINE_MODELS_TTL:
        _online_models_cache["models"] = await _fetch_online_models()
        _online_models_cache["ts"] = now
    for m in _online_models_cache["models"]:
        if m and m not in known_models:
            known_models.append(m)

    unique_models = list(dict.fromkeys([m for m in known_models if m.strip()]))

    model_objects = [
        {
            "id": m,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nim-proxy",
        }
        for m in unique_models
    ]

    return {
        "object": "list",
        "data": model_objects,
    }


@router.post("/v1/messages")
async def handle_anthropic_messages(req: MessagesRequest):
    """Anthropic Messages API (/v1/messages) Handler for Claude Code & Pi."""
    body = anthropic_request_to_openai_body(req)

    try:
        if req.stream:
            web_tools_present = bool(req.tools) and any(
                is_web_tool(t.name) for t in req.tools
            ) and (settings.enable_web_search or settings.enable_web_fetch)

            if web_tools_present:
                # 流式请求含网络工具时无法中途拦截，改为服务端缓冲解析后回放最终文本
                body["stream"] = False
                final = await run_with_web_tools(_client, body)
                return StreamingResponse(
                    stream_text_as_anthropic_sse(_extract_text(final), req.model),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            raw_stream = await _client.execute_chat(body)
            return StreamingResponse(
                format_anthropic_sse_stream(raw_stream, req.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        
        # 非流式：透明解析 web_search/web_fetch 本地工具后处理
        response = await run_with_web_tools(_client, body)
        choice = response.choices[0]
        msg = choice.message

        content_blocks = []
        
        # 1. 提取原生 reasoning 字段
        reasoning = getattr(msg, "reasoning_content", getattr(msg, "reasoning", None))
        if reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})

        # 2. 从内容中解析 <think> 标签
        if msg.content:
            from nim_proxy.api.sse import ThinkTagParser
            parser = ThinkTagParser()
            for chunk_type, content in parser.feed(msg.content):
                if chunk_type == "thinking":
                    content_blocks.append({"type": "thinking", "thinking": content})
                else:
                    content_blocks.append({"type": "text", "text": content})
            
            remaining = parser.flush()
            if remaining:
                r_type, r_content = remaining
                if r_type == "thinking":
                    content_blocks.append({"type": "thinking", "thinking": r_content})
                else:
                    content_blocks.append({"type": "text", "text": r_content})

        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                
                # 解换 unsafe 别名参数（如 _nim_arg_type -> type）
                clean_args = unalias_tool_call_arguments(args)

                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": clean_args if isinstance(clean_args, dict) else args,
                })

        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"

        return {
            "id": f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") else 0,
                "output_tokens": getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") else 0,
            }
        }

    except Exception as err:
        logger.error(f"Error handling /v1/messages: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


@router.post("/v1/chat/completions")
@router.post("/v1/responses")
async def handle_openai_chat(request: Request):
    """OpenAI Chat Completions / Responses Handler.

    /v1/responses is converted from the OpenAI Responses shape into the
    Chat Completions shape (see ``responses_request_to_openai_body``), executed
    against NIM, then converted back into a minimal Responses-shaped payload.
    """
    try:
        body = await request.json()
        is_responses = (request.url.path == "/v1/responses")
        if is_responses:
            from nim_proxy.core.openai import responses_request_to_openai_body
            body = responses_request_to_openai_body(body)
            model_name = body.get("model", settings.nim_default_model)
        else:
            model_name = body.get("model", settings.nim_default_model)

        body["model"] = settings.get_model_for_alias(model_name)

        resolved_reasoning = settings.get_reasoning_for_model(model_name)
        if resolved_reasoning.enabled and resolved_reasoning.budget_tokens:
            body.setdefault("extra_body", {})["reasoning_budget"] = resolved_reasoning.budget_tokens

        stream = bool(body.get("stream", False))
        if stream:
            # 流式 Responses 暂按透传处理
            res = await _client.execute_chat(body)
            return res

        # 非流式：解析本地 web 工具后执行
        body["stream"] = False
        oai_response = await run_with_web_tools(_client, body)
        if is_responses:
            from nim_proxy.core.openai import openai_response_to_responses
            return openai_response_to_responses(oai_response, model_name)
        return oai_response
    except Exception as err:
        logger.error(f"Error handling OpenAI chat completions: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err
