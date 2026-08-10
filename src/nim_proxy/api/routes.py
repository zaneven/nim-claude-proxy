"""API Routes for Anthropic & OpenAI compatible endpoints."""

import time
import uuid
from dataclasses import replace
from typing import Any, cast

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from loguru import logger

from nim_proxy.config.settings import settings
from nim_proxy.core.agent_loop import (
    openai_final_to_anthropic_message,
    run_with_web_tools,
    stream_message_as_anthropic_sse,
)
from nim_proxy.fcc_port.application.errors import ApplicationError, InvalidRequestError
from nim_proxy.fcc_port.application.execution import ProviderExecutor
from nim_proxy.fcc_port.application.routing import ModelRouter
from nim_proxy.fcc_port.config.nim import NimSettings
from nim_proxy.fcc_port.config.provider_catalog import SUPPORTED_PROVIDER_IDS
from nim_proxy.fcc_port.config.reasoning import ReasoningPreference
from nim_proxy.fcc_port.config.settings import Settings
from nim_proxy.fcc_port.core.anthropic import (
    MessagesRequest,
    TokenCountRequest,
    aggregate_anthropic_sse_to_message,
    anthropic_error_payload,
    anthropic_failure_payload,
    anthropic_status_for_error_type,
    get_token_count,
)
from nim_proxy.fcc_port.core.failures import ExecutionFailure, find_execution_failure
from nim_proxy.fcc_port.core.reasoning import ReasoningControl, ReasoningPolicy
from nim_proxy.fcc_port.providers.admission import ProviderAdmissionController
from nim_proxy.fcc_port.providers.base import ProviderConfig
from nim_proxy.fcc_port.providers.nvidia_nim.client import NvidiaNimProvider
from nim_proxy.fcc_port.core.trace import close_stream_input
from nim_proxy.provider.client import NimClient
from nim_proxy.provider.key_pool import KeyPoolExhaustedError, KeyPoolManager
from nim_proxy.web_tools.executor import is_web_tool

from .detection import is_safety_classifier_request
from .optimization_handlers import try_optimizations
from .streaming import (
    EmptyStreamError,
    ManagedStreamingResponse,
)


router = APIRouter()


# Shared KeyPoolManager
_key_pool = KeyPoolManager(
    api_keys=settings.parsed_api_keys(),
    cooldown_seconds=settings.key_cooldown_seconds,
)

nim_provider = NvidiaNimProvider(
    config=ProviderConfig(
        api_key="dynamic",
        base_url=settings.nvidia_nim_base_url,
        proxy=settings.nvidia_nim_proxy,
        http_read_timeout=settings.http_timeout,
    ),
    nim_settings=NimSettings(),
    admission=ProviderAdmissionController(provider_name="nim", rate_limit=50, rate_window=60.0),
    key_pool=_key_pool
)

_client = NimClient(key_pool=_key_pool)  # retained for OpenAI streaming passthrough and admin test-chat


# Online model discovery cache (TTL)
_ONLINE_MODELS_TTL = 60.0
_online_models_cache: dict[str, Any] = {"ts": 0.0, "models": []}

# ---------------------------------------------------------------------------
# Routing bridge: expose the live proxy settings through the surface that the
# original free-claude-code ModelRouter / ProviderExecutor expect.
# ---------------------------------------------------------------------------


def _nim_model_ref(value: str | None) -> str | None:
    """Qualify a bare NIM model name with the ``nvidia_nim/`` provider prefix."""
    if not value or not value.strip():
        return None
    ref = value.strip()
    head = ref.split("/", 1)[0]
    if head in SUPPORTED_PROVIDER_IDS:
        return ref
    return f"nvidia_nim/{ref}"


def _reasoning_pref(value: str | None, fallback: ReasoningPreference) -> ReasoningPreference:
    try:
        return ReasoningPreference((value or "").strip().lower())
    except ValueError:
        return fallback


class _RoutingSettingsAdapter:
    """Read-through adapter from the live proxy settings to the router surface."""

    @property
    def model(self) -> str:
        return _nim_model_ref(settings.nim_default_model) or "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"

    @property
    def model_fable(self) -> str | None:
        return _nim_model_ref(settings.model_fable)

    @property
    def model_opus(self) -> str | None:
        return _nim_model_ref(settings.model_opus)

    @property
    def model_sonnet(self) -> str | None:
        return _nim_model_ref(settings.model_sonnet)

    @property
    def model_haiku(self) -> str | None:
        return _nim_model_ref(settings.model_haiku)

    @property
    def reasoning_policy(self) -> ReasoningPreference:
        return _reasoning_pref(settings.reasoning_preference, ReasoningPreference.CLIENT)

    @property
    def reasoning_fable(self) -> ReasoningPreference:
        return _reasoning_pref(settings.reasoning_fable, ReasoningPreference.INHERIT)

    @property
    def reasoning_opus(self) -> ReasoningPreference:
        return _reasoning_pref(settings.reasoning_opus, ReasoningPreference.INHERIT)

    @property
    def reasoning_sonnet(self) -> ReasoningPreference:
        return _reasoning_pref(settings.reasoning_sonnet, ReasoningPreference.INHERIT)

    @property
    def reasoning_haiku(self) -> ReasoningPreference:
        return _reasoning_pref(settings.reasoning_haiku, ReasoningPreference.INHERIT)


def _resolve_provider(provider_id: str) -> NvidiaNimProvider:
    if provider_id != "nvidia_nim":
        raise InvalidRequestError(f"Unknown provider_type: '{provider_id}'. Supported: 'nvidia_nim'")
    return nim_provider


_model_router = ModelRouter(cast(Settings, _RoutingSettingsAdapter()))
_generation_counter = 0


def _new_executor() -> ProviderExecutor:
    """Create a request-scoped executor with a unique generation id."""
    global _generation_counter
    _generation_counter += 1
    return ProviderExecutor(_resolve_provider, generation_id=_generation_counter)


# ---------------------------------------------------------------------------
# Error helpers (Anthropic wire format, same semantics as the original)
# ---------------------------------------------------------------------------


def _error_response(status_code: int, error_type: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=anthropic_error_payload(
            error_type=error_type,
            message=message,
            request_id=request_id,
        ),
    )


def _failure_response(failure: ExecutionFailure, request_id: str) -> JSONResponse:
    logger.bind(request_id=request_id).error(
        "Upstream provider failure: kind={kind} status={status} retryable={retryable} message={message}",
        kind=failure.kind.value,
        status=failure.status_code,
        retryable=failure.retryable,
        message=failure.message,
    )
    return JSONResponse(
        status_code=failure.status_code,
        content=anthropic_failure_payload(failure, request_id=request_id),
    )


def _exception_response(exc: BaseException, request_id: str, context: str) -> JSONResponse:
    failure = find_execution_failure(exc)
    if failure is not None:
        return _failure_response(failure, request_id)
    if isinstance(exc, KeyPoolExhaustedError):
        # 所有 Key 冷却中 -> 对客户端表现为上游过载（Claude Code 会按 529 退避重试）
        logger.bind(request_id=request_id).warning("{}: {}", context, exc)
        return _error_response(529, "overloaded_error", str(exc), request_id)
    logger.bind(request_id=request_id).exception("{}: {}", context, type(exc).__name__)
    return _error_response(500, "api_error", f"{type(exc).__name__}", request_id)


def _stream_error_fields(error: dict[str, Any]) -> tuple[str, str]:
    raw_type = error.get("type")
    error_type = raw_type.strip() if isinstance(raw_type, str) and raw_type.strip() else "api_error"
    raw_message = error.get("message")
    message = (
        raw_message.strip()
        if isinstance(raw_message, str) and raw_message.strip()
        else "Provider request failed unexpectedly."
    )
    return error_type, message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
                return [str(m["id"]) for m in online_models if isinstance(m, dict) and m.get("id")]
    except Exception as exc:
        logger.debug(f"Failed to fetch online models: {exc}")
    return []


def _has_local_web_tools(req: MessagesRequest) -> bool:
    tools = req.tools or []
    return (
        bool(tools)
        and any(is_web_tool(t.name) for t in tools)
        and (settings.enable_web_search or settings.enable_web_fetch)
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health")
async def health_check():
    stats = _key_pool.get_stats()
    return {
        "status": "ok",
        "service": "nim-claude-proxy",
        "keys": stats,
        "default_model": settings.nim_default_model,
    }


@router.post("/stop")
async def stop_service():
    """Stop the proxy service gracefully (for CLI/managed lifecycle)."""
    logger.info("Received /stop request — proxy shutting down via request signal.")
    return {"status": "stopped", "message": "Proxy received stop signal."}


@router.get("/v1/models")
@router.get("/models")
async def list_models():
    """Return model list compliant with OpenAI and Anthropic client discovery specs."""
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

    # Online model list with TTL caching
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


async def _stream_optimized_response(msg: Any):
    """Replay an optimized MessagesResponse as Anthropic SSE events."""
    import json
    usage = msg.usage
    inp = int(getattr(usage, "input_tokens", 0) or 0)
    out = int(getattr(usage, "output_tokens", 0) or 0)
    msg_id = str(getattr(msg, "id", "") or f"msg_{uuid.uuid4().hex[:24]}")
    model = str(getattr(msg, "model", "") or "")
    content = getattr(msg, "content", None) or []

    def _evt(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield _evt("message_start", {
        "type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": inp, "output_tokens": 1},
        },
    })

    for i, b in enumerate(content):
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            yield _evt("content_block_start", {
                "type": "content_block_start", "index": i,
                "content_block": {"type": "text", "text": ""},
            })
            yield _evt("content_block_delta", {
                "type": "content_block_delta", "index": i,
                "delta": {"type": "text_delta", "text": b.get("text", "")},
            })
            yield _evt("content_block_stop", {
                "type": "content_block_stop", "index": i,
            })

    yield _evt("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": str(getattr(msg, "stop_reason", None) or "end_turn"),
            "stop_sequence": None,
        },
        "usage": {"input_tokens": inp, "output_tokens": out},
    })
    yield _evt("message_stop", {"type": "message_stop"})


async def _handle_web_tools_request(req: MessagesRequest, request_id: str):
    """Resolve local web tools server-side, then answer in Anthropic format."""
    routed = _model_router.resolve_messages_request(req)
    body = nim_provider._build_request_body(routed.request, reasoning=routed.reasoning)
    body["stream"] = False

    async def _execute(body: dict[str, Any]) -> Any:
        return await nim_provider.execute_chat_non_streaming(body, request_id=request_id)

    final = await run_with_web_tools(_execute, body)
    message = openai_final_to_anthropic_message(final, routed.resolved.original_model)
    if req.stream:
        return ManagedStreamingResponse(
            stream_message_as_anthropic_sse(message),
            request_id=request_id,
        )
    return JSONResponse(content=message)


@router.post("/v1/messages")
async def handle_anthropic_messages(req: MessagesRequest):
    """Anthropic Messages API (/v1/messages) Handler for Claude Code & Pi.

    Mirrors the original free-claude-code pipeline exactly: model routing with
    route-specific reasoning policies, provider preflight, the streaming
    engine as the single execution path, and SSE aggregation for
    non-streaming clients.
    """
    request_id = f"msg_{uuid.uuid4().hex[:24]}"
    lg = logger.bind(request_id=request_id)
    lg.info(
        "REQ model={} stream={} msg_count={} tools={} max_tokens={}",
        req.model, req.stream, len(req.messages), len(req.tools or []), req.max_tokens,
    )
    try:
        if not req.messages:
            return _error_response(
                400, "invalid_request_error", "messages: at least one message is required", request_id
            )

        # Local web tools are resolved server-side (proxy-specific feature);
        # everything else flows through the original engine pipeline.
        if _has_local_web_tools(req):
            return await _handle_web_tools_request(req, request_id)

        routed = _model_router.resolve_messages_request(req)

        # Safety classifier: Claude Code's auto-mode safety prompt should never
        # consume reasoning budget — it's a simple yes/no classification.
        if is_safety_classifier_request(req):
            if routed.reasoning.control is not ReasoningControl.OFF:
                routed = replace(routed, reasoning=ReasoningPolicy.off())

        # Fast-path: intercept known Claude Code housekeeping calls
        # (title generation, quota probe, prefix detection, suggestion mode,
        # filepath extraction) and serve them locally without touching NIM.
        optimized = try_optimizations(routed.request, settings, response_model=routed.resolved.original_model)
        if optimized is not None:
            if req.stream:
                return ManagedStreamingResponse(
                    _stream_optimized_response(optimized),
                    request_id=request_id,
                )
            return JSONResponse(content=cast(dict, optimized.model_dump()))

        stream = _new_executor().stream(
            routed,
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload=routed.request.model_dump(),
            request_id=request_id,
        )

        if not req.stream:
            # Non-streaming clients get the engine's SSE stream aggregated
            # into a complete Message (identical to the original handler).
            try:
                message, error = await aggregate_anthropic_sse_to_message(stream)
            except ExecutionFailure as exc:
                return _failure_response(exc, request_id)
            except Exception as exc:
                return _exception_response(exc, request_id, "CREATE_MESSAGE_NON_STREAM_ERROR")
            if error is not None:
                error_type, message_text = _stream_error_fields(error)
                return _error_response(
                    anthropic_status_for_error_type(error_type), error_type, message_text, request_id
                )
            return JSONResponse(content=message)

        # Streaming: surface failures that happen before the first byte as
        # JSON errors (same contract as the original pre-start handling).
        try:
            first_chunk = await anext(stream)
        except StopAsyncIteration:
            await close_stream_input(
                stream, owner="routes", source="api",
                preserved_error=EmptyStreamError("Upstream provider returned an empty stream."),
            )
            return _error_response(500, "api_error", "Upstream provider returned an empty stream.", request_id)
        except ExecutionFailure as exc:
            await close_stream_input(
                stream, owner="routes", source="api", preserved_error=exc,
            )
            return _failure_response(exc, request_id)
        except Exception as exc:
            await close_stream_input(
                stream, owner="routes", source="api", preserved_error=exc,
            )
            return _exception_response(exc, request_id, "CREATE_MESSAGE_STREAM_START_ERROR")

        async def body():
            yield first_chunk
            async for chunk in stream:
                yield chunk

        return ManagedStreamingResponse(body(), request_id=request_id)

    except ApplicationError as exc:
        status_code = getattr(exc, "status_code", 400)
        error_type = "invalid_request_error" if status_code == 400 else "api_error"
        return _error_response(status_code, error_type, str(exc), request_id)
    except ExecutionFailure as exc:
        return _failure_response(exc, request_id)
    except Exception as exc:
        return _exception_response(exc, request_id, "CREATE_MESSAGE_ERROR")


@router.post("/v1/messages/count_tokens")
async def handle_count_tokens(req: TokenCountRequest):
    """Anthropic token-count API (same routing + counting as the original)."""
    request_id = f"msg_{uuid.uuid4().hex[:24]}"
    try:
        if not req.messages:
            return _error_response(
                400, "invalid_request_error", "messages: at least one message is required", request_id
            )
        routed = _model_router.resolve_token_count_request(req)
        tokens = get_token_count(routed.request.messages, routed.request.system, routed.request.tools)
        return {"input_tokens": tokens}
    except ApplicationError as exc:
        status_code = getattr(exc, "status_code", 400)
        error_type = "invalid_request_error" if status_code == 400 else "api_error"
        return _error_response(status_code, error_type, str(exc), request_id)
    except Exception as exc:
        return _exception_response(exc, request_id, "COUNT_TOKENS_ERROR")


@router.post("/v1/chat/completions")
@router.post("/v1/responses")
async def handle_openai_chat(request: Request):
    """OpenAI Chat Completions / Responses Handler.

    /v1/responses is converted from the OpenAI Responses shape into the
    Chat Completions shape, executed against NIM, then converted back
    into a minimal Responses-shaped payload.
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
            # Streaming OpenAI: pass through NimClient which handles key
            # rotation, rate limiting, and native tool cleaning natively.
            # Full engine routing would require Anthropic→OpenAI SSE conversion;
            # the NimClient passthrough is pragmatic and battle-tested.
            try:
                res = await _client.execute_chat(body)
            except Exception as exc:
                return _exception_response(exc, f"msg_{uuid.uuid4().hex[:24]}", "OPENAI_CHAT_STREAM")
            return res

        # Non-streaming: resolve local web tools through the admission-controlled provider
        body["stream"] = False

        async def _execute_non_streaming(b: dict[str, Any]) -> Any:
            return await nim_provider.execute_chat_non_streaming(b)

        oai_response = await run_with_web_tools(_execute_non_streaming, body)
        if is_responses:
            from nim_proxy.core.openai import openai_response_to_responses
            return openai_response_to_responses(oai_response, model_name)
        return oai_response
    except Exception as err:
        logger.error(f"Error handling OpenAI chat completions: {err}")
        return _error_response(500, "api_error", str(err), f"msg_{uuid.uuid4().hex[:24]}")
