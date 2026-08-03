from __future__ import annotations

"""API Routes for Anthropic & OpenAI compatible endpoints."""

import json
import time
import uuid
import httpx
from typing import Any

import openai
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from nim_proxy.api.sse import (
    SseHoldback,
    _error_cleanup_events,
    format_anthropic_sse_stream,
    sse_error_event,
)
from nim_proxy.config.settings import settings
from nim_proxy.core.anthropic import MessagesRequest, anthropic_request_to_openai_body
from nim_proxy.core.agent_loop import run_with_web_tools, stream_text_as_anthropic_sse
from nim_proxy.web_tools.executor import is_web_tool
from nim_proxy.provider.client import NimClient
from nim_proxy.provider.key_pool import KeyPoolManager
from nim_proxy.provider.schema import unalias_tool_call_arguments
from nim_proxy.provider.stream_recovery import (
    continuation_suffix,
    make_text_continuation_body,
    make_tool_repair_body,
    accept_tool_json_repair,
)

router = APIRouter()

# Shared KeyPoolManager & NimClient instances
_key_pool = KeyPoolManager(
    api_keys=settings.parsed_api_keys(),
    cooldown_seconds=settings.key_cooldown_seconds,
)
_client = NimClient(key_pool=_key_pool)

# Online model discovery cache (TTL)
_ONLINE_MODELS_TTL = 60.0
_online_models_cache: dict[str, Any] = {"ts": 0.0, "models": []}

# Early transparent retry configuration
_EARLY_TRANSPARENT_RETRIES = 3
_MIDSTREAM_RECOVERY_ATTEMPTS = 2


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
    except Exception as exc:
        logger.debug(f"Failed to fetch online models: {exc}")
    return []


def _extract_text(response) -> str:
    """Pull assistant text out of a (non-streaming) OpenAI chat response."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


# ── Stream recovery helpers ────────────────────────────────────────────


def _make_stub_ledger(model: str):
    """Return a minimal object that supports close_all_open_blocks()."""
    class _Stub:
        has_tool_blocks = False
        def close_all_open_blocks(self):
            return []
    return _Stub()


def _tool_schemas_from_body(body: dict[str, Any]) -> dict[str, Any]:
    """Extract tool name → input_schema mapping from a request body."""
    schemas: dict[str, Any] = {}
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            params = fn.get("parameters")
            if isinstance(name, str) and isinstance(params, dict):
                schemas[name] = params
    return schemas


async def _try_midstream_recovery(
    body: dict[str, Any],
    accumulated_text: str,
    accumulated_reasoning: str,
    tool_blocks: list[dict[str, Any]],
    request_id: str,
) -> list[str] | None:
    """Attempt midstream recovery: text continuation or tool repair.

    Returns a list of SSE event strings or None if recovery failed.
    """
    from nim_proxy.api.sse import _sse_event

    events: list[str] = []

    # ── Tool repair: try to complete incomplete tool calls ──
    if tool_blocks:
        schemas = _tool_schemas_from_body(body)
        all_repaired = True
        for tb in tool_blocks:
            if tb.get("stopped"):
                continue
            name = tb["name"]
            prefix = tb["content"]
            recovery_body = make_tool_repair_body(body, name, prefix, schemas.get(name))
            if recovery_body is None:
                all_repaired = False
                continue
            try:
                text, _ = await _client.collect_continuation_text(
                    recovery_body, max_attempts=_MIDSTREAM_RECOVERY_ATTEMPTS
                )
                suffix = accept_tool_json_repair(prefix, text, name, schemas)
                if suffix:
                    # Calculate the correct block index for this tool
                    events.append(_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tb["block_index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": suffix,
                        },
                    }))
                    logger.info(
                        "Tool repair succeeded for '{}' ({} chars recovered)",
                        name, len(suffix),
                    )
                else:
                    all_repaired = False
                    logger.warning(
                        "Tool repair could not parse JSON for '{}'", name,
                    )
            except Exception as err:
                all_repaired = False
                logger.warning(
                    "Tool repair request failed for '{}': {}", name, err,
                )
        if all_repaired and events:
            logger.info("Full tool repair succeeded for all incomplete tool calls")
            return events
        if not all_repaired and events:
            logger.warning("Partial tool repair: some tool calls could not be recovered")

    # ── Text continuation: try to continue truncated text ──
    if accumulated_text or accumulated_reasoning:
        continuation_body = make_text_continuation_body(body, accumulated_text)
        if continuation_body is not None:
            try:
                text, reasoning = await _client.collect_continuation_text(
                    continuation_body, max_attempts=_MIDSTREAM_RECOVERY_ATTEMPTS
                )
                text_suffix = continuation_suffix(accumulated_text, text)
                thinking_suffix = continuation_suffix(accumulated_reasoning, reasoning)

                if thinking_suffix:
                    events.append(
                        _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": thinking_suffix,
                            },
                        })
                    )
                if text_suffix:
                    events.append(
                        _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 1,
                            "delta": {
                                "type": "text_delta",
                                "text": text_suffix,
                            },
                        })
                    )
                if events:
                    logger.info(
                        "Text continuation recovered text_suffix={} thinking_suffix={}",
                        len(text_suffix),
                        len(thinking_suffix),
                    )
                return events
            except Exception as err:
                logger.warning("Text continuation request failed: {}", err)

    return None


# ── Core streaming async generator with recovery ───────────────────────


async def _stream_with_recovery(
    body: dict[str, Any],
    model: str,
    *,
    request_id: str = "",
) -> Any:
    """Stream with early transparent retries and midstream recovery.

    - If the NIM stream fails before any content_block_start, retry silently
      (the client sees nothing).
    - If the stream fails after content has been emitted, try text continuation
      or tool JSON repair.
    - If all recovery fails, inject an inline error SSE event and close
      gracefully.
    """
    tag = f" request_id={request_id}" if request_id else ""

    for attempt in range(_EARLY_TRANSPARENT_RETRIES):
        holdback = SseHoldback()
        ledger_ref: list[Any] = []
        committed_events_sent = False

        try:
            raw_stream = await _client.execute_chat({**body, "stream": True})

            async for event in format_anthropic_sse_stream(
                raw_stream,
                model,
                holdback=holdback,
                _ledger_ref=ledger_ref,
            ):
                committed_events_sent = True
                yield event

            # Stream completed successfully — flush any deferred events
            for event in holdback.flush():
                yield event
            return

        except (openai.APIError, openai.APIConnectionError) as err:
            committed = holdback.committed or committed_events_sent

            if not committed:
                # Early transparent retry — nothing was shown to the client
                logger.warning(
                    "Stream attempt {}/{} failed before any content_block_start{}: {}",
                    attempt + 1,
                    _EARLY_TRANSPARENT_RETRIES,
                    tag,
                    err,
                )
                holdback.discard()
                if attempt < _EARLY_TRANSPARENT_RETRIES - 1:
                    continue

            # Midstream failure — content was already committed to client
            logger.error(
                "Stream failed mid-content at attempt {}/{}{}: {}",
                attempt + 1,
                _EARLY_TRANSPARENT_RETRIES,
                tag,
                err,
            )

            # Try recovery if we had committed content
            if committed:
                try:
                    ledger = ledger_ref[0] if ledger_ref else None
                    recovery_events = await _try_midstream_recovery(
                        body=body,
                        accumulated_text=ledger.accumulated_text if ledger else "",
                        accumulated_reasoning=ledger.accumulated_reasoning if ledger else "",
                        tool_blocks=ledger.tool_block_snapshot() if ledger else [],
                        request_id=request_id,
                    )
                except Exception:
                    recovery_events = None

                if recovery_events:
                    for event in recovery_events:
                        yield event

            # Emit error and close cleanly
            error_msg = f"NIM stream failed: {err}"
            for event in _error_cleanup_events(
                _make_stub_ledger(model), error_msg,
            ):
                yield event
            return

        except Exception as err:
            # Non-retryable error — inject error inline
            logger.error("Non-retryable stream error{}: {}", tag, err)
            if holdback.committed or committed_events_sent:
                for event in _error_cleanup_events(
                    _make_stub_ledger(model),
                    f"Stream error: {err}",
                ):
                    yield event
            else:
                yield sse_error_event(f"Stream error: {err}")
            return


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
                # Streaming with web tools: buffer & replay final text as SSE
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

            # Regular streaming with early transparent retries + recovery
            return StreamingResponse(
                _stream_with_recovery(
                    body,
                    req.model,
                    request_id=f"msg_{uuid.uuid4().hex[:8]}",
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: resolve web tools server-side
        response = await run_with_web_tools(_client, body)
        choice = response.choices[0]
        msg = choice.message

        content_blocks = []

        # 1. Extract native reasoning fields
        reasoning = getattr(msg, "reasoning_content", getattr(msg, "reasoning", None))
        if reasoning:
            content_blocks.append({"type": "thinking", "thinking": reasoning})

        # 2. Parse thinking tags from text content
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

                # Unalias unsafe parameter names (e.g. _nim_arg_type -> type)
                clean_args = unalias_tool_call_arguments(args)

                tc_id = getattr(tc, "id", None)
                if not tc_id:
                    tc_id = f"call_nim_{uuid.uuid4().hex[:10]}"

                content_blocks.append({
                    "type": "tool_use",
                    "id": str(tc_id),
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
            },
        }

    except Exception as err:
        logger.error(f"Error handling /v1/messages: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err


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
            # Streaming Responses: passthrough for now
            res = await _client.execute_chat(body)
            return res

        # Non-streaming: resolve local web tools
        body["stream"] = False
        oai_response = await run_with_web_tools(_client, body)
        if is_responses:
            from nim_proxy.core.openai import openai_response_to_responses
            return openai_response_to_responses(oai_response, model_name)
        return oai_response
    except Exception as err:
        logger.error(f"Error handling OpenAI chat completions: {err}")
        raise HTTPException(status_code=500, detail=str(err)) from err