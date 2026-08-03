"""NVIDIA NIM Dedicated Client featuring Multi-Key failover and stream normalization."""

import json
from collections.abc import AsyncIterator
from typing import Any

import openai
from loguru import logger

from nim_proxy.config.settings import settings
from nim_proxy.provider.key_pool import KeyPoolManager
from nim_proxy.provider.native_tool import normalize_nim_native_tool_stream
from nim_proxy.provider.retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
)
from nim_proxy.provider.schema import sanitize_nim_tools


from nim_proxy.provider.rate_limit import SmoothRateLimiter


class NimClient:
    """NVIDIA NIM Dedicated Execution Client."""

    def __init__(self, key_pool: KeyPoolManager) -> None:
        self.key_pool = key_pool
        self._limiter: SmoothRateLimiter | None = None
        self._limiter_rpm = 0

    async def execute_non_streaming(self, body: dict[str, Any]) -> Any:
        """Execute a non-streaming chat request, used for recovery continuation.

        Unlike execute_chat, this always forces stream=False and returns
        the complete response object.
        """
        body = {**body, "stream": False}
        return await self.execute_chat(body)

    async def collect_continuation_text(
        self, body: dict[str, Any], max_attempts: int = 2
    ) -> tuple[str, str]:
        """Collect text + reasoning from a continuation/recovery request.

        Returns (text, reasoning) tuple extracted from the non-streaming response.
        Used by stream recovery to complete truncated text or tool calls.
        """
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = await self.execute_non_streaming(body)
                choice = response.choices[0]
                msg = choice.message
                text = getattr(msg, "content", None) or ""
                reasoning = getattr(
                    msg, "reasoning_content",
                    getattr(msg, "reasoning", None),
                ) or ""
                return text, reasoning
            except Exception as err:
                last_error = err
                logger.warning(
                    f"Recovery continuation attempt {attempt + 1}/{max_attempts} failed: {err}"
                )
        raise last_error  # type: ignore[misc]

    async def _check_rate_limit(self) -> None:
        if settings.rate_limit_rpm != self._limiter_rpm:
            self._limiter_rpm = settings.rate_limit_rpm
            if self._limiter_rpm > 0:
                self._limiter = SmoothRateLimiter(self._limiter_rpm, 60.0)
                logger.info(f"Global rate limit configured: {self._limiter_rpm} RPM")
            else:
                self._limiter = None
                logger.info("Global rate limit disabled")

        if self._limiter:
            await self._limiter.acquire()

    def _get_async_client(self, api_key: str) -> openai.AsyncOpenAI:
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": settings.nvidia_nim_base_url,
        }
        if settings.nvidia_nim_proxy:
            import httpx
            kwargs["http_client"] = httpx.AsyncClient(
                proxy=settings.nvidia_nim_proxy,
                timeout=httpx.Timeout(
                    settings.http_timeout,
                    connect=10.0,
                    read=settings.http_timeout,
                    write=10.0,
                ),
            )
        return openai.AsyncOpenAI(**kwargs)

    async def execute_chat(self, body: dict[str, Any]) -> Any:
        """Execute non-streaming or streaming Chat Completion with Multi-Key Failover."""
        await self._check_rate_limit()

        # 清洗 tools 定义
        if "tools" in body and isinstance(body["tools"], list):
            sanitized_tools, _ = sanitize_nim_tools(body["tools"])
            body["tools"] = sanitized_tools

        stats = self.key_pool.get_stats()
        # 至少重试 5 次（对齐原项目的 UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS）
        # 如果 Key 特别多（比如 10 个），则保证能尝试完所有 Key 并至少循环两轮
        max_attempts = max(5, stats["total"] * 2)

        last_exception = None

        for attempt in range(max_attempts):
            api_key = await self.key_pool.get_next_key()
            client = self._get_async_client(api_key)

            try:
                # 尝试执行 API 调用
                response = await self._call_with_rejection_retry(client, body)
                await self.key_pool.report_success(api_key)

                if body.get("stream", False):
                    # 对流式响应进行原生 Tool 标记归一化处理
                    if settings.enable_native_tool_cleaning:
                        return normalize_nim_native_tool_stream(response)
                    return response
                return response

            except openai.BadRequestError as err:
                # 400 Bad Request 不属于 Key 故障，不进行 Key 切流
                await self.key_pool.report_success(api_key)
                raise err

            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.AuthenticationError) as err:
                status_code = getattr(err, "status_code", None)
                if isinstance(err, openai.RateLimitError):
                    status_code = 429
                elif isinstance(err, openai.AuthenticationError):
                    status_code = 401

                await self.key_pool.report_status(api_key, status_code=status_code, error=err)
                last_exception = err
                logger.warning(
                    f"Attempt {attempt + 1}/{max_attempts} failed using NIM Key [{api_key[:8]}...]. "
                    f"Retrying with next key... Error: {err}"
                )

        # 若所有可尝试 Key 均失败
        if last_exception:
            raise last_exception
        raise RuntimeError("Failed to execute request with all available NVIDIA NIM API keys.")

    async def _call_with_rejection_retry(self, client: openai.AsyncOpenAI, body: dict[str, Any]) -> Any:
        """Call NIM endpoint, retrying with degraded body if NIM rejects specific parameters."""
        current_body = body
        try:
            return await client.chat.completions.create(**current_body)
        except (openai.BadRequestError, openai.InternalServerError) as err:
            err_text = str(err).lower()
            
            # 1. 尝试降级 reasoning_budget
            if "reasoning_budget" in err_text or "thinking_token_budget" in err_text:
                degraded = clone_body_without_reasoning_budget(current_body)
                if degraded:
                    logger.warning("NIM rejected reasoning_budget; retrying with degraded body...")
                    return await client.chat.completions.create(**degraded)

            # 2. 尝试降级 chat_template
            if "chat_template" in err_text:
                degraded = clone_body_without_chat_template(current_body)
                if degraded:
                    logger.warning("NIM rejected chat_template; retrying with degraded body...")
                    return await client.chat.completions.create(**degraded)

            raise err
