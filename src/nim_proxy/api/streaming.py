"""Managed SSE streaming response with proper error-frame emission and stream close.

Mirrors free-claude-code's ManagedStreamingResponse + _PrefetchedStream pattern:
- Wraps the body in a fault-tolerant generator that emits a terminal SSE error frame
  when the upstream stream fails, so Claude Code sees a clean protocol-level termination
  instead of a broken TCP stream.
- On client disconnect (cancellation), ensures the upstream stream is properly closed.
"""

import asyncio
import sys
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from nim_proxy.fcc_port.core.anthropic.streaming import (
    ANTHROPIC_SSE_RESPONSE_HEADERS,
    anthropic_terminal_error_frame,
    anthropic_terminal_failure_frame,
)
from nim_proxy.fcc_port.core.failures import find_execution_failure
from nim_proxy.fcc_port.core.trace import close_stream_input, trace_event


class EmptyStreamError(RuntimeError):
    """Raised when a public stream ends before emitting any protocol chunk."""


class ManagedStreamingResponse(StreamingResponse):
    """StreamingResponse that guarantees correct terminal SSE frames on failure.

    When the upstream provider stream fails mid-response, the default StreamingResponse
    simply stops sending bytes — Claude Code sees a broken SSE stream and doesn't know
    whether to retry. This class intercepts stream failures and emits an Anthropic
    terminal error frame before closing.

    Usage::

        async def provider_stream() -> AsyncIterator[str]:
            ...

        response = ManagedStreamingResponse(
            provider_stream(),
            error_payload_fn=lambda exc: anthropic_error_payload(...),
            request_id=request_id,
        )
    """

    def __init__(
        self,
        stream: AsyncIterator[str],
        *,
        request_id: str,
        media_type: str = "text/event-stream",
    ) -> None:
        wrapped = _FaultTolerantBody(stream, request_id=request_id)
        super().__init__(
            wrapped,
            media_type=media_type,
            headers=dict(ANTHROPIC_SSE_RESPONSE_HEADERS),
        )
        self._wrapped_body = wrapped


class _FaultTolerantBody:
    """Async iterable that emits a terminal error event on upstream failure.

    On StreamingResponse, this generator survives exceptions and re-emits
    them as inline SSE error frames that standard Anthropic clients can parse.
    """

    def __init__(self, stream: AsyncIterator[str], *, request_id: str) -> None:
        self._stream = stream
        self._request_id = request_id
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[str]:
        """Iterate the provider stream, emitting a terminal SSE error on failure."""
        try:
            async for chunk in self._stream:
                yield chunk
        except asyncio.CancelledError:
            # Client disconnected — yield a terminal error frame so future
            # clients reading error logs see it, then let cancellation propagate.
            try:
                yield _cancelled_frame(self._request_id)
            finally:
                await self._close(preserved_error=None)
            raise
        except BaseExceptionGroup as exc:
            await self._close(preserved_error=exc)
            yield _terminal_exception_frame(exc, request_id=self._request_id)
        except Exception as exc:
            await self._close(preserved_error=exc)
            yield _terminal_exception_frame(exc, request_id=self._request_id)
        finally:
            # Normal completion — close the upstream stream cleanly
            if not self._closed:
                await self._close(preserved_error=sys.exception())

    async def _close(
        self, *, preserved_error: BaseException | None
    ) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = (
            None if preserved_error is None
            else preserved_error
        )
        await close_stream_input(
            self._stream,
            owner="ManagedStreamingResponse",
            source="api",
            preserved_error=error,
        )


def _cancelled_frame(request_id: str) -> str:
    return anthropic_terminal_error_frame(
        "Request cancelled by client.", error_type="api_error"
    )


def _terminal_exception_frame(
    exc: BaseException, *, request_id: str
) -> str:
    """Build the best terminal SSE error event for the given exception."""
    failure = find_execution_failure(exc)
    if failure is not None:
        trace_event(
            stage="egress",
            event="nim_proxy.api.response.terminal_failure",
            source="api",
            request_id=request_id,
            failure_kind=failure.kind.value,
            status_code=failure.status_code,
        )
        return anthropic_terminal_failure_frame(failure)
    trace_event(
        stage="egress",
        event="nim_proxy.api.response.terminal_error",
        source="api",
        request_id=request_id,
        exc_type=type(exc).__name__,
    )
    return anthropic_terminal_error_frame(str(exc))


async def pre_fetch_stream(
    stream: AsyncIterator[str],
    *,
    request_id: str,
) -> tuple[str, AsyncIterator[str]]:
    """Pre-fetch the first SSE chunk to catch pre-load failures.

    Errors during the first iteration (no bytes yet sent) return as JSON
    to the client, not SSE events. That's what routes.py does.

    Returns (first_chunk, tail_stream).
    """
    try:
        first_chunk = await anext(stream)
    except StopAsyncIteration:
        await close_stream_input(
            stream,
            owner="prefetch_stream",
            source="api",
            preserved_error=EmptyStreamError("Stream ended before emitting a response."),
        )
        raise EmptyStreamError("Stream ended before emitting a response.")
    return first_chunk, _PrefetchedIterator(first_chunk, stream)


class _PrefetchedIterator(AsyncIterator[str]):
    """Replay one prefetched chunk while retaining ownership of the tail."""

    def __init__(
        self,
        first_chunk: str,
        body: AsyncIterator[str],
    ) -> None:
        self._first: str | None = first_chunk
        self._body = body

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if self._first is not None:
            chunk = self._first
            self._first = None
            return chunk
        return await anext(self._body)