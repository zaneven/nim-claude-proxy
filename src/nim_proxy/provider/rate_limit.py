"""Shared strict sliding-window rate limiting primitives."""

import asyncio
import time
from collections import deque
from collections.abc import Callable


class SmoothRateLimiter:
    """Smooth rate limiter that paces requests evenly.

    Guarantees: exactly 1 request per `rate_window / rate_limit` interval.
    This prevents burst requests that could trigger upstream 429 rate limits.

    Implemented as an async context manager so call sites can do::

        async with limiter:
            ...
    """

    def __init__(self, rate_limit: int, rate_window: float = 60.0) -> None:
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")

        self._interval = float(rate_window) / float(rate_limit)
        self._next_allowed_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._acquire(None)

    async def acquire_if(self, allowed: Callable[[], bool]) -> bool:
        """Record an acquisition only if ``allowed`` still holds at admission."""
        return await self._acquire(allowed)

    async def _acquire(self, allowed: Callable[[], bool] | None) -> bool:
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed_time:
                    if allowed is not None and not allowed():
                        return False
                    self._next_allowed_time = now + self._interval
                    return True
                
                wait_time = self._next_allowed_time - now

            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def __aenter__(self) -> "SmoothRateLimiter":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False
