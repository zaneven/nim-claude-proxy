"""Tests for the StrictSlidingWindowLimiter."""

import asyncio
import time
import pytest
from nim_proxy.provider.rate_limit import StrictSlidingWindowLimiter


@pytest.mark.asyncio
async def test_strict_sliding_window_limiter():
    # Setup limiter: max 2 requests per 0.5 seconds
    limiter = StrictSlidingWindowLimiter(rate_limit=2, rate_window=0.5)

    start_time = time.monotonic()
    
    # First two requests should pass immediately
    await limiter.acquire()
    await limiter.acquire()
    
    elapsed = time.monotonic() - start_time
    assert elapsed < 0.1, "First two acquisitions should not block"

    # Third request should block until the window clears (approx 0.5s)
    await limiter.acquire()
    elapsed2 = time.monotonic() - start_time
    assert elapsed2 >= 0.5, "Third acquisition should be delayed by the window"


@pytest.mark.asyncio
async def test_acquire_if():
    limiter = StrictSlidingWindowLimiter(rate_limit=1, rate_window=0.5)
    
    flag = False
    
    # Allowed condition is False, should not acquire and should return False
    acquired = await limiter.acquire_if(lambda: flag)
    assert not acquired
    
    # Since it was not acquired, quota is not consumed. 
    # Next request with True condition should pass immediately.
    start = time.monotonic()
    acquired2 = await limiter.acquire_if(lambda: True)
    assert acquired2
    elapsed = time.monotonic() - start
    assert elapsed < 0.1
    
    # Next request should be blocked by quota
    await limiter.acquire()
    elapsed2 = time.monotonic() - start
    assert elapsed2 >= 0.5


def test_invalid_params():
    with pytest.raises(ValueError):
        StrictSlidingWindowLimiter(0, 1.0)
    with pytest.raises(ValueError):
        StrictSlidingWindowLimiter(1, -1.0)
