"""Tests for the SmoothRateLimiter."""

import asyncio
import time
import pytest
from nim_proxy.provider.rate_limit import SmoothRateLimiter


@pytest.mark.asyncio
async def test_smooth_rate_limiter():
    # Setup limiter: max 4 requests per 1.0 seconds
    # Interval should be 0.25 seconds
    limiter = SmoothRateLimiter(rate_limit=4, rate_window=1.0)

    start_time = time.monotonic()
    
    # First request should pass immediately
    await limiter.acquire()
    elapsed = time.monotonic() - start_time
    assert elapsed < 0.1, "First acquisition should not block"

    # Second request should block for 0.25s
    await limiter.acquire()
    elapsed2 = time.monotonic() - start_time
    assert elapsed2 >= 0.2, "Second acquisition should be paced"


@pytest.mark.asyncio
async def test_acquire_if():
    # Interval is 0.5 seconds
    limiter = SmoothRateLimiter(rate_limit=2, rate_window=1.0)
    
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
    assert elapsed2 >= 0.4


def test_invalid_params():
    with pytest.raises(ValueError):
        SmoothRateLimiter(0, 1.0)
    with pytest.raises(ValueError):
        SmoothRateLimiter(1, -1.0)
