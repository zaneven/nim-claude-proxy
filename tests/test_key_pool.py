"""Tests for KeyPoolManager (Round-Robin and Cooldown Isolation)."""

import asyncio
import pytest
from nim_proxy.provider.key_pool import KeyPoolManager


@pytest.mark.asyncio
async def test_round_robin_dispatching():
    pool = KeyPoolManager(api_keys=["key-1", "key-2", "key-3"], cooldown_seconds=10.0)

    # 验证平滑轮询顺序: 1 -> 2 -> 3 -> 1
    k1 = await pool.get_next_key()
    k2 = await pool.get_next_key()
    k3 = await pool.get_next_key()
    k4 = await pool.get_next_key()

    assert k1 == "key-1"
    assert k2 == "key-2"
    assert k3 == "key-3"
    assert k4 == "key-1"


@pytest.mark.asyncio
async def test_429_cooldown_isolation():
    pool = KeyPoolManager(api_keys=["key-1", "key-2"], cooldown_seconds=60.0)

    # 报告 key-1 遇到 429 限流
    await pool.report_status("key-1", status_code=429)

    # 轮询获取 key，此时应跳过处于 60s 冷却中的 key-1，连续输出 key-2
    assert await pool.get_next_key() == "key-2"
    assert await pool.get_next_key() == "key-2"

    stats = pool.get_stats()
    assert stats["active"] == 1
    assert stats["cooldown"] == 1


@pytest.mark.asyncio
async def test_401_permanent_disabling():
    pool = KeyPoolManager(api_keys=["key-1", "key-2"], cooldown_seconds=60.0)

    # 报告 key-1 遇到 401 鉴权失效
    await pool.report_status("key-1", status_code=401)

    # key-1 应被永久禁用
    assert await pool.get_next_key() == "key-2"
    assert await pool.get_next_key() == "key-2"

    stats = pool.get_stats()
    assert stats["disabled"] == 1
    assert stats["active"] == 1
