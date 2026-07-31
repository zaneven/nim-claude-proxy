"""Tests for NimClient failover and retry mechanisms."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch
import openai

from nim_proxy.provider.client import NimClient
from nim_proxy.provider.key_pool import KeyPoolManager


@pytest.mark.asyncio
async def test_client_execute_chat_retries_on_429(monkeypatch):
    """Test that client retries multiple times even if only 1 key is configured,
    and relies on KeyPoolManager's asyncio.sleep for cooldown.
    """
    key_pool = KeyPoolManager(["key-1"], cooldown_seconds=0.1)  # small cooldown for fast test
    client = NimClient(key_pool)
    
    call_count = 0
    
    async def _mock_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 4:
            raise openai.RateLimitError(
                "Rate limit exceeded",
                response=MagicMock(),
                body=None
            )
        return "success"
    
    monkeypatch.setattr(client, "_call_with_rejection_retry", _mock_call)
    
    # 尽管只有 1 个 key，重试上限被设计为 >= 5
    result = await client.execute_chat({"messages": []})
    
    assert result == "success"
    assert call_count == 4
    
    stats = key_pool.get_stats()
    assert stats["total"] == 1

