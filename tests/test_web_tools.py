"""Tests for Web Tools (Local web_search & web_fetch & multi-providers)."""

import pytest
from nim_proxy.config.settings import settings
from nim_proxy.web_tools.fetch import html_to_clean_text
from nim_proxy.web_tools.executor import is_web_tool, handle_local_web_tool
from nim_proxy.web_tools.search import search_tavily, search_doubao


def test_is_web_tool():
    assert is_web_tool("web_search") is True
    assert is_web_tool("web_fetch") is True
    assert is_web_tool("bash") is False


def test_html_to_clean_text():
    html = "<html><body><h1>Title</h1><script>alert(1)</script><p>Hello World!</p></body></html>"
    clean = html_to_clean_text(html)
    assert "Title" in clean
    assert "Hello World!" in clean
    assert "alert" not in clean


@pytest.mark.asyncio
async def test_handle_local_web_fetch():
    res = await handle_local_web_tool("web_fetch", {"url": "https://example.com"})
    assert res is not None
    assert "Example Domain" in res or "Content of" in res


@pytest.mark.asyncio
async def test_tavily_fallback_without_key():
    # 当 Tavily Key 为空时，优雅自动回退至 DuckDuckGo
    res = await search_tavily("Python 3.14", api_key="")
    assert res is not None
    assert "Python" in res or "DuckDuckGo" in res or "Search Results" in res


@pytest.mark.asyncio
async def test_doubao_fallback_without_key():
    # 当豆包 Key 为空时，优雅自动回退至 DuckDuckGo
    res = await search_doubao("Telegram Bot", api_key="")
    assert res is not None
    assert "Telegram" in res or "DuckDuckGo" in res or "Search Results" in res
