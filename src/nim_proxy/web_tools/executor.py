"""Executor handling intercepting and executing Anthropic local web tools."""

from nim_proxy.config.settings import settings
from nim_proxy.web_tools.fetch import execute_web_fetch
from nim_proxy.web_tools.search import execute_web_search


def is_web_tool(tool_name: str) -> bool:
    name = tool_name.lower()
    return name in ("web_search", "web_fetch", "search", "fetch")


async def handle_local_web_tool(tool_name: str, tool_input: dict) -> str | None:
    """Intercept and run local web_search or web_fetch tool if enabled."""
    name = tool_name.lower()

    if name in ("web_search", "search") and settings.enable_web_search:
        query = tool_input.get("query") or tool_input.get("q") or str(tool_input)
        return await execute_web_search(query, max_results=settings.duckduckgo_max_results)

    if name in ("web_fetch", "fetch") and settings.enable_web_fetch:
        url = tool_input.get("url") or tool_input.get("link") or str(tool_input)
        return await execute_web_fetch(url)

    return None
