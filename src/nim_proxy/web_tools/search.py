"""Multi-provider Web Search Implementation (DuckDuckGo, Tavily, Doubao Search)."""

import re
import urllib.parse
import httpx
from loguru import logger
from nim_proxy.config.settings import settings


async def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Execute DuckDuckGo search + Wikipedia fallback."""
    clean_query = query.strip()
    encoded_query = urllib.parse.quote(clean_query)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    }

    results = []

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            ddg_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            resp = await client.get(ddg_url, headers=headers)
            
            if resp.status_code == 200:
                html = resp.text
                matches = re.findall(
                    r'<a class="result__url" href="([^"]+)">\s*(.*?)\s*</a>[\s\S]*?<a class="result__snippet[^"]*"[^>]*>\s*(.*?)\s*</a>',
                    html
                )
                for link, title, snippet in matches[:max_results]:
                    clean_title = re.sub(r"<[^>]+>", "", title).strip()
                    clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                    if "duckduckgo.com/l/?" in link:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        if "uddg" in parsed:
                            link = parsed["uddg"][0]
                    if clean_title and clean_snippet:
                        results.append(f"### {clean_title}\n- **URL**: {link}\n- **Snippet**: {clean_snippet}")

            if not results:
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json"
                w_resp = await client.get(wiki_url, headers=headers)
                if w_resp.status_code == 200:
                    wiki_items = w_resp.json().get("query", {}).get("search", [])
                    for item in wiki_items[:max_results]:
                        t = item.get("title")
                        s = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
                        u = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(t)}"
                        if t and s:
                            results.append(f"### {t}\n- **URL**: {u}\n- **Snippet**: {s}")

    except Exception as err:
        logger.warning(f"DuckDuckGo search error for '{query}': {err}")

    if not results:
        return f"No search results found for query: '{query}'."

    return f"Search Results (DuckDuckGo) for '{query}':\n\n" + "\n\n".join(results)


async def search_tavily(query: str, max_results: int = 5, api_key: str = "") -> str:
    """Execute search using Tavily AI Search API."""
    if not api_key:
        logger.warning("Tavily API key unconfigured, falling back to DuckDuckGo.")
        return await search_duckduckgo(query, max_results)

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": True,
    }

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                answer = data.get("answer")
                if answer:
                    results.append(f"**Tavily Direct Answer**: {answer}\n")

                for item in data.get("results", [])[:max_results]:
                    title = item.get("title", "")
                    link = item.get("url", "")
                    content = item.get("content", "")
                    results.append(f"### {title}\n- **URL**: {link}\n- **Snippet**: {content}")

                if results:
                    return f"Search Results (Tavily) for '{query}':\n\n" + "\n\n".join(results)
            else:
                logger.warning(f"Tavily API error HTTP {resp.status_code}: {resp.text}")
    except Exception as err:
        logger.warning(f"Tavily search exception: {err}")

    return await search_duckduckgo(query, max_results)


async def search_doubao(query: str, max_results: int = 5, api_key: str = "") -> str:
    """Execute search using Volcengine / Doubao Search API."""
    if not api_key:
        logger.warning("Doubao Search API key unconfigured, falling back to DuckDuckGo.")
        return await search_duckduckgo(query, max_results)

    url = "https://ark.cn-beijing.volces.com/api/v3/bots/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "doubao-seed-search",
        "messages": [
            {"role": "user", "content": f"请对网络搜索关键词：'{query}' 检索并列出最相关的 {max_results} 条参考资料，给出标题、链接和简短摘要。"}
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return f"Search Results (豆包搜索) for '{query}':\n\n{content}"
            else:
                logger.warning(f"Doubao Search API HTTP {resp.status_code}: {resp.text}")
    except Exception as err:
        logger.warning(f"Doubao search exception: {err}")

    return await search_duckduckgo(query, max_results)


async def execute_web_search(query: str, max_results: int = 5) -> str:
    """Multi-provider dispatcher for web search."""
    provider = settings.web_search_provider.lower().strip()

    if provider == "tavily":
        return await search_tavily(query, max_results, settings.tavily_api_key)
    elif provider in ("doubao", "volcengine"):
        return await search_doubao(query, max_results, settings.doubao_search_api_key)
    else:
        return await search_duckduckgo(query, max_results)
