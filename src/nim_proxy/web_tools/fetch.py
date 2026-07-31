"""Local web_fetch implementation for HTML retrieving and text parsing."""

import re
import httpx
from loguru import logger

_MAX_BYTES = 100_000


def html_to_clean_text(html: str) -> str:
    """Extract clean readable text from HTML string."""
    # 移除 script 与 style
    html = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
    # 替换 <p>, <br>, <h1>.. 等换行
    html = re.sub(r"<(p|br|div|h1|h2|h3|h4|li)[^>]*>", "\n", html, flags=re.IGNORECASE)
    # 剥离所有 HTML 标签
    text = re.sub(r"<[^>]+>", "", html)
    # 合并多余空白
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


async def execute_web_fetch(url: str, max_bytes: int = _MAX_BYTES) -> str:
    """Fetch content from target URL and return clean text summary."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return f"Error fetching {url}: HTTP {resp.status_code}"
            
            content = resp.text[:max_bytes]
            clean_text = html_to_clean_text(content)
            return f"Content of {url}:\n\n{clean_text[:4000]}"
    except Exception as err:
        logger.warning(f"Failed to fetch {url}: {err}")
        return f"Error fetching {url}: {err}"
