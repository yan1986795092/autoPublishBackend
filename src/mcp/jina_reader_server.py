"""Jina Reader MCP Server — 网页 Markdown 化抓取

基于 Model Context Protocol (MCP) SDK，
使用 Jina Reader API 将任意网页转换为干净的 Markdown。

Jina Reader 特点：
- 完全免费，无需 API Key
- 自动提取正文，去除广告/导航/脚注
- 返回 Markdown 格式，AI 友好
- API: https://r.jina.ai/https://example.com
  免费层级：无限制使用，无 Key 需求

支持两种抓取模式：
1. Jina Reader（默认，完全免费）
2. 直接 HTTP 抓取（兜底降级）
"""

from __future__ import annotations

import logging
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="jina-reader",
    instructions="Jina Reader — 将任意网页转换为干净的 Markdown。完全免费，无需 API Key。适用于从搜索结果中获取详细内容、文档页面抓取等场景。",
    dependencies=["httpx>=0.25.0"],
)


@mcp.tool(
    name="read_url",
    description="使用 Jina Reader 将网页转换为干净的 Markdown 格式。完全免费，无需 API Key。输入 URL，返回去除了广告/导航的正文内容，AI 友好。适用于获取文档、教程、新闻等长文内容。",
)
async def read_url(
    url: str,
    max_length: int = 8000,
    return_format: str = "markdown",
) -> str:
    """抓取 URL 并转换为 Markdown"""
    try:
        import httpx
        import urllib.parse

        # Jina Reader API
        jina_url = f"https://r.jina.ai/{urllib.parse.quote(url, safe='')}"

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(
                jina_url,
                headers={
                    "Accept": "text/plain",
                    "X-Return-Format": "markdown",
                    "User-Agent": "Mozilla/5.0 (compatible; AutoPublish/1.0; +https://github.com/auto-publish)",
                },
            )

        if resp.status_code == 200:
            content = resp.text
            if max_length and len(content) > max_length:
                content = content[:max_length] + f"\n\n_(内容已被截断至 {max_length} 字)_"

            # Jina 返回格式：title + --- + markdown
            parts = content.split("\n---\n", 1)
            if len(parts) == 2:
                title, body = parts
                return f"**{title.strip()}**\n\n🔗 来源：{url}\n\n---\n\n{body.strip()}"
            else:
                return f"🔗 来源：{url}\n\n---\n\n{content.strip()}"

        logger.warning(f"Jina Reader failed for {url} (HTTP {resp.status_code})")
        return await _direct_fetch(url, max_length)

    except Exception as e:
        logger.warning(f"Jina Reader error for {url}: {e}")
        return await _direct_fetch(url, max_length)


@mcp.tool(
    name="read_urls_batch",
    description="批量抓取多个 URL（使用 Jina Reader）。输入 URL 列表，依次抓取并返回 Markdown 格式内容。适用于批量获取文章详情、研究资料收集等场景。",
)
async def read_urls_batch(
    urls: list[str],
    max_length_per_url: int = 4000,
) -> str:
    """批量抓取多个 URL"""
    if not urls:
        return "未提供 URL 列表。"

    results = []
    for i, url in enumerate(urls, 1):
        result = await read_url(url, max_length=max_length_per_url, return_format="markdown")
        results.append(f"## [{i}] {url}\n\n{result}\n\n{'=' * 40}")

    return f"**批量抓取结果（共 {len(urls)} 个 URL）**\n\n" + "\n\n".join(results)


async def _direct_fetch(url: str, max_length: int) -> str:
    """直接 HTTP 抓取（降级兜底）"""
    try:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
            )

        if resp.status_code != 200:
            return f"网页抓取失败（HTTP {resp.status_code}）：{url}"

        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除干扰元素
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "form", "button"]):
            tag.decompose()

        # 提取标题
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 提取 meta 描述
        meta_desc = ""
        if soup.find("meta", attrs={"name": "description"}):
            meta_desc = soup.find("meta", attrs={"name": "description"}).get("content", "")

        # 提取正文
        article = soup.find("article") or soup.find("main") or soup
        paragraphs = article.find_all("p")

        text_parts = []
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 15:
                text_parts.append(text)

        content = "\n\n".join(text_parts)[:max_length]

        result = f"**{title or url}**"
        if meta_desc:
            result += f"\n\n📝 {meta_desc[:200]}"
        result += f"\n\n🔗 来源：{url}\n\n---\n\n{content}"

        if len(content) >= max_length:
            result += f"\n\n_(内容已被截断至 {max_length} 字)_"

        return result

    except Exception as e:
        return f"网页抓取失败：{e}\n\nURL: {url}"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8767)
    else:
        mcp.run(transport="stdio")
