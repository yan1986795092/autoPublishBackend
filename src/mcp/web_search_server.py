"""Open-WebSearch MCP Server — 实时网络搜索服务

基于 Model Context Protocol (MCP) SDK，
提供网页搜索、新闻搜索、内容抓取等工具。
支持多引擎：DuckDuckGo / Bing (需 API Key) / SerpAPI
"""

# Note: DO NOT use `from __future__ import annotations` here.
# The MCP SDK's `issubclass(param.annotation, Context)` inspection requires
# eagerly-evaluated type annotations, not strings.
import asyncio
import logging
import os
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# 初始化 FastMCP Server
mcp = FastMCP(
    name="open-websearch",
    instructions="实时网络搜索工具，支持网页搜索、新闻搜索和页面内容抓取",
    dependencies=["duckduckgo_search>=7.0"],
)


# ──────────────────────────────────────────
# 搜索工具
# ──────────────────────────────────────────


@mcp.tool(
    name="web_search",
    description="搜索互联网上的最新信息。输入搜索关键词，返回相关网页结果。适用于查找新闻、文章、教程等资料。",
)
async def web_search(
    query: str,
    max_results: int = 5,
    source: str = "auto",
) -> str:
    """网页搜索 — 支持多引擎自动降级"""
    engines_tried = []

    # ── 引擎 1: DuckDuckGo (免费，无需 API Key) ──
    if source in ("auto", "duckduckgo"):
        engines_tried.append("DuckDuckGo")
        try:
            from duckduckgo_search import DDGS

            with DDGS(timeout=20) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if results:
                return _format_web_results(results, query)
            logger.warning("DuckDuckGo returned empty results, trying next engine")
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

    # ── 引擎 2: Bing Web Search (需 BING_API_KEY) ──
    bing_key = os.getenv("BING_API_KEY")
    if bing_key and source in ("auto", "bing"):
        engines_tried.append("Bing")
        try:
            results = await _bing_search(query, bing_key, max_results)
            if results:
                return _format_web_results(results, query)
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")

    # ── 引擎 3: SerpAPI (需 SERPAPI_API_KEY) ──
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key and source in ("auto", "serpapi"):
        engines_tried.append("SerpAPI")
        try:
            results = await _serpapi_search(query, serpapi_key, max_results)
            if results:
                return _format_web_results(results, query)
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")

    return (
        f"搜索「{query}」失败。尝试过的引擎：{', '.join(engines_tried) or '无'}。"
        "请检查网络连接或配置 BING_API_KEY / SERPAPI_API_KEY 环境变量。"
    )


@mcp.tool(
    name="news_search",
    description="搜索最新新闻和热点话题。输入关键词，返回相关的最新新闻报道。适用于查找行业动态、热点事件等。",
)
async def news_search(
    query: str,
    max_results: int = 5,
    region: str = "cn-zh",
) -> str:
    """新闻搜索"""
    try:
        from duckduckgo_search import DDGS

        with DDGS(timeout=20) as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

        if not results:
            return f"未找到「{query}」相关的最新新闻。"

        lines = [f"**{query}** 相关新闻（共 {len(results)} 条）：\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            source = r.get("source", "")
            date = r.get("date", "")
            url = r.get("url", "")
            body = r.get("body", "")[:200]

            lines.append(f"**{i}. {title}**")
            if source:
                lines.append(f"   来源：{source}  |  时间：{date}")
            if body:
                lines.append(f"   {body}...")
            lines.append(f"   🔗 {url}\n")

        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return f"新闻搜索暂时不可用：{e}"


@mcp.tool(
    name="fetch_page",
    description="抓取指定网页的主要内容。输入 URL，返回去除了导航栏/广告后的干净文本内容。适用于从搜索结果中获取详细内容。",
)
async def fetch_page(
    url: str,
    max_length: int = 5000,
) -> str:
    """抓取网页正文内容"""
    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)

        if resp.status_code != 200:
            return f"网页抓取失败，HTTP {resp.status_code}"

        # 解析 HTML
        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除干扰元素
        for tag in soup.find_all(
            ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "form"]
        ):
            tag.decompose()

        # 提取标题
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 提取 meta 描述
        meta_desc = ""
        if soup.find("meta", attrs={"name": "description"}):
            meta_desc = soup.find("meta", attrs={"name": "description"}).get("content", "")

        # 提取正文（优先 article/main 标签）
        article = soup.find("article") or soup.find("main") or soup
        paragraphs = article.find_all("p")

        text_parts = []
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 20:  # 过滤掉太短的段落
                text_parts.append(text)

        content = "\n\n".join(text_parts)[:max_length]

        result = f"**{title or url}**\n\n"
        if meta_desc:
            result += f"📝 {meta_desc[:200]}\n\n"
        result += f"**正文内容：**\n\n{content}"

        if len(content) >= max_length:
            result += f"\n\n_(内容已被截断至 {max_length} 字)_"

        result += f"\n\n🔗 来源：{url}"
        return result

    except httpx.TimeoutException:
        return f"抓取超时：{url}"
    except Exception as e:
        logger.warning(f"Page fetch failed for {url}: {e}")
        return f"网页抓取失败：{e}"


# ──────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────

def _format_web_results(results: list[dict], query: str) -> str:
    """将搜索结果格式化为友好的 Markdown 文本"""
    lines = [f"**「{query}」搜索结果（共 {len(results)} 条）：**\n"]

    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")
        href = r.get("href", "")
        body = r.get("body", "")[:200]

        lines.append(f"**{i}. {title}**")
        if body:
            lines.append(f"   {body}...")
        if href:
            lines.append(f"   🔗 {href}")
        lines.append("")

    lines.append("_内容来自网络搜索，仅供参考_")
    return "\n".join(lines)


async def _bing_search(query: str, api_key: str, max_results: int) -> list[dict]:
    """Bing Web Search API"""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": max_results, "mkt": "zh-CN"},
            headers={"Ocp-Apim-Subscription-Key": api_key},
        )
        data = resp.json()
        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append({
                "title": item.get("name", ""),
                "href": item.get("url", ""),
                "body": item.get("snippet", ""),
            })
        return results


async def _serpapi_search(query: str, api_key: str, max_results: int) -> list[dict]:
    """SerpAPI (支持 Google/Bing/DuckDuckGo 等)"""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": api_key, "num": max_results},
        )
        data = resp.json()
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "href": item.get("link", ""),
                "body": item.get("snippet", ""),
            })
        return results


# ──────────────────────────────────────────
# 入口
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        # HTTP 模式（供 langchain-mcp-adapters streamable_http 使用）
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
    else:
        # stdio 模式（供 langchain-mcp-adapters 使用）
        mcp.run(transport="stdio")
