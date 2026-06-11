"""Tavily Search MCP Server — AI 优化的新闻与博客搜索

基于 Model Context Protocol (MCP) SDK，
提供 Tavily AI Search API 的高质量新闻/博客搜索。

Tavily Search 特点：
- 专为 AI 应用设计，返回结构化、去重复的结果
- 覆盖新闻、博客、论坛等多源内容
- 支持指定领域/日期范围过滤
- API Key 获取：https://app.tavily.com
  免费层级：每月 1,000 次查询

如未配置 TAVILY_API_KEY，自动降级到 DuckDuckGo 新闻搜索。
"""

from __future__ import annotations

import os
import logging

# 抑制 MCP SDK 内部 INFO 日志
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="tavily-search",
    instructions="Tavily Search — AI 优化的新闻与博客搜索。专为 AI 应用设计，返回高质量、去重复、上下文丰富的搜索结果。",
    dependencies=["httpx>=0.25.0"],
)


@mcp.tool(
    name="tavily_search",
    description="使用 Tavily AI Search 进行深度新闻/博客搜索。专为 AI 应用设计，返回结构化、去重复的高质量结果。适用于查找最新资讯、行业动态、深度分析等。",
)
async def tavily_search(
    query: str,
    max_results: int = 5,
    search_type: str = "news",
    days: int = 7,
) -> str:
    api_key = os.getenv("TAVILY_API_KEY")

    if api_key:
        try:
            return await _tavily_api_search(query, api_key, max_results, search_type, days)
        except Exception as e:
            logger.warning(f"Tavily API failed: {e}, falling back to DuckDuckGo")
    else:
        logger.info("TAVILY_API_KEY not set, using DuckDuckGo fallback")

    return await _duckduckgo_fallback(query, max_results)


async def _tavily_api_search(
    query: str,
    api_key: str,
    max_results: int,
    search_type: str,
    days: int,
) -> str:
    import httpx
    import json

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_type": search_type,
                "days": days,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
            },
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code == 401:
        return "Tavily API Key 无效，请检查 TAVILY_API_KEY 配置。降级到 DuckDuckGo：\n" + await _duckduckgo_fallback(query, max_results)

    if resp.status_code != 200:
        return f"Tavily Search 请求失败（HTTP {resp.status_code}）。降级到 DuckDuckGo：\n" + await _duckduckgo_fallback(query, max_results)

    data = resp.json()

    # AI 生成的答案摘要
    answer = data.get("answer", "")
    results = data.get("results", [])

    if not results:
        return await _duckduckgo_fallback(query, max_results)

    lines = [f"**「{query}」Tavily 搜索结果（共 {len(results)} 条）：**"]

    if answer:
        lines.append(f"\n📌 **AI 摘要：**\n{answer}\n")

    lines.append("\n**详细结果：**\n")
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("content", "")[:250]
        score = r.get("score", 0)
        published_date = r.get("published_date", "")

        lines.append(f"**{i}. {title}**")
        if score:
            lines.append(f"   相关度：{'★' * int(score * 5)}{'☆' * (5 - int(score * 5))} ({score:.2f})")
        if snippet:
            lines.append(f"   {snippet}...")
        lines.append(f"   🔗 {url}")
        if published_date:
            lines.append(f"   📅 {published_date}")
        lines.append("")

    lines.append("_内容来自 Tavily Search（专为 AI 优化）_")
    return "\n".join(lines)


@mcp.tool(
    name="tavily_news",
    description="使用 Tavily AI Search 搜索最新新闻。专注新闻来源，返回时间排序的最新资讯。适用于查找行业动态、突发事件、热点新闻等。",
)
async def tavily_news(
    query: str,
    max_results: int = 8,
    days: int = 3,
) -> str:
    return await tavily_search(query, max_results=max_results, search_type="news", days=days)


@mcp.tool(
    name="tavily_deep_search",
    description="使用 Tavily 进行深度搜索，返回最全面的结果集。包含 AI 生成的摘要答案，覆盖所有相关来源。适用于需要全面了解某个主题的场景。",
)
async def tavily_deep_search(
    query: str,
    max_results: int = 10,
) -> str:
    return await tavily_search(query, max_results=max_results, search_type="full", days=30)


async def _duckduckgo_fallback(query: str, max_results: int) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS(timeout=20) as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

        if not results:
            return f"搜索「{query}」未找到新闻。"

        lines = [f"**「{query}」最新新闻（共 {len(results)} 条）：**\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            body = r.get("body", "")[:200]
            source = r.get("source", "")
            date = r.get("date", "")

            lines.append(f"**{i}. {title}**")
            if source or date:
                lines.append(f"   来源：{source}  |  时间：{date}")
            if body:
                lines.append(f"   {body}...")
            lines.append(f"   🔗 {url}\n")

        lines.append("_内容来自 DuckDuckGo（免费无需 API Key）_")
        return "\n".join(lines)
    except Exception as e:
        return f"新闻搜索暂时不可用：{e}。请配置 TAVILY_API_KEY。"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8768)
    else:
        mcp.run(transport="stdio")
