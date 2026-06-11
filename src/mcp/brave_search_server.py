"""Brave Search MCP Server — 隐私优先的网页搜索

基于 Model Context Protocol (MCP) SDK，
提供基于 Brave Search API 的高质量网页搜索。

Brave Search 特点：
- 隐私优先，不追踪用户
- 结果质量高，适合 AI 场景
- API Key 获取：https://brave.com/search/api/
  免费层级：每月 2,000 次查询

如未配置 BRAVE_API_KEY，自动降级到 DuckDuckGo。
"""

from __future__ import annotations

import os
import logging

# 抑制 MCP SDK 内部 INFO 日志
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="brave-search",
    instructions="Brave Search — 隐私优先的高质量网页搜索。返回去除了追踪参数的干净 URL。",
    dependencies=["httpx>=0.25.0"],
)


@mcp.tool(
    name="brave_search",
    description="使用 Brave Search API 进行网页搜索。返回高质量、去隐私化的搜索结果，适用于查找最新资讯、教程和产品信息。如未配置 API Key 则降级到 DuckDuckGo。",
)
async def brave_search(
    query: str,
    max_results: int = 5,
    source: str = "auto",
) -> str:
    api_key = os.getenv("BRAVE_API_KEY")

    if api_key and source in ("auto", "brave"):
        try:
            return await _brave_api_search(query, api_key, max_results)
        except Exception as e:
            logger.warning(f"Brave Search API failed: {e}, falling back to DuckDuckGo")

    return await _duckduckgo_fallback(query, max_results)


async def _brave_api_search(query: str, api_key: str, max_results: int) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )

    if resp.status_code == 401:
        return "Brave API Key 无效，请检查 BRAVE_API_KEY 配置。降级到 DuckDuckGo：\n" + await _duckduckgo_fallback(query, max_results)

    if resp.status_code != 200:
        return f"Brave Search 请求失败（HTTP {resp.status_code}）。降级到 DuckDuckGo：\n" + await _duckduckgo_fallback(query, max_results)

    data = resp.json()
    web_results = data.get("web", {}).get("results", [])

    if not web_results:
        return await _duckduckgo_fallback(query, max_results)

    lines = [f"**「{query}」Brave 搜索结果（共 {len(web_results)} 条）：**\n"]
    for i, r in enumerate(web_results[:max_results], 1):
        title = r.get("title", "无标题")
        url = r.get("url", "")
        snippet = r.get("description", "")
        age = r.get("age", "")

        lines.append(f"**{i}. {title}**")
        if snippet:
            lines.append(f"   {snippet[:200]}")
        lines.append(f"   🔗 {url}")
        if age:
            lines.append(f"   📅 {age}")
        lines.append("")

    lines.append("_内容来自 Brave Search_")
    return "\n".join(lines)


async def _duckduckgo_fallback(query: str, max_results: int) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS(timeout=20) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"搜索「{query}」未找到结果。"

        lines = [f"**「{query}」DuckDuckGo 搜索结果（共 {len(results)} 条）：**\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            href = r.get("href", "")
            body = r.get("body", "")[:200]
            lines.append(f"**{i}. {title}**")
            if body:
                lines.append(f"   {body}...")
            lines.append(f"   🔗 {href}")
            lines.append("")

        lines.append("_内容来自 DuckDuckGo（免费无需 API Key）_")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索「{query}」暂时不可用：{e}。请配置 BRAVE_API_KEY 或检查网络连接。"


if __name__ == "__main__":
    import sys
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8766)
    else:
        mcp.run(transport="stdio")
