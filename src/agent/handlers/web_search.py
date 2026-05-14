"""WebSearch Agent — 实时搜索 (MCP Server → DuckDuckGo → Bing/SerpAPI)"""

import os
import re
import logging
import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def handle_web_search(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """网络搜索 — 支持通过 MCP 工具或直接调用

    优先级：
    1. MCP 工具（web_search）— 支持多引擎自动降级
    2. 直接 DuckDuckGo 调用
    3. 有 API Key 时使用 Bing / SerpAPI
    """
    query = task_input.get("query", "")
    count = task_input.get("count", 5)

    # 优先：从 task_input._mcp_tools 获取（MCP 工具由 nodes.py 注入）
    if mcp_tools is None:
        mcp_tools = task_input.get("_mcp_tools", [])
        for tool in mcp_tools:
            tool_name = getattr(tool, "name", None) or str(tool)
            if "web_search" in tool_name or "search" in tool_name:
                try:
                    result = await tool._arun(query=query, max_results=count)
                    return {
                        "status": "completed",
                        "agent_type": "WebSearch",
                        "output": {"content": result, "results": [], "source": "mcp"},
                        "tokens_used": 0,
                        "thinking_steps": [{
                            "agent": "WebSearch",
                            "action": "搜索完成",
                            "reasoning": f"MCP 搜索：{query}",
                            "output_summary": f"通过 MCP 工具「{tool_name}」搜索到结果",
                            "status": "done",
                        }],
                    }
                except Exception as e:
                    logger.warning(f"MCP tool {tool_name} failed, falling back to direct search: {e}")

    # ── 兜底：直接 DuckDuckGo ──
    results = []
    source = "none"

    # Layer 1: SerpAPI
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={"q": query, "api_key": serpapi_key, "num": count},
                )
                data = resp.json()
                results = [
                    {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", ""), "source": r.get("source", "")}
                    for r in data.get("organic_results", [])[:count]
                ]
                source = "serpapi"
        except Exception as e:
            logger.warning(f"SerpAPI failed: {e}")

    # Layer 2: DuckDuckGo HTML
    if not results:
        try:
            from duckduckgo_search import DDGS
            with DDGS(timeout=20) as ddgs:
                raw_results = list(ddgs.text(query, max_results=count))
            results = [
                {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", ""), "source": "duckduckgo"}
                for r in raw_results
            ]
            source = "duckduckgo"
        except Exception as e:
            logger.warning(f"DuckDuckGo failed: {e}")

    if not results:
        return {
            "status": "completed", "agent_type": "WebSearch",
            "output": {"content": f"搜索服务暂时不可用，Agent 将使用内部知识生成关于「{query}」的内容。", "results": [], "fallback": "web_search_unavailable"},
            "tokens_used": 0,
            "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": f"搜索：{query}", "output_summary": "搜索服务不可用，将使用内部知识", "status": "done"}],
        }

    content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: {source}）："]
    for r in results:
        content_parts.append(f"- [{r['title']}]({r['url']}): {r.get('snippet', '')}")

    return {
        "status": "completed", "agent_type": "WebSearch",
        "output": {"content": "\n".join(content_parts), "results": results, "source": source, "query": query},
        "tokens_used": 0,
        "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": f"搜索：{query}", "output_summary": f"找到 {len(results)} 条结果（来源: {source}）", "status": "done"}],
    }


async def handle_url_crawl(task_input: dict, llm: ChatOpenAI | None = None, mcp_tools: list | None = None) -> dict:
    """URLCrawler — 抓取指定 URL 的文本内容"""
    url = task_input.get("url", "")
    max_length = task_input.get("max_length", 3000)
    query = task_input.get("query", "")

    if not url:
        return {
            "status": "failed", "agent_type": "URLCrawler",
            "error": "No URL provided",
            "output": {"content": ""},
            "tokens_used": 0, "thinking_steps": [],
        }

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            if resp.status_code != 200:
                return {
                    "status": "failed", "agent_type": "URLCrawler",
                    "error": f"HTTP {resp.status_code}",
                    "output": {"content": f"抓取失败，HTTP {resp.status_code}"},
                    "tokens_used": 0, "thinking_steps": [],
                }

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    tag.decompose()
                title = soup.title.string.strip() if soup.title and soup.title.string else url
                text = soup.get_text(separator="\n", strip=True)
                text = "\n".join(line for line in text.split("\n") if line.strip())[:max_length]
                source = "beautifulsoup"
            except ImportError:
                text = resp.text[:max_length]
                title = url
                source = "raw"

            content = f"标题: {title}\n\n来源: {url}\n\n{text}"
            return {
                "status": "completed", "agent_type": "URLCrawler",
                "output": {"content": content, "url": url, "title": title, "source": source},
                "tokens_used": 0,
                "thinking_steps": [{
                    "agent": "URLCrawler", "action": "crawl",
                    "reasoning": f"Crawling: {url}",
                    "output_summary": f"Got {len(text)} chars from {title}",
                    "status": "done",
                }],
            }
    except httpx.TimeoutException:
        return {
            "status": "failed", "agent_type": "URLCrawler",
            "error": "Request timeout",
            "output": {"content": f"抓取超时：{url}"},
            "tokens_used": 0, "thinking_steps": [],
        }
    except Exception as e:
        logger.warning(f"URLCrawler failed: {e}")
        return {
            "status": "failed", "agent_type": "URLCrawler",
            "error": str(e),
            "output": {"content": f"抓取失败：{e}"},
            "tokens_used": 0, "thinking_steps": [],
        }
