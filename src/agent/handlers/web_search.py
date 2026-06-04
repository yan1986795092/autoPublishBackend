"""WebSearch Agent — 实时搜索 (MCP Server → DuckDuckGo → Bing/SerpAPI)"""

import asyncio
import os
import re
import logging
import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def handle_web_search(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """网络搜索 — 支持通过 MCP 工具或直接调用

    优先级：
    1. ToolRegistry 中的 search 工具（web_search）
    2. MCP 工具列表（向后兼容）
    3. 直接 DuckDuckGo 调用
    4. 有 API Key 时使用 Bing / SerpAPI
    """
    query = task_input.get("query", "")
    count = task_input.get("count", 5)

    # 优先：通过 ToolRegistry 查找 search 工具（统一工具入口）
    tool_registry = task_input.get("_tool_registry")
    if tool_registry is not None:
        search_tool = tool_registry.get("web_search")
        if search_tool is None:
            # 也尝试通过 category 查找
            search_tools = tool_registry.get_by_category("search")
            search_tool = search_tools[0] if search_tools else None

        if search_tool is not None:
            try:
                result = await search_tool.execute(query=query, max_results=count)
                return {
                    "status": "completed",
                    "agent_type": "WebSearch",
                    "output": {"content": result, "results": [], "source": "mcp"},
                    "tokens_used": 0,
                    "thinking_steps": [{
                        "agent": "WebSearch",
                        "action": "搜索完成",
                        "reasoning": f"ToolRegistry 搜索：{query}",
                        "output_summary": f"通过工具「{search_tool.name}」搜索到结果",
                        "status": "done",
                    }],
                }
            except Exception as e:
                logger.warning(f"ToolRegistry search failed, falling back: {e}")

    # 兜底：从原始 mcp_tools 列表中查找（向后兼容）
    if mcp_tools is None:
        mcp_tools = task_input.get("_mcp_tools", [])
    for tool in mcp_tools:
        tool_name = getattr(tool, "name", None) or str(tool)
        if "web_search" in tool_name or "search" in tool_name:
            try:
                from langchain_core.runnables import RunnableConfig
                result = await tool._arun(query=query, max_results=count, config=RunnableConfig())
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

    # ── 直接搜索降级链（ToolRegistry/MCP 均不可用时）──
    # 优先级：Bing中国（国内直连）→ 360搜索 → DuckDuckGo（境外）→ SerpAPI（需 key）→ LLM 知识

    # Layer 1: Bing 中国版（国内可直连，完全免费）
    try:
        results = await _search_bing_cn(query, count)
        if results:
            content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: Bing中国）："]
            for r in results:
                content_parts.append("- [%s](%s): %s" % (r['title'], r['url'], r.get('snippet', '')))
            return {
                "status": "completed", "agent_type": "WebSearch",
                "output": {"content": "\n".join(content_parts), "results": results, "source": "bing_cn", "query": query},
                "tokens_used": 0,
                "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": "搜索：%s" % query, "output_summary": "Bing中国搜索找到 %d 条结果" % len(results), "status": "done"}],
            }
    except Exception as e:
        logger.warning("Bing CN direct search failed: %s" % e)

    # Layer 2: 360搜索（国内可直连，完全免费）
    try:
        results = await _search_360(query, count)
        if results:
            content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: 360搜索）："]
            for r in results:
                content_parts.append("- [%s](%s): %s" % (r['title'], r['url'], r.get('snippet', '')))
            return {
                "status": "completed", "agent_type": "WebSearch",
                "output": {"content": "\n".join(content_parts), "results": results, "source": "360", "query": query},
                "tokens_used": 0,
                "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": "搜索：%s" % query, "output_summary": "360搜索找到 %d 条结果" % len(results), "status": "done"}],
            }
    except Exception as e:
        logger.warning("360 search failed: %s" % e)

    # Layer 2: DuckDuckGo HTML (境外网络)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS
        raw_results = await asyncio.to_thread(
            lambda: list(DDGS(timeout=20).text(query, max_results=count))
        )
        results = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", ""), "source": "duckduckgo"}
            for r in raw_results
        ]
        if results:
            content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: DuckDuckGo）："]
            for r in results:
                content_parts.append(f"- [{r['title']}]({r['url']}): {r.get('snippet', '')}")
            return {
                "status": "completed", "agent_type": "WebSearch",
                "output": {"content": "\n".join(content_parts), "results": results, "source": "duckduckgo", "query": query},
                "tokens_used": 0,
                "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": f"搜索：{query}", "output_summary": f"DuckDuckGo 找到 {len(results)} 条结果", "status": "done"}],
            }
    except Exception as e:
        logger.warning(f"DuckDuckGo failed: {e}")

    # Layer 3: SerpAPI（需 API Key）
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search",
                    params={"q": query, "num": count},
                    headers={"Authorization": f"Bearer {serpapi_key}"},
                )
                data = resp.json()
                results = [
                    {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", ""), "source": "serpapi"}
                    for r in data.get("organic_results", [])[:count]
                ]
                if results:
                    content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: SerpAPI）："]
                    for r in results:
                        content_parts.append(f"- [{r['title']}]({r['url']}): {r.get('snippet', '')}")
                    return {
                        "status": "completed", "agent_type": "WebSearch",
                        "output": {"content": "\n".join(content_parts), "results": results, "source": "serpapi", "query": query},
                        "tokens_used": 0,
                        "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": f"搜索：{query}", "output_summary": f"SerpAPI 找到 {len(results)} 条结果", "status": "done"}],
                    }
        except Exception as e:
            logger.warning(f"SerpAPI failed: {e}")

    # 全部失败：让 LLM 用自身知识生成（不阻塞链路）
    return {
        "status": "completed", "agent_type": "WebSearch",
        "output": {"content": f"所有搜索引擎均不可用，Agent 将使用内部知识生成关于「{query}」的内容。", "results": [], "fallback": "llm_knowledge"},
        "tokens_used": 0,
        "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": f"搜索：{query}", "output_summary": "搜索引擎不可用，将使用 LLM 内部知识生成", "status": "done"}],
    }


async def _search_bing_cn(query: str, max_results: int = 5) -> list[dict]:
    """Bing 中国版网页爬取 — 国内可直连，完全免费"""
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    results = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(
            "https://cn.bing.com/search",
            params={"q": query, "ensearch": "0"},
            headers=headers,
        )

        if resp.status_code != 200:
            return []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
        except ImportError:
            soup = __import__("bs4").BeautifulSoup(resp.text, "html.parser")

        for item in soup.select("li.b_algo, .b_algo"):
            title_elem = item.select_one("h2 a") or item.select_one("a")
            if not title_elem:
                continue
            url = title_elem.get("href", "")
            if not url or not url.startswith("http"):
                continue
            title = title_elem.get_text(strip=True)
            snippet_elem = item.select_one(".b_paractl") or item.select_one(".b_caption p")
            snippet = snippet_elem.get_text(separator=" ", strip=True) if snippet_elem else ""
            results.append({"title": title, "url": url, "snippet": snippet, "source": "bing_cn"})
            if len(results) >= max_results:
                break

    return results


async def _search_360(query: str, max_results: int = 5) -> list[dict]:
    """360 搜索爬取 — 国内可直连，完全免费"""
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    results = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(
            "https://www.so.com/s",
            params={"q": query},
            headers=headers,
        )

        if resp.status_code != 200:
            return []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
        except ImportError:
            soup = __import__("bs4").BeautifulSoup(resp.text, "html.parser")

        for item in soup.select(".res-list, .result"):
            title_elem = item.select_one("h3 a") or item.select_one(".res-title a")
            if not title_elem:
                continue
            url = title_elem.get("href", "")
            if not url or not url.startswith("http"):
                continue
            title = title_elem.get_text(strip=True)
            snippet_elem = item.select_one(".res-desc, .res-con")
            snippet = snippet_elem.get_text(separator=" ", strip=True) if snippet_elem else ""
            results.append({"title": title, "url": url, "snippet": snippet, "source": "360"})
            if len(results) >= max_results:
                break

    return results


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
