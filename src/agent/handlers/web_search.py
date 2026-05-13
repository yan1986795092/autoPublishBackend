"""WebSearch Agent — 实时搜索 (SerpAPI → DuckDuckGo → 内部知识) + URL 抓取"""

import os
import re
import json
import logging
import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def handle_web_search(task_input: dict, llm: ChatOpenAI) -> dict:
    query = task_input.get("query", "")
    count = task_input.get("count", 5)
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
            async with httpx.AsyncClient(timeout=10.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }) as client:
                resp = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
                text = resp.text
                # Use lookaheads so the regex works regardless of HTML attribute order
                link_pattern = re.findall(r'<a(?=[^>]*class="result__a")[^>]*href="([^"]*)"[^>]*>([^<]*)</a>', text)
                snippet_pattern = re.findall(r'<a(?=[^>]*class="result__snippet")[^>]*>([^<]*(?:<[^/][^>]*>[^<]*</[^>]*>)?[^<]*)</a>', text)
                for i, (url, title) in enumerate(link_pattern[:count]):
                    snippet = snippet_pattern[i] if i < len(snippet_pattern) else ""
                    snippet = re.sub(r"<[^>]+>", "", snippet).strip()
                    results.append({"title": title.strip(), "url": url, "snippet": snippet, "source": "duckduckgo"})
                if results:
                    source = "duckduckgo"
        except Exception as e:
            logger.warning(f"DuckDuckGo failed: {e}")

    if not results:
        return {
            "status": "completed", "agent_type": "WebSearch",
            "output": {"content": f"搜索服务暂时不可用，Agent 将使用内部知识生成关于「{query}」的内容。", "results": [], "fallback": "web_search_unavailable"},
            "tokens_used": 0,
            "thinking_steps": [{"agent": "WebSearch", "action": "search", "reasoning": f"Searching: {query}", "output_summary": "Search unavailable", "status": "done"}],
        }

    content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: {source}）："]
    for r in results:
        content_parts.append(f"- [{r['title']}]({r['url']}): {r.get('snippet', '')}")

    return {
        "status": "completed", "agent_type": "WebSearch",
        "output": {"content": "\n".join(content_parts), "results": results, "source": source, "query": query},
        "tokens_used": 0,
        "thinking_steps": [{"agent": "WebSearch", "action": "search", "reasoning": f"Searching: {query}", "output_summary": f"Found {len(results)} results via {source}", "status": "done"}],
    }


async def handle_url_crawl(task_input: dict, llm: ChatOpenAI) -> dict:
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
