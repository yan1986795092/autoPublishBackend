"""WebSearch Agent — 博查搜索"""

import logging
import os
from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.core.request_guards import validate_public_http_url

logger = logging.getLogger(__name__)

_BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"


def _prepare_bocha_query(query: str) -> str:
    """准备发送给博查的搜索查询。

    故意隔离处理，以便后续与 DeepSeek 的查询策略对齐，
    而无需改变 WebSearch 的调用流程。
    目前保持用户原始表述不变。
    """
    return (query or "").strip()


def _get_bocha_api_key() -> str:
    settings = get_settings()
    return (
        settings.bocha_api_key
        or os.getenv("BOCHA_API_KEY")
        or os.getenv("BOCHA_SEARCH_API_KEY")
        or ""
    )


async def handle_web_search(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """使用博查执行联网搜索。

    约束：
    1. 仅使用博查，不降级到其他搜索引擎
    2. 搜索词先经过统一预处理，再提交给博查
    3. 若未配置博查 Key，则明确返回失败
    """
    original_query = task_input.get("query", "")
    query = _prepare_bocha_query(original_query)
    count = max(1, min(int(task_input.get("count", 5) or 5), 50))
    freshness = task_input.get("freshness", "noLimit") or "noLimit"
    summary = bool(task_input.get("summary", True))

    api_key = _get_bocha_api_key()
    if not api_key:
        return {
            "status": "failed",
            "agent_type": "WebSearch",
            "error": "未配置博查搜索 API Key",
            "output": {
                "content": "未配置博查搜索 API Key。请设置 `BOCHA_API_KEY` 后重试。",
                "results": [],
                "source": "bocha",
                "query": query,
                "original_query": original_query,
            },
            "tokens_used": 0,
            "thinking_steps": [{
                "agent": "WebSearch",
                "action": "搜索失败",
                "reasoning": f"搜索：{original_query}",
                "output_summary": "缺少 BOCHA_API_KEY",
                "status": "failed",
            }],
        }

    try:
        results = await _search_bocha(
            query=query,
            api_key=api_key,
            max_results=count,
            freshness=freshness,
            summary=summary,
        )
    except Exception as exc:
        logger.warning("博查搜索失败: %s", exc)
        return {
            "status": "failed",
            "agent_type": "WebSearch",
            "error": str(exc),
            "output": {
                "content": f"博查搜索失败：{exc}",
                "results": [],
                "source": "bocha",
                "query": query,
                "original_query": original_query,
            },
            "tokens_used": 0,
            "thinking_steps": [{
                "agent": "WebSearch",
                "action": "搜索失败",
                "reasoning": f"搜索：{original_query}",
                "output_summary": f"博查请求失败：{exc}",
                "status": "failed",
            }],
        }

    content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: 博查）："]
    for result in results:
        content_parts.append(
            "- [%s](%s): %s" % (
                result["title"],
                result["url"],
                result.get("snippet", ""),
            )
        )

    from ..result import agent_response

    return agent_response(
        "WebSearch",
        content="\n".join(content_parts),
        output_type="text",
        output_extra={
            "results": results,
            "source": "bocha",
            "query": query,
            "original_query": original_query,
        },
        thinking_steps=[{
            "agent": "WebSearch",
            "action": "搜索完成",
            "reasoning": f"搜索：{original_query}",
            "output_summary": f"博查找到 {len(results)} 条结果",
            "status": "done",
        }],
    )


async def _search_bocha(
    query: str,
    api_key: str,
    max_results: int = 5,
    freshness: str = "noLimit",
    summary: bool = True,
) -> list[dict[str, Any]]:
    payload = {
        "query": query,
        "count": max_results,
        "freshness": freshness,
        "summary": summary,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(_BOCHA_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return _extract_bocha_results(data)


def _extract_bocha_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    if isinstance(data.get("data"), dict):
        payload = data["data"]

        web_pages = payload.get("webPages")
        if isinstance(web_pages, dict) and isinstance(web_pages.get("value"), list):
            candidates = web_pages["value"]
        else:
            for key in ("items", "results", "value"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = value
                    break
    elif isinstance(data.get("data"), list):
        candidates = data["data"]
    elif isinstance(data.get("results"), list):
        candidates = data["results"]

    normalized: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        title = item.get("name") or item.get("title") or ""
        url = item.get("url") or item.get("link") or item.get("href") or ""
        snippet = (
            item.get("summary")
            or item.get("snippet")
            or item.get("description")
            or item.get("body")
            or ""
        )
        site_name = item.get("siteName") or item.get("site") or ""
        publish_time = item.get("datePublished") or item.get("publishTime") or item.get("date") or ""

        if not title or not url:
            continue

        normalized.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "site_name": site_name,
            "publish_time": publish_time,
            "source": "bocha",
        })

    return normalized


async def handle_url_crawl(task_input: dict, llm: ChatOpenAI | None = None, mcp_tools: list | None = None) -> dict:
    """URLCrawler — 抓取指定 URL 的文本内容"""
    url = task_input.get("url", "")
    max_length = task_input.get("max_length", 3000)
    query = task_input.get("query", "")

    if not url:
        return {
            "status": "failed", "agent_type": "URLCrawler",
            "error": "未提供 URL",
            "output": {"content": ""},
            "tokens_used": 0, "thinking_steps": [],
        }

    try:
        url = validate_public_http_url(url)
    except Exception as exc:
        from ..result import error_response

        return error_response(
            "URLCrawler",
            error=getattr(exc, "detail", str(exc)),
            failure_type="invalid_input",
            content=f"URL 校验失败: {url}",
        )

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

            from ..result import agent_response

            content = f"标题: {title}\n\n来源: {url}\n\n{text}"
            return agent_response(
                "URLCrawler",
                content=content,
                output_type="text",
                output_extra={"url": url, "title": title, "source": source},
                thinking_steps=[{
                    "agent": "URLCrawler", "action": "抓取",
                    "reasoning": f"抓取: {url}",
                    "output_summary": f"从 {title} 获取 {len(text)} 字符",
                    "status": "done",
                }],
            )
    except httpx.TimeoutException:
        from ..result import error_response
        return error_response(
            "URLCrawler",
            error="请求超时",
            failure_type="timeout",
            content=f"抓取超时：{url}",
        )
    except Exception as e:
        logger.warning(f"URLCrawler 执行失败: {e}")
        return error_response(
            "URLCrawler",
            error=str(e),
            failure_type="unknown",
            content=f"抓取失败：{e}",
        )
