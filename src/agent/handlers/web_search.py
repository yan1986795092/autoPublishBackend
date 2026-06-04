"""WebSearch Agent — Bing中国 + 360搜索 并行降级"""

import asyncio
import logging
import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def handle_web_search(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """网络搜索 — Bing中国 + 360搜索 并行降级，全部失败则 LLM 内部知识

    路径：
    1. 并行降级：Bing中国 + 360搜索同时试，最快返回（总超时 10s）
    2. 全部失败 → LLM 内部知识（不阻塞链路）
    """
    query = task_input.get("query", "")
    count = task_input.get("count", 5)

    # ── 并行降级：Bing中国 + 360搜索 同时发起，谁快用谁 ──
    async def _parallel_search() -> tuple[str, list] | None:
        async def _try_bing():
            try:
                r = await _search_bing_cn(query, count)
                return r if r else None
            except Exception as e:
                logger.warning("Bing CN search failed: %s", e)
                return None
        async def _try_360():
            try:
                r = await _search_360(query, count)
                return r if r else None
            except Exception as e:
                logger.warning("360 search failed: %s", e)
                return None

        done, pending = await asyncio.wait(
            [asyncio.create_task(f()) for f in (_try_bing, _try_360)],
            timeout=4.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for p in pending:
            p.cancel()
        for coro in done:
            res = coro.result()
            if res:
                return ("parallel", res)
        return None

    try:
        parallel_result = await asyncio.wait_for(_parallel_search(), timeout=5.0)
    except asyncio.TimeoutError:
        parallel_result = None

    if parallel_result:
        source, results = parallel_result
        label = "Bing中国" if source == "bing_cn" else "360搜索"
        content_parts = [f"搜索「{query}」找到 {len(results)} 条结果（来源: {label}）："]
        for r in results:
            content_parts.append("- [%s](%s): %s" % (r['title'], r['url'], r.get('snippet', '')))
        return {
            "status": "completed", "agent_type": "WebSearch",
            "output": {"content": "\n".join(content_parts), "results": results, "source": source, "query": query},
            "tokens_used": 0,
            "thinking_steps": [{"agent": "WebSearch", "action": "搜索完成", "reasoning": "搜索：%s" % query, "output_summary": "%s找到 %d 条结果" % (label, len(results)), "status": "done"}],
        }

    # 全部失败：LLM 内部知识（不阻塞链路）
    return {
        "status": "completed", "agent_type": "WebSearch",
        "output": {"content": f"搜索引擎不可用，Agent 将使用内部知识生成关于「{query}」的内容。", "results": [], "fallback": "llm_knowledge"},
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
