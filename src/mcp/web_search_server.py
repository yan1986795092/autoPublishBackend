"""Open-WebSearch MCP Server — 实时网络搜索服务

基于 Model Context Protocol (MCP) SDK，
提供网页搜索、新闻搜索、内容抓取等工具。

引擎优先级（国内可直连）：
  1. 百度搜索（baidu）— 免费，无需 API Key，国内秒访问
  2. DuckDuckGo — 免费，无需 API Key（依赖境外网络）
  3. Bing Web Search — 需 BING_API_KEY
  4. SerpAPI — 需 SERPAPI_API_KEY
"""
import asyncio
import logging
import os
import re
from typing import Any

# 抑制 MCP SDK 内部 INFO 日志（如 "Processing request of type ListToolsRequest"），
# 这些消息来自 mcp.server 子进程的 rich 日志输出，会混入主进程的终端显示。
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# 创建 FastMCP 服务实例
mcp = FastMCP(
    name="open-websearch",
    instructions="实时网络搜索工具，支持网页搜索、新闻搜索和页面内容抓取。优先使用百度搜索（国内可直连），无需 API Key。",
    dependencies=["duckduckgo_search>=7.0", "httpx", "beautifulsoup4"],
)


# ──────────────────────────────────────────
# 搜索工具函数
# ──────────────────────────────────────────


async def _search_bing_cn(query: str, max_results: int = 5) -> list[dict]:
    """Bing 中国版搜索 — 国内可直接访问，完全免费

    直接爬取 https://cn.bing.com/search 的结果，无需任何 API Key。
    返回格式: list[dict]，每个 dict 包含 title/href/body/source 字段
    """
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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

            # 提取搜索结果摘要片段
            snippet_elem = item.select_one(".b_paractl") or item.select_one(".b_caption p")
            snippet = snippet_elem.get_text(separator=" ", strip=True) if snippet_elem else ""

            results.append({
                "title": title,
                "href": url,
                "body": snippet,
                "source": "bing",
            })

            if len(results) >= max_results:
                break

    return results


async def _search_360(query: str, max_results: int = 5) -> list[dict]:
    """360 搜索 — 国内可直接访问，完全免费

    直接爬取 https://www.so.com/s 的结果，无需任何 API Key。
    """
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

            results.append({
                "title": title,
                "href": url,
                "body": snippet,
                "source": "360",
            })

            if len(results) >= max_results:
                break

    return results


async def _search_baidu_news(query: str, max_results: int = 5) -> list[dict]:
    """国内新闻搜索 — 优先使用 Bing 中国版新闻分类（国内可直接访问）"""
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
        # 使用 Bing 新闻搜索接口
        resp = await client.get(
            "https://cn.bing.com/news/search",
            params={"q": query, "qft": "区间：今日"},
            headers=headers,
        )

        if resp.status_code == 200:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
            except ImportError:
                soup = __import__("bs4").BeautifulSoup(resp.text, "html.parser")

            for item in soup.select(".news-item, .b_algo"):
                title_elem = item.select_one("h2 a") or item.select_one("a")
                if not title_elem:
                    continue
                url = title_elem.get("href", "")
                if not url or not url.startswith("http"):
                    continue
                title = title_elem.get_text(strip=True)

                date_elem = item.select_one(".b_attn, .news-date")
                date_str = date_elem.get_text(strip=True) if date_elem else ""

                snippet_elem = item.select_one(".b_paractl") or item.select_one("p")
                snippet = snippet_elem.get_text(separator=" ", strip=True)[:150] if snippet_elem else ""

                results.append({
                    "title": title,
                    "url": url,
                    "body": snippet,
                    "source": date_str or "Bing新闻",
                    "date": date_str,
                })

                if len(results) >= max_results:
                    break

    return results


def _ddgs_text(query: str, max_results: int = 5) -> list[dict]:
    """统一的 DuckDuckGo 文本搜索入口，自动兼容新/旧版本包名"""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS as _DDGS

        with _DDGS(timeout=20) as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except ImportError:
        from duckduckgo_search import DDGS as _DDGS
        with _DDGS(timeout=20) as ddgs:
            return list(ddgs.text(query, max_results=max_results))


def _ddgs_news(query: str, max_results: int = 5) -> list[dict]:
    """统一的 DuckDuckGo 新闻搜索入口，自动兼容新/旧版本包名"""
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            from duckduckgo_search import DDGS as _DDGS

        with _DDGS(timeout=20) as ddgs:
            return list(ddgs.news(query, max_results=max_results))
    except ImportError:
        from duckduckgo_search import DDGS as _DDGS
        with _DDGS(timeout=20) as ddgs:
            return list(ddgs.news(query, max_results=max_results))


@mcp.tool(
    name="web_search",
    description="搜索互联网上的最新信息。输入搜索关键词，返回相关网页结果。适用于查找新闻、文章、教程等资料。",
)
async def web_search(
    query: str,
    max_results: int = 5,
    source: str = "auto",
) -> str:
    """网页搜索 — 多搜索引擎自动降级（国内可直接访问）"""
    engines_tried = []

    # ── 引擎 1: Bing 中国版（国内可直接访问，完全免费） ──
    if source in ("auto", "bing", "baidu"):
        engines_tried.append("Bing中国")
        try:
            results = await _search_bing_cn(query, max_results=max_results)
            if results:
                return _format_web_results(results, query)
            logger.warning("Bing 中国版返回空结果，尝试下一个搜索引擎")
        except Exception as e:
            logger.warning(f"Bing 中国版搜索失败: {e}")

    # ── 引擎 2: DuckDuckGo（免费，需要境外网络访问能力） ──
    if source in ("auto", "duckduckgo"):
        engines_tried.append("DuckDuckGo")
        try:
            results = _ddgs_text(query, max_results=max_results)
            if results:
                return _format_web_results(results, query)
            logger.warning("DuckDuckGo 返回空结果，尝试下一个搜索引擎")
        except Exception as e:
            logger.warning(f"DuckDuckGo 搜索失败: {e}")

    # ── 引擎 3: Bing Web Search（需要 BING_API_KEY 环境变量） ──
    bing_key = os.getenv("BING_API_KEY")
    if bing_key and source in ("auto", "bing"):
        engines_tried.append("Bing")
        try:
            results = await _bing_search(query, bing_key, max_results)
            if results:
                return _format_web_results(results, query)
        except Exception as e:
            logger.warning(f"Bing 搜索失败: {e}")

    # ── 引擎 4: SerpAPI（需要 SERPAPI_API_KEY 环境变量） ──
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key and source in ("auto", "serpapi"):
        engines_tried.append("SerpAPI")
        try:
            results = await _serpapi_search(query, serpapi_key, max_results)
            if results:
                return _format_web_results(results, query)
        except Exception as e:
            logger.warning(f"SerpAPI 搜索失败: {e}")

    tried_str = "、".join(engines_tried) or "无"
    return (
        f"搜索「{query}」暂时不可用（已尝试引擎：{tried_str}）。"
        "建议配置 BING_API_KEY 或 SERPAPI_API_KEY 环境变量。"
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
    """新闻搜索 — 优先使用 Bing 新闻（国内可直接访问）"""
    # ── 引擎 1: Bing 新闻搜索 ──
    try:
        results = await _search_baidu_news(query, max_results=max_results)
        if results:
            lines = [f"**{query}** 相关新闻（共 {len(results)} 条）：\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"**{i}. {r['title']}**")
                if r.get("date"):
                    lines.append(f"   时间：{r['date']}")
                if r.get("body"):
                    lines.append(f"   {r['body'][:150]}...")
                lines.append(f"   🔗 {r['url']}\n")
            return "\n".join(lines)
    except Exception as e:
        logger.warning(f"百度新闻搜索失败: {e}")

    # ── 引擎 2: DuckDuckGo 新闻搜索（降级方案） ──
    try:
        results = _ddgs_news(query, max_results=max_results)
        if results:
            lines = [f"**{query}** 相关新闻（共 {len(results)} 条）：\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"**{i}. {r.get('title', '')}**")
                if r.get("date"):
                    lines.append(f"   来源：{r.get('source', '')}  |  时间：{r.get('date', '')}")
                body = r.get("body", "")[:150]
                if body:
                    lines.append(f"   {body}...")
                lines.append(f"   🔗 {r.get('url', '')}\n")
            return "\n".join(lines)
    except Exception as e:
        logger.warning(f"DuckDuckGo 新闻搜索失败: {e}")

    return f"未找到「{query}」相关的最新新闻。"


@mcp.tool(
    name="fetch_page",
    description="抓取指定网页的主要内容。输入 URL，返回去除了导航栏/广告后的干净文本内容。适用于从搜索结果中获取详细内容。",
)
async def fetch_page(
    url: str,
    max_length: int = 5000,
) -> str:
    """抓取指定网页的正文内容，去除导航栏和广告等干扰元素"""
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

        # 解析 HTML 文档
        soup = BeautifulSoup(resp.text, "html.parser")

        # 移除脚本、样式、导航等干扰元素
        for tag in soup.find_all(
            ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "form"]
        ):
            tag.decompose()

        # 从页面 HTML 中提取标题
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 从页面的 meta 标签中提取描述信息
        meta_desc = ""
        if soup.find("meta", attrs={"name": "description"}):
            meta_desc = soup.find("meta", attrs={"name": "description"}).get("content", "")

        # 提取正文内容，优先使用 article 或 main 标签
        article = soup.find("article") or soup.find("main") or soup
        paragraphs = article.find_all("p")

        text_parts = []
        for p in paragraphs:
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 20:  # 过滤掉长度不足 20 个字符的短段落
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
        logger.warning(f"页面抓取失败 {url}: {e}")
        return f"网页抓取失败：{e}"


# ──────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────

def _format_web_results(results: list[dict], query: str) -> str:
    """将搜索到的结果列表格式化为易于阅读的 Markdown 文本"""
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
    """Bing Web Search API 调用接口"""
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
    """SerpAPI 搜索接口（支持 Google、Bing、DuckDuckGo 等搜索引擎）"""
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
# 主入口：根据传入的 transport 参数启动服务
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if transport == "http":
        # HTTP 模式，供 langchain-mcp-adapters 的 streamable_http 使用
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
    else:
        # stdio 模式，供 langchain-mcp-adapters 使用
        mcp.run(transport="stdio")
