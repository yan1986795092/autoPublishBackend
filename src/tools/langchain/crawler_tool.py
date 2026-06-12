"""Crawler — LangChain BaseTool 版本"""
from __future__ import annotations
import json
import httpx
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class CrawlerInput(BaseModel):
    url: str = Field(description="要抓取的网页 URL")
    max_length: int = Field(default=2000, description="最大内容长度")


class CrawlerTool(BaseTool):
    name: str = "Crawler"
    description: str = "抓取指定 URL 的文本内容"
    args_schema: Type[BaseModel] = CrawlerInput
    api_key: str = ""
    provider: str = "mock"

    async def _arun(self, url: str, max_length: int = 2000, **kwargs) -> str:
        if self.provider != "mock" and self.api_key:
            result = await self._real_crawl(url, max_length)
            return json.dumps(result, ensure_ascii=False)
        return json.dumps(self._mock_crawl(url, max_length), ensure_ascii=False)

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("请使用异步 _arun")

    async def _real_crawl(self, url: str, max_length: int) -> dict:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    title = soup.title.string if soup.title else ""
                    text = soup.get_text(separator="\n", strip=True)
                    return {
                        "success": True,
                        "data": {
                            "url": url,
                            "title": title,
                            "content": text[:max_length],
                            "content_length": len(text),
                        },
                        "summary": f"成功抓取：{title or url}",
                    }
        except ImportError:
            pass  # beautifulsoup 未安装
        except Exception as e:
            return {"success": False, "error": str(e)}
        return self._mock_crawl(url, max_length)

    def _mock_crawl(self, url: str, max_length: int) -> dict:
        return {
            "success": True,
            "data": {
                "url": url,
                "title": "Mock Page Title",
                "content": f"这是从 {url} 抓取的内容（模拟）。当前 AI 技术发展迅速，各大厂商纷纷推出新一代大语言模型。在自然语言处理、计算机视觉、多模态理解等领域都取得了突破性进展。"[:max_length],
                "content_length": min(200, max_length),
                "mock": True,
            },
            "summary": f"[Mock] 抓取：{url}",
        }
