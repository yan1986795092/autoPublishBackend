"""WebSearch — LangChain BaseTool 版本"""
from __future__ import annotations
import json
import httpx
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class WebSearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    count: int = Field(default=5, description="返回结果数量")


class WebSearchTool(BaseTool):
    name: str = "WebSearch"
    description: str = "搜索互联网获取实时信息"
    args_schema: Type[BaseModel] = WebSearchInput
    api_key: str = ""
    provider: str = "mock"

    async def _arun(self, query: str, count: int = 5, **kwargs) -> str:
        if self.provider == "mock" or not self.api_key:
            return json.dumps(self._mock_search(query, count), ensure_ascii=False)

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for topic in data.get("RelatedTopics", [])[:count]:
                        if "Text" in topic:
                            results.append({
                                "title": topic.get("Text", "")[:100],
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", "")[:200],
                            })
                    return json.dumps({
                        "success": bool(results),
                        "data": {"query": query, "results": results},
                        "summary": f"找到 {len(results)} 条搜索结果",
                    }, ensure_ascii=False)
        except Exception:
            pass

        return json.dumps(self._mock_search(query, count), ensure_ascii=False)

    def _run(self, query: str, count: int = 5, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")

    def _mock_search(self, query: str, count: int = 5) -> dict:
        return {
            "success": True,
            "data": {
                "query": query,
                "results": [
                    {
                        "title": f"{query} - 最新趋势分析",
                        "url": f"https://example.com/{query}",
                        "snippet": f"关于{query}的最新分析和趋势预测。",
                    },
                    {
                        "title": f"{query}实战指南",
                        "url": f"https://example.com/{query}/guide",
                        "snippet": f"详细的{query}实战教程。",
                    },
                ][:count],
            },
            "summary": f"Mock: 找到 {min(count, 2)} 条搜索结果",
        }
