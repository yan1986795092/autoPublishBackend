"""HotTopics — LangChain BaseTool 版本"""
from __future__ import annotations
import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import httpx


class HotTopicsInput(BaseModel):
    platform: str = Field(default="weibo", description="平台: weibo, zhihu")
    count: int = Field(default=10, description="返回数量")


class HotTopicsTool(BaseTool):
    name: str = "HotTopics"
    description: str = "获取各平台热搜/热榜"
    args_schema: Type[BaseModel] = HotTopicsInput
    api_key: str = ""
    provider: str = "mock"

    async def _arun(self, platform: str = "weibo", count: int = 10, **kwargs) -> str:
        if self.provider == "mock":
            return json.dumps(self._mock_topics(platform, count), ensure_ascii=False)
        try:
            result = await self._fetch_real_topics(platform, count)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps(self._mock_topics(platform, count), ensure_ascii=False)

    async def _fetch_real_topics(self, platform: str, count: int) -> dict:
        topics = []
        source = ""
        if platform == "weibo":
            topics, source = await self._fetch_weibo_topics(count)
        elif platform == "zhihu":
            topics, source = await self._fetch_zhihu_topics(count)
        if not topics:
            topics = self._mock_topics(platform, count)["data"]["topics"]
            source = "fallback"
        return {
            "success": True,
            "data": {
                "platform": platform,
                "topics": topics[:count],
                "source": source,
                "count": min(count, len(topics)),
            },
            "summary": f"从 {platform} 获取到 {min(count, len(topics))} 条热搜",
        }

    async def _fetch_weibo_topics(self, count: int) -> tuple[list, str]:
        topics = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://weibo.com/ajax/side/hotSearch",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    band_list = data.get("data", {}).get("band_list", [])
                    for item in band_list[:count]:
                        topics.append({
                            "title": f"#{item.get('word', '')}#",
                            "heat": f"🔥 {item.get('num', 0)}",
                            "tag": item.get("category", "热点"),
                        })
                    if topics:
                        return topics, "weibo_official"
        except Exception:
            pass
        return topics, ""

    async def _fetch_zhihu_topics(self, count: int) -> tuple[list, str]:
        topics = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.zhihu.com/top-story/hot-list",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    for item in items[:count]:
                        target = item.get("target", {})
                        topics.append({
                            "title": target.get("title", ""),
                            "heat": f"🔥 {item.get('detail_text', '')}",
                            "tag": "科技",
                        })
                    if topics:
                        return topics, "zhihu_official"
        except Exception:
            pass
        return topics, ""

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async _arun")

    def _mock_topics(self, platform: str, count: int) -> dict:
        weibo_topics = [
            {"title": "#AI 编程助手发布新版本#", "heat": "🔥 520万", "tag": "科技"},
            {"title": "#2026 全球 AI 大会#", "heat": "🔥 480万", "tag": "科技"},
            {"title": "#多模态大模型突破#", "heat": "🔥 350万", "tag": "科技"},
            {"title": "#今日热点新闻#", "heat": "🔥 300万", "tag": "社会"},
            {"title": "#周末去哪玩#", "heat": "🔥 280万", "tag": "生活"},
            {"title": "#新片上映推荐#", "heat": "🔥 250万", "tag": "娱乐"},
            {"title": "#健康饮食指南#", "heat": "🔥 200万", "tag": "健康"},
            {"title": "#AI 绘画新玩法#", "heat": "🔥 180万", "tag": "科技"},
            {"title": "#开源项目推荐#", "heat": "🔥 150万", "tag": "科技"},
            {"title": "#每日英语学习#", "heat": "🔥 120万", "tag": "教育"},
        ]
        zhihu_topics = [
            {"title": "如何评价 2026 年 AI 编程工具的发展？", "heat": "🔥 890万", "tag": "科技"},
            {"title": "大模型落地应用的挑战有哪些？", "heat": "🔥 760万", "tag": "科技"},
            {"title": "2026 年最值得学习的编程语言", "heat": "🔥 650万", "tag": "科技"},
            {"title": "如何搭建个人知识管理系统？", "heat": "🔥 520万", "tag": "生活"},
            {"title": "独立开发者如何推广产品？", "heat": "🔥 480万", "tag": "创业"},
        ]
        topics = weibo_topics if platform in ("weibo", "unknown") else zhihu_topics
        return {
            "success": True,
            "data": {
                "platform": platform,
                "topics": topics[:count],
                "source": "mock",
                "count": min(count, len(topics)),
            },
            "summary": f"从 {platform} 获取到 {min(count, len(topics))} 条热搜",
        }
