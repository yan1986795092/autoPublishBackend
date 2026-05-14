"""HotTopicAgent — 热搜/热点抓取 (微博/知乎) + 趋势分析"""

import logging
import httpx
from langchain_openai import ChatOpenAI
from .trend_analysis import analyze_topics

logger = logging.getLogger(__name__)


async def handle_hot_topic(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    platform = task_input.get("platform", "weibo")
    count = task_input.get("count", 10)
    topics = []
    source = "mock"

    if platform in ("weibo", ""):
        try:
            async with httpx.AsyncClient(timeout=10.0, headers={
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
            }) as client:
                resp = await client.get(
                    "https://m.weibo.cn/api/container/getIndex",
                    params={"containerid": "106003type=1", "page_type": "searchall"}
                )
                data = resp.json()
                hot_list = data.get("data", {}).get("cards", [{}])[0].get("card_group", [])
                topics = [
                    {"title": item.get("desc", ""), "heat": f"🔥 {item.get('raw_hot_num', 0)}", "platform": "weibo", "tag": "热点"}
                    for item in hot_list[:count] if item.get("desc")
                ]
                if topics:
                    source = "weibo"
        except Exception as e:
            logger.warning(f"Weibo hot topics failed: {e}")

    if platform == "zhihu" or (not topics and platform != "weibo"):
        try:
            async with httpx.AsyncClient(timeout=10.0, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Cookie": "q_c1=fake; _zap=fake; d_c0=fake",
            }) as client:
                resp = await client.get("https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total")
                data = resp.json()
                zhihu_topics = [
                    {"title": item.get("target", {}).get("title", ""), "heat": f"🔥 {item.get('detail_text', '')}", "platform": "zhihu", "tag": "科技"}
                    for item in data.get("data", [])[:count]
                ]
                if zhihu_topics:
                    topics = zhihu_topics
                    source = "zhihu"
        except Exception as e:
            logger.warning(f"Zhihu hot topics failed: {e}")

    if not topics:
        topics = [{"title": f"#热点话题{i}#", "heat": f"🔥 {500 - i * 30}万", "tag": "社会"} for i in range(count)]
        source = "mock"

    # 趋势分析
    trend = analyze_topics(topics)
    content_ideas = trend.get("content_ideas", [])

    return {
        "status": "completed", "agent_type": "HotTopicAgent",
        "output": {
            "platform": platform, "topics": topics, "source": source,
            "trend_analysis": trend,
            "content_ideas": content_ideas,
        },
        "tokens_used": 0,
        "thinking_steps": [{
            "agent": "HotTopicAgent",             "action": "热点获取完成",
            "reasoning": f"获取 {platform} 热点话题",
            "output_summary": f"获取 {len(topics)} 个热点话题，{trend.get('tag_count', 0)} 个分类",
            "status": "done",
        }],
    }
