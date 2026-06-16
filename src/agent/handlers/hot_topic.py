"""HotTopicAgent — 基于 WebSearch 结果的热点话题分析

依赖 WebSearch 的输出，分析搜索到的信息中的热点趋势、主题分布和关键洞察。
必须在 WebSearch 之后执行，不可独立使用。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .tool_utils import call_llm
from ..result import agent_response
from src.core.exceptions import HandlerError
from src.logging_shared.config import get_logger

logger = get_logger("agent.hot_topic")

SYSTEM_PROMPT = """你是一个热点趋势分析专家。你的任务是基于搜索结果中的信息，识别和分析当前的热点话题与趋势。

分析要求：
1. 从搜索结果中提取关键主题和趋势，不要依赖 LLM 训练数据中的记忆
2. 如果搜索结果为空或信息不足，如实说明，不要编造
3. 按重要性对主题排序
4. 输出结构化的分析结果

输出格式：
先输出分析摘要，然后列出关键发现。"""


async def handle_hot_topic(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """HotTopicAgent 主入口：分析搜索结果中的热点趋势"""
    try:
        # 从依赖上下文中获取 WebSearch / URLCrawler 的搜索结果
        dep_context = task_input.get("_context", {})
        search_contents: list[str] = []

        for _cid, cdata in dep_context.items():
            agent = cdata.get("agent", "")
            content = cdata.get("content", "")
            if agent in ("WebSearch", "URLCrawler") and content:
                search_contents.append(content)

        combined_search = "\n\n---\n\n".join(search_contents)
        topic = task_input.get("topic", "热点分析")

        # 降级：无搜索数据
        if not combined_search:
            logger.warning("HotTopicAgent 未收到搜索数据，返回空分析")
            return agent_response(
                "HotTopicAgent",
                content="",
                output_type="text",
                output_extra={
                    "trends": [],
                    "keywords": [],
                    "summary": "无搜索数据，无法进行热点分析",
                    "has_data": False,
                },
                thinking_steps=[{
                    "agent": "HotTopicAgent",
                    "action": "跳过",
                    "reasoning": "无搜索数据，跳过热点分析",
                    "output_summary": "无数据",
                    "status": "done",
                }],
            )

        prompt = f"""## 分析主题
{topic}

## 搜索结果素材
{combined_search[:8000]}

请分析以上搜索结果，提取：
1. 当前最热门的 3-5 个话题/趋势
2. 每个话题的关键词
3. 整体情感倾向（正面/中性/负面）
4. 如果与产品推广相关，提取可用于推广的洞察
直接输出分析结果，不要加「好的」等前缀。"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        llm_result = await call_llm(
            llm, messages,
            agent_type="HotTopicAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
        )

        analysis = llm_result.content.strip()

        # 从分析文本中简单提取关键词（也可以让 LLM 输出结构化 JSON，但为稳定性先保持纯文本）
        keywords = []
        for line in analysis.split("\n"):
            line = line.strip()
            if line.startswith("##") or line.startswith("**") or (line and line[0].isdigit()):
                # 提取标题行中的关键词
                clean = line.lstrip("#* .").strip()
                if clean and len(clean) < 50:
                    keywords.append(clean)

        return agent_response(
            "HotTopicAgent",
            content=analysis,
            output_type="text",
            tokens_used=llm_result.total_tokens,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            output_extra={
                "trends": keywords[:10],
                "keywords": keywords[:10],
                "summary": analysis[:200],
                "has_data": True,
            },
            thinking_steps=[{
                "agent": "HotTopicAgent",
                "action": "分析完成",
                "reasoning": f"分析 {len(search_contents)} 个搜索结果，识别 {len(keywords)} 个热点主题",
                "output_summary": f"分析了 {len(search_contents)} 条搜索结果",
                "tokens_used": llm_result.total_tokens,
                "status": "done",
            }],
        )

    except Exception as e:
        logger.error("HotTopicAgent 执行失败", error=str(e), exc_info=True)
        raise HandlerError(str(e), handler="HotTopicAgent") from e
