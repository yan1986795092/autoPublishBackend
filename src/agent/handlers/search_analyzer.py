"""SearchAnalyzer — 基于 WebSearch 结果的搜索结果分析

依赖 WebSearch 的输出，分析搜索到的信息中的关键主题、结构和洞察。
必须在 WebSearch 之后执行，不可独立使用。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .tool_utils import call_llm
from ..result import agent_response
from src.core.exceptions import HandlerError
from src.logging_shared.config import get_logger

logger = get_logger("agent.search_analyzer")

SYSTEM_PROMPT = """你是一个搜索结果分析专家。你的任务是基于搜索到的信息，提取关键主题、结构和洞察。

分析要求：
1. 从搜索结果中提取关键主题和结构，不要依赖 LLM 训练数据中的记忆
2. 如果搜索结果为空或信息不足，如实说明，不要编造
3. 按重要性对主题排序
4. 输出结构化的分析结果，方便后续内容创作使用

输出格式：
先输出分析摘要，然后列出关键发现。"""


async def handle_search_analyzer(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """SearchAnalyzer 主入口：分析搜索结果中的关键主题和结构"""
    try:
        # ── 输入：只读 Schema 声明的字段 ──
        # search_results 可能为空（当没有前置 WebSearch 或 WebSearch 未产出结果时）
        search_results_raw: list | str = task_input.get("search_results", [])
        topic: str = task_input.get("topic", "")

        # 将 search_results 数组转为文本
        search_contents: list[str] = []
        if isinstance(search_results_raw, list):
            for item in search_results_raw:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    url = item.get("url", "")
                    snippet = item.get("snippet", "") or item.get("content", "") or item.get("summary", "")
                    parts = []
                    if title:
                        parts.append(f"[{title}]({url})" if url else title)
                    if snippet:
                        parts.append(snippet)
                    if parts:
                        search_contents.append(": ".join(parts))
                elif isinstance(item, str):
                    search_contents.append(item)
        elif isinstance(search_results_raw, str):
            search_contents.append(search_results_raw)

        combined_search = "\n\n".join(search_contents) if search_contents else ""

        # 降级：无搜索数据
        if not combined_search:
            logger.warning("SearchAnalyzer 未收到搜索数据，返回空分析")
            return agent_response(
                "SearchAnalyzer",
                content="",
                output_type="text",
                output_extra={
                    "trends": [],
                    "keywords": [],
                    "summary": "无搜索数据，无法进行分析",
                    "has_data": False,
                },
                output_fields={
                    "search_analysis": "",
                },
                thinking_steps=[{
                    "agent": "SearchAnalyzer",
                    "action": "跳过",
                    "reasoning": "无搜索数据，跳过分析",
                    "output_summary": "无数据",
                    "status": "done",
                }],
            )

        prompt = f"""## 分析主题
{topic}

## 搜索结果素材
{combined_search[:8000]}

请分析以上搜索结果，提取：
1. 最相关的 3-5 个关键主题/方向
2. 每个主题的关键信息
3. 整体信息结构和脉络
4. 可用于内容创作的核心素材
直接输出分析结果，不要加「好的」等前缀。"""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 流式回调：发送 reasoning 和 llm_chunk 事件到前端
        on_reasoning = task_input.get("_on_reasoning")
        on_llm_chunk = task_input.get("_on_llm_chunk")

        llm_result = await call_llm(
            llm, messages,
            agent_type="SearchAnalyzer",
            stream=True,
            on_chunk=on_llm_chunk,
            on_reasoning=on_reasoning,
        )

        analysis = llm_result.content.strip()

        # 从分析文本中简单提取关键词
        keywords = []
        for line in analysis.split("\n"):
            line = line.strip()
            if line.startswith("##") or line.startswith("**") or (line and line[0].isdigit()):
                clean = line.lstrip("#* .").strip()
                if clean and len(clean) < 50:
                    keywords.append(clean)

        return agent_response(
            "SearchAnalyzer",
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
            output_fields={
                "search_analysis": analysis,
            },
            thinking_steps=[{
                "agent": "SearchAnalyzer",
                "action": "分析完成",
                "reasoning": f"分析 {len(search_contents)} 个搜索结果，提取 {len(keywords)} 个关键主题",
                "output_summary": f"分析了 {len(search_contents)} 条搜索结果",
                "tokens_used": llm_result.total_tokens,
                "status": "done",
            }],
        )

    except Exception as e:
        logger.error("SearchAnalyzer 执行失败: %s", e, exc_info=True)
        raise HandlerError(str(e), handler="SearchAnalyzer") from e
