"""FixAgent — 根据 ReviewAgent 的审核意见自动修复内容

FixAgent 接收 WritingAgent 的原始内容 + ReviewAgent 的审核意见，
逐条根据改进建议修改原文，输出优化后的最终版本。

这个 Agent 让整个 pipeline 形成闭环：写 → 审 → 修 → 输出。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .tool_utils import call_llm
from ..agent_utils import extract_output
from src.core.exceptions import HandlerError
from src.logging_shared.config import get_logger

logger = get_logger("agent.fix")

FIX_SYSTEM_PROMPT = """你是一个专业的内容优化编辑。
你的工作是根据审核意见，对原文进行精准修改，而不是重写。

核心原则：
1. 逐条处理每一条审核意见，确保每条都被覆盖
2. 保留原文的核心观点、风格和个人表达
3. 不要过度修改——只有审核意见指出的问题才需要改
4. 输出修改后的完整内容，不要只输出修改的部分
5. 如果审核意见是"通过"，直接返回原文即可"""


async def handle_fix(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """FixAgent 主入口：根据审核意见修改内容"""
    try:
        # 从依赖上下文中获取 WritingAgent 的内容
        original_content = task_input.get("content", "")
        dep_context = task_input.get("_context", {})

        # 找 WritingAgent 的输出
        writing_content = ""
        review_content = ""

        for cid, cdata in dep_context.items():
            agent = cdata.get("agent", "")
            content = cdata.get("content", "")
            if agent == "WritingAgent" and content:
                writing_content = content
            elif agent == "ReviewAgent" and content:
                review_content = content

        # fallback: 直接从 task_input
        if not writing_content:
            writing_content = original_content or task_input.get("topic", "")

        if not writing_content:
            return {
                "status": "completed", "agent_type": "FixAgent",
                "output": {"content": "", "type": "text"},
                "tokens_used": 0,
                "thinking_steps": [{
                    "agent": "FixAgent", "action": "跳过",
                    "reasoning": "无内容可修复",
                    "output_summary": "无内容",
                    "status": "done",
                }],
            }

        # 如果没有审核意见或审核结论是"通过"，直接返回原文
        if not review_content or "**审核结论：** 通过" in review_content:
            return {
                "status": "completed", "agent_type": "FixAgent",
                "output": {"content": writing_content, "type": "text"},
                "tokens_used": 0,
                "thinking_steps": [{
                    "agent": "FixAgent", "action": "无需修改",
                    "reasoning": "审核通过，内容无需修改",
                    "output_summary": f"保留原文 {len(writing_content)} 字",
                    "status": "done",
                }],
            }

        if llm is None:
            # 无 LLM 时返回原文（优雅降级）
            return {
                "status": "completed", "agent_type": "FixAgent",
                "output": {"content": writing_content, "type": "text"},
                "tokens_used": 0,
                "thinking_steps": [{
                    "agent": "FixAgent", "action": "完成（降级）",
                    "reasoning": "无 LLM，保留原文",
                    "output_summary": f"保留原文 {len(writing_content)} 字",
                    "status": "done",
                }],
            }

        prompt = f"""## 内容修复任务

### 原文
{writing_content[:4000]}

### 审核意见
{review_content[:3000]}

请逐条处理审核意见，输出修改后的完整内容。
如果某条意见不适用或修改会损害原文质量，可以跳过。
直接输出修改后的完整内容，不要加「以下是修改后的内容」等前缀。"""

        lc_messages = [
            SystemMessage(content=FIX_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        on_chunk = task_input.get("_on_llm_chunk")
        llm_result = await call_llm(
            llm, lc_messages,
            agent_type="FixAgent",
            stream=True,
            on_chunk=on_chunk,
        )

        fixed = extract_output(llm_result.content) or llm_result.content.strip()
        if not fixed:
            fixed = writing_content

        return {
            "status": "completed", "agent_type": "FixAgent",
            "output": {"content": fixed, "type": "text"},
            "tokens_used": llm_result.total_tokens,
            "prompt_tokens": llm_result.prompt_tokens,
            "completion_tokens": llm_result.completion_tokens,
            "thinking_steps": [{
                "agent": "FixAgent", "action": "修复完成",
                "reasoning": f"根据审核意见修改，{len(review_content)}字审核意见",
                "output_summary": f"修改后约 {len(fixed)} 字",
                "tokens_used": llm_result.total_tokens,
                "status": "done",
            }],
        }

    except Exception as e:
        logger.error("FixAgent 执行失败", error=str(e), exc_info=True)
        raise HandlerError(str(e), handler="FixAgent") from e
