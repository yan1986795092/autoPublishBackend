"""ReviewAgent — 内容质量审核、违禁词检查、事实核查

使用 PromptEngine 构建审核 prompt，支持记忆上下文注入。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import check_banned_words
from .factcheck import factcheck_report
from src.prompts import PromptBuilder
from src.core.exceptions import HandlerError
from src.utils.tokenizer import estimate_tokens
from src.logging_shared.config import get_logger

logger = get_logger("agent.review")


async def handle_review(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        content = task_input.get("content", "")
        if not content:
            dep_context = task_input.get("_context", {})
            for cdata in dep_context.values():
                if cdata.get("content"):
                    content = cdata["content"]
                    break

        content_type = task_input.get("content_type", "general")
        banned_found = check_banned_words(content) if content else []
        known_facts = task_input.get("known_facts")

        if not content:
            return {
                "status": "completed", "agent_type": "ReviewAgent",
                "output": {"passed": False, "banned_words": [], "factcheck": None, "suggestions": ["无内容可审核"]},
                "tokens_used": 0, "thinking_steps": [],
            }

        # 事实核查
        fc_report = factcheck_report(content, known_facts) if content else None

        # 记忆上下文
        memory_context = task_input.get("_memory_context", "")
        memory_section = f"\n\n{memory_context}" if memory_context else ""
        # 创作上下文（CreativeAnchors 等）
        extra_context = task_input.get("_extra_context", "")
        if extra_context:
            memory_section += f"\n\n{extra_context}"

        # ── 用 PromptBuilder 构建 ──
        builder = PromptBuilder()
        builder.system("system.reviewer")

        # ── Skill system prompt injection ──
        skill_system_prompt = task_input.get("_skill_system_prompt", "")
        if skill_system_prompt:
            builder.context(f"\n\n=== 创作指导 (Skill: {task_input.get('_skill_name', 'Custom')}) ===\n{skill_system_prompt}\n===")

        builder.instructions("review.task", {
            "content": content[:3000],
            "content_type": content_type,
            "memory_section": memory_section,
        })

        # 注入工具描述
        tool_descriptions = task_input.get("_tool_descriptions", "")
        if tool_descriptions:
            builder.context(f"\n可用工具：\n{tool_descriptions}")

        composed = builder.build()
        lc_messages = builder.build_langchain()

        # ── 工具感知的 LLM 调用 ──
        from .tool_utils import call_llm

        quality = await call_llm(
            llm, lc_messages,
            tool_registry=task_input.get("_tool_registry"),
            agent_type="ReviewAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
        )
        passed = len(banned_found) == 0 and "需要修改" not in quality
        suggestions = [
            line.strip() for line in quality.split("\n")
            if any(kw in line for kw in ["建议", "问题", "修改", "不足"])
        ]

        return {
            "status": "completed", "agent_type": "ReviewAgent",
            "output": {
                "passed": passed,
                "banned_words": banned_found,
                "factcheck": fc_report,
                "suggestions": suggestions[:5],
                "content": quality,
            },
            "tokens_used": estimate_tokens(quality),
            "prompt_versions": composed.used_templates,
            "thinking_steps": [{
                "agent": "ReviewAgent", "action": "审核完成",
                "reasoning": f"审核{content_type}内容质量",
                "output_summary": f"{'通过 ✓' if passed else '需修改'} · 发现违禁词 {len(banned_found)} 个 · 事实核查 {fc_report['total_claims_checked'] if fc_report else 0} 处",
                "status": "done",
            }],
        }
    except Exception as e:
        logger.error("ReviewAgent 执行失败", error=str(e), exc_info=True)
        raise HandlerError(str(e), handler="ReviewAgent") from e
