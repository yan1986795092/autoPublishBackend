"""ReviewAgent — 内容质量审核、违禁词检查、事实核查

使用 PromptEngine 构建审核 prompt，支持记忆上下文注入。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import check_banned_words
from .factcheck import factcheck_report
from src.prompts import PromptBuilder


async def handle_review(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
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

    # ── 用 PromptBuilder 构建 ──
    composed = (PromptBuilder()
        .system("system.reviewer")
        .instructions("review.task", {
            "content": content[:3000],
            "content_type": content_type,
            "memory_section": memory_section,
        })
        .build())

    lc_messages = composed.build_langchain()

    # ── 流式 LLM 调用 ──
    on_llm_chunk = task_input.get("_on_llm_chunk")
    chunks: list[str] = []
    async for chunk in llm.astream(lc_messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_llm_chunk:
                await on_llm_chunk(chunk.content)

    quality = "".join(chunks)
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
        "tokens_used": len(quality) // 4,
        "prompt_versions": composed.used_templates,
        "thinking_steps": [{
            "agent": "ReviewAgent", "action": "审核完成",
            "reasoning": f"审核{content_type}内容质量",
            "output_summary": f"{'通过 ✓' if passed else '需修改'} · 发现违禁词 {len(banned_found)} 个 · 事实核查 {fc_report['total_claims_checked'] if fc_report else 0} 处",
            "status": "done",
        }],
    }
