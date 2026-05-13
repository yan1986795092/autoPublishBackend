"""ReviewAgent — 内容质量审核、违禁词检查"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import QUALITY_CHECKS, check_banned_words


async def handle_review(task_input: dict, llm: ChatOpenAI) -> dict:
    content = task_input.get("content", "")
    if not content:
        dep_context = task_input.get("_context", {})
        for cdata in dep_context.values():
            if cdata.get("content"):
                content = cdata["content"]
                break

    content_type = task_input.get("content_type", "general")
    banned_found = check_banned_words(content) if content else []

    if not content:
        return {
            "status": "completed", "agent_type": "ReviewAgent",
            "output": {"passed": False, "banned_words": [], "suggestions": ["无内容可审核"]},
            "tokens_used": 0, "thinking_steps": [],
        }

    prompt = f"""## 审核任务

审核以下{content_type}内容的质量：
{chr(10).join(f'- {c}' for c in QUALITY_CHECKS)}

内容：
{content[:3000]}

输出格式：REASONING: ... OUTPUT: 总体评价：[通过/需要修改] 改进建议：..."""

    response = await llm.ainvoke([
        SystemMessage(content="你是一个内容审核专家。"),
        HumanMessage(content=prompt),
    ])

    quality = response.content
    passed = len(banned_found) == 0 and "不通过" not in quality
    suggestions = [line for line in quality.split("\n") if any(kw in line for kw in ["建议", "需要", "改进"])][:5]

    return {
        "status": "completed", "agent_type": "ReviewAgent",
        "output": {"passed": passed, "banned_words": banned_found, "suggestions": suggestions, "content": quality},
        "tokens_used": len(response.content) // 4,
        "thinking_steps": [{
            "agent": "ReviewAgent", "action": "validate",
            "reasoning": f"Reviewing {content_type} content",
            "output_summary": f"{'Passed' if passed else 'Needs improvement'}, {len(banned_found)} banned words",
            "status": "done",
        }],
    }
