"""ReviewAgent — 内容质量审核、违禁词检查、事实核查"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import QUALITY_CHECKS, check_banned_words
from .factcheck import factcheck_report


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

    # ── 记忆注入：历史审核经验 ──
    memory_context = task_input.get("_memory_context", "")
    memory_section = f"\n\n{memory_context}" if memory_context else ""

    prompt = f"""## 审核任务{memory_section}

请对以下{content_type}内容进行审核，重点关注：
- 内容是否完整，有没有虎头蛇尾
- 逻辑是否通顺，有没有前后矛盾
- 语言是否自然，读起来顺不顺口
- 有没有违禁词、敏感词或者容易引发争议的表达
- 段落节奏怎么样，长短搭配是否舒适

内容如下：
{content[:3000]}

请用中文回复，直接给出审核意见，不要用英文。格式参考：
**审核结论：** [通过 / 需要修改]
**主要问题：** [列出发现的问题，如果没有就说"整体质量良好"]
**改进建议：** [具体的改进方向，如果有的话]"""

    messages = [
        SystemMessage(content="你是一位资深内容编辑，擅长发现文章中的问题并给出切实可行的改进建议。回复请用中文，语气友好专业，像在和作者交流而不是打分评判。"),
        HumanMessage(content=prompt),
    ]

    # ── 流式 LLM 调用 ──
    on_llm_chunk = task_input.get("_on_llm_chunk")
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        if chunk.content:
            chunks.append(chunk.content)
            if on_llm_chunk:
                await on_llm_chunk(chunk.content)

    quality = "".join(chunks)
    passed = len(banned_found) == 0 and "需要修改" not in quality
    suggestions = [line.strip() for line in quality.split("\n") if any(kw in line for kw in ["建议", "问题", "修改", "不足"])]

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
        "thinking_steps": [{
            "agent": "ReviewAgent", "action": "审核完成",
            "reasoning": f"审核{content_type}内容质量",
            "output_summary": f"{'通过 ✓' if passed else '需修改'} · 发现违禁词 {len(banned_found)} 个 · 事实核查 {fc_report['total_claims_checked'] if fc_report else 0} 处",
            "status": "done",
        }],
    }
