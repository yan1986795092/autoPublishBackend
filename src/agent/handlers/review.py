"""ReviewAgent — 内容质量审核、违禁词检查、事实核查，可选自动修正

使用 PromptEngine 构建审核 prompt，支持记忆上下文注入和平台特定质量检查。
当 rewrite=True 时，审核发现问题会自动触发修正并输出 fixed_content。
"""

import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import check_banned_words
from ..result import agent_response
from .factcheck import factcheck_report
from src.prompts import PromptBuilder
from src.core.exceptions import HandlerError
from src.logging_shared.config import get_logger
from ..platform_profile import get_profile

logger = get_logger("agent.review")


# 修正指令模版
_FIX_SYSTEM_PROMPT = """你是一位专业的内容编辑，负责根据审核意见对原文进行精准修改。

修改原则：
1. 严格按每一条审核意见修改，不要做额外的风格或内容调整
2. 保持原文的结构、语气和核心信息不变
3. 只修改审核意见中指出的问题部分
4. 输出修改后的完整内容（不要只输出修改的部分）
5. 如果审核意见有冲突，优先采纳更严格的那条"""

_FIX_USER_PROMPT = """请根据以下审核意见修改原文。

## 原文
{original_content}

## 审核意见

### 违禁词（需替换）
{banned_words_text}

### 修改建议
{suggestions_text}

### 事实核查问题
{factcheck_text}

## 要求
- 替换所有违禁词为推荐用词
- 逐条应用修改建议
- 修正事实性错误
- 保持原文风格和结构
- 直接输出修改后的完整内容"""


async def handle_review(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        # ── 输入：只读 Schema 声明的字段（article_body 已由 validate_input 保证存在） ──
        content: str = task_input["article_body"]
        platform: str = task_input.get("platform", "")
        content_type: str = task_input.get("content_type", "general")
        banned_found = check_banned_words(content, platform) if content else []
        known_facts = task_input.get("known_facts")

        if not content:
            return agent_response(
                "ReviewAgent",
                output_type="metadata",
                output_extra={
                    "passed": False, "banned_words": [],
                    "factcheck": None, "suggestions": ["无内容可审核"],
                },
                output_fields={
                    "review_text": "",
                    "passed": False,
                    "score": 0,
                    "suggestions": [],
                },
            )

        # ── 加载平台质量检查标准 ──
        platform_quality = task_input.get("_platform_quality_check", "")
        profile = get_profile(platform) if platform and not platform_quality else None
        if not platform_quality and profile:
            platform_quality = profile.get("quality_check", "")
        if platform_quality:
            content_type = f"{content_type}（{platform}）"

        # 事实核查
        fc_report = factcheck_report(content, known_facts) if content else None

        # 记忆上下文（由 nodes.py 统一注入，handler 不主动检索）
        memory_context = task_input.get("_memory_context", "")
        memory_section = f"\n\n{memory_context}" if memory_context else ""
        # 创作上下文（CreativeAnchors 等）
        extra_context = task_input.get("_extra_context", "")
        if extra_context:
            memory_section += f"\n\n{extra_context}"

        # ── 用 PromptBuilder 构建 ──
        builder = PromptBuilder()
        builder.system("system.reviewer")

        # 注入平台质量检查标准
        quality_context = ""
        if platform_quality:
            quality_context = f"\n\n**平台质量检查标准：**\n{platform_quality}"

        # 违禁词检查结果 + 替换建议（注入到 prompt 让 LLM 给出修复意见）
        if banned_found:
            banned_section = "\n\n**违禁词检查结果（请逐条给出修改建议）：**\n"
            for b in banned_found:
                word = b.get("word", "")
                suggestion = b.get("suggestion", "")
                banned_section += f"- 「{word}」"
                if suggestion:
                    banned_section += f" → 建议替换为：{suggestion}"
                banned_section += "\n"
            quality_context += banned_section

        builder.instructions("review.task", {
            "content": content[:3000],
            "content_type": content_type,
            "memory_section": memory_section + quality_context,
        })

        # 注入工具描述
        tool_descriptions = task_input.get("_tool_descriptions", "")
        if tool_descriptions:
            builder.context(f"\n可用工具：\n{tool_descriptions}")

        composed = builder.build()
        lc_messages = builder.build_langchain()

        # ── 工具感知的 LLM 调用 ──
        from .tool_utils import call_llm

        # ── Review-Fix 循环上下文注入 ──
        review_round = task_input.get("_review_round", 0)
        if review_round > 0:
            lc_messages.append(SystemMessage(content=(
                f"注意：这是第 {review_round} 轮复审。内容是经过上一轮审核意见修改后的版本。\n"
                f"请仔细检查修改是否到位。如果内容质量已达标，请明确给出通过结论。\n\n"
                f"【强制输出格式】\n"
                f"**审核结论：** 通过 / 需修改\n"
                f"**主要问题：** （引用原文用「」，限 3 条，无则写'整体质量良好'）\n"
                f"**改进建议：** （限 5 条，无则写'无'）\n"
                f"请在你的回复末尾单独一行输出审核结论：**审核结论：通过** 或 **审核结论：需修改**"
            )))

        llm_result = await call_llm(
            llm, lc_messages,
            agent_type="ReviewAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
        )
        quality_text = llm_result.content

        # 提取 LLM 结论（"**审核结论：通过**" 或 "**审核结论：需修改**"）
        explicit_pass = bool(re.search(
            r"\*\*审核结论[:：]\*\*\s*\[?通过\]?", quality_text
        ))

        # 提取隐藏评分行（格式 "SCORE:85"），用于 pass/fail 判定
        score = 0
        score_match = re.search(r"SCORE\s*[:：]\s*(\d{1,3})", quality_text)
        if score_match:
            score = int(score_match.group(1))

        # 判定标准：总分≥70 或 LLM 显式给出"通过"结论
        passed = explicit_pass or score >= 70

        # 提取改进建议（从 "**改进建议：**" 之后的内容）
        suggestions = []
        in_suggestion_section = False
        for line in quality_text.split("\n"):
            if "**改进建议：**" in line:
                in_suggestion_section = True
                continue
            if in_suggestion_section:
                stripped = line.strip().lstrip("- ")
                if stripped and stripped != "无":
                    suggestions.append(stripped)

        # 展示内容：去掉 SCORE 隐藏行，只留结论+问题+建议
        display_lines = [l for l in quality_text.split("\n") if not re.match(r"^\s*SCORE\s*[:：]\s*\d", l.strip())]
        display_text = "\n".join(display_lines).strip()

        # ── 自动修正 ──
        # 复审时禁用自动修正（由 FixAgent 在循环中处理）
        fixed_content = ""
        rewrite_applied = False
        should_rewrite = task_input.get("_rewrite", True) and review_round == 0
        has_issues = bool(banned_found) or bool(suggestions) or not passed

        if should_rewrite and has_issues:
            try:
                original_for_fix = content or task_input.get("content", "")
                if original_for_fix:
                    banned_words_text = "\n".join(
                        [f"- 「{w.get('word', w)}」→ 建议替换为「{w.get('suggestion', '')}」"
                         for w in (banned_found or [])]
                    ) if banned_found else "无"

                    suggestions_text = "\n".join(
                        [f"- {s}" for s in (suggestions or [])]
                    ) if suggestions else "无"

                    fc_issues = []
                    if fc_report and fc_report.get("claims"):
                        fc_issues = [
                            c for c in fc_report["claims"]
                            if c.get("verdict") in ("false", "unsupported")
                        ]
                    factcheck_text = "\n".join(
                        [f"- {c.get('claim', '')}（{c.get('verdict', '')}）" for c in fc_issues]
                    ) if fc_issues else "无"

                    fix_messages = [
                        SystemMessage(content=_FIX_SYSTEM_PROMPT),
                        HumanMessage(content=_FIX_USER_PROMPT.format(
                            original_content=original_for_fix[:5000],
                            banned_words_text=banned_words_text,
                            suggestions_text=suggestions_text,
                            factcheck_text=factcheck_text,
                        )),
                    ]
                    fix_result = await call_llm(
                        llm, fix_messages,
                        agent_type="ReviewAgent",
                        stream=False,
                    )
                    fixed_content = fix_result.content.strip()
                    rewrite_applied = bool(fixed_content) and fixed_content != original_for_fix[:5000]
            except Exception as fix_err:
                logger.warning("ReviewAgent 自动修正失败，跳过修正: %s", fix_err)

        # 汇总 token（审核 + 修正）
        total_tokens = llm_result.total_tokens + (fix_result.total_tokens if rewrite_applied else 0)
        prompt_tokens = llm_result.prompt_tokens + (fix_result.prompt_tokens if rewrite_applied else 0)
        completion_tokens = llm_result.completion_tokens + (fix_result.completion_tokens if rewrite_applied else 0)

        thinking_summary = ("✅ 通过" if passed else "❌ 需修改") + f" · 违禁词 {len(banned_found)} 个"
        if fc_report:
            thinking_summary += f" · 事实核查 {fc_report.get('total_claims_checked', 0)} 处"
        if rewrite_applied:
            thinking_summary += " · 已自动修正 ✨"

        return agent_response(
            "ReviewAgent",
            content=display_text,
            output_type="text",
            tokens_used=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            output_extra={
                "passed": passed,
                "score": score,
                "banned_words": banned_found,
                "factcheck": fc_report,
                "suggestions": suggestions[:5],
                "clean_content": display_text,
                "fixed_content": fixed_content if rewrite_applied else "",
                "rewrite_applied": rewrite_applied,
                "platform": platform,
            },
            output_fields={
                "review_text": display_text,
                "passed": passed,
                "suggestions": suggestions[:5],
            },
            prompt_versions=composed.used_templates,
            thinking_steps=[{
                "agent": "ReviewAgent",
                "action": "审核完成" + (" + 自动修正" if rewrite_applied else ""),
                "reasoning": f"审核{content_type}内容质量",
                "output_summary": thinking_summary,
                "status": "done",
            }],
        )
    except Exception as e:
        logger.error("ReviewAgent 执行失败: %s", e, exc_info=True)
        raise HandlerError(str(e), handler="ReviewAgent") from e
