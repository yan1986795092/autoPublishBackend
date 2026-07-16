"""WritingAgent — 多风格内容写作

使用 PromptEngine 构建 prompt，支持记忆上下文注入和工具调用（memory_search 等）。
支持平台深度写作引擎：根据 platform 加载对应写作 DNA。
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import extract_output
from src.prompts import PromptBuilder, CONTENT_TYPE_PROMPTS, STYLE_GUIDES
from src.prompts.builtins.writing import TONE_GUIDES, LENGTH_GUIDES, AUDIENCE_GUIDES
from src.core.exceptions import HandlerError
from .tool_utils import call_llm
from src.logging_shared.config import get_logger
from ..platform_profile import get_profile, format_writing_rules, get_names_map

logger = get_logger("agent.writing")

# ── 创作偏好 → LLM 参数映射 ──────────────────────────────
def _creativity_to_temperature(creativity: float | None) -> float | None:
    """前端 creativity (0~1) 映射为 LLM temperature (0.2~0.9)"""
    if creativity is None:
        return None
    return round(0.2 + creativity * 0.7, 2)


LENGTH_TO_MAX_TOKENS: dict[str, int] = {
    "short": 1200,
    "medium": 2500,
    "long": 5000,
}


async def handle_writing(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    try:
        # ── 输入：只读 Schema 声明的字段（必填字段已由 validate_input 保证存在） ──
        topic: str = task_input["topic"]
        content_type: str = task_input.get("content_type", "blog")
        style: str = task_input.get("style", "")
        platform: str = task_input.get("platform", "")
        platform_name: str = ""

        # ── 创作偏好参数（从 ConversationEngine trigger 或前端 preferences 传入） ──
        # 映射链路：前端 preferences → trigger fields → task_input
        creativity: float | None = task_input.get("creativity")
        tone: str = task_input.get("tone", "")
        length: str = task_input.get("length", "medium")
        audience: str = task_input.get("audience", "intermediate")

        # consume 字段：由依赖链自动注入
        search_analysis: str = task_input.get("search_analysis", "")

        # ── 平台深度配置加载 ──
        profile = get_profile(platform) if platform else None
        if profile:
            platform_name = profile.get("name", platform)
            content_type = platform

        # ── 构建搜索/上游素材上下文 ──
        context_str_parts = []
        if search_analysis:
            context_str_parts.append(search_analysis[:3000])

        # 记忆上下文（由 nodes.py 统一注入）
        memory_context = task_input.get("_memory_context", "")

        # ── 用 PromptBuilder 构建 ──
        builder = PromptBuilder()

        # system prompt: 优先使用平台定制的 system_prompt
        platform_system_prompt = task_input.get("_platform_system_prompt", "")
        if platform_system_prompt and profile:
            # 使用平台角色定位
            builder.system_text(f"你是一个内容创作助手。\n\n{platform_system_prompt}")
        else:
            builder.system("system.base")

        # ── Skill system prompt injection ──
        skill_system_prompt = task_input.get("_skill_system_prompt", "")
        if skill_system_prompt:
            builder.context(f"\n\n=== 创作指导 (Skill: {task_input.get('_skill_name', 'Custom')}) ===\n{skill_system_prompt}\n===")

        # ── 任务指令 ──
        # 优先使用平台的结构化写作规则
        platform_rules = task_input.get("_platform_writing_rules", {})
        if platform_rules:
            type_guide = format_writing_rules(platform_rules)
        else:
            type_guide = CONTENT_TYPE_PROMPTS.get(content_type, CONTENT_TYPE_PROMPTS["blog"])

        style_guide = STYLE_GUIDES.get(style, "自然流畅，有自己的观点和态度")
        platform_line = f"**目标平台：** {platform_name}" if platform_name else ""

        # ── 创作偏好 → 写作指南（Prompt 渲染用）─────────────────────────────
        tone_guide = TONE_GUIDES.get(tone, TONE_GUIDES.get("friendly", "语气自然流畅，像和朋友聊天一样亲切友好。"))
        length_guide = LENGTH_GUIDES.get(length, LENGTH_GUIDES.get("medium", "适中篇幅，约 1000-2000 字，观点有展开，节奏适中。"))
        audience_guide = AUDIENCE_GUIDES.get(audience, AUDIENCE_GUIDES.get("intermediate", "面向有基础的中级读者，假设读者了解基础知识，重点讲解进阶技巧。"))

        # ── 体裁信息（用于 agent 行为控制）─────────────────────
        genre = task_input.get("_content_genre", "text")
        genre_display = {
            "text": "图文",
            "video": "视频脚本",
            "audio": "音频脚本",
        }.get(genre, "图文")
        genre_line = f"**内容体裁：** {genre_display}"

        # ── 创作偏好 → LLM 参数（temperature / max_tokens）───────────────────
        # creativity 映射 temperature，length 映射 max_tokens
        effective_temperature = _creativity_to_temperature(creativity)
        effective_max_tokens = LENGTH_TO_MAX_TOKENS.get(length)

        builder.instructions("writing.task", {
            "topic": topic,
            "content_type": content_type,
            "style_guide": style_guide,
            "platform_line": platform_line,
            "genre_line": genre_line,
            "type_guide": type_guide,
            "tone_guide": tone_guide,
            "length_guide": length_guide,
            "audience_guide": audience_guide,
        })

        # 注入工具描述
        tool_descriptions = task_input.get("_tool_descriptions", "")
        if tool_descriptions:
            builder.context(f"\n\n可用工具：\n{tool_descriptions}")

        # 引用上下文
        if context_str_parts:
            builder.context(f"\n[已有内容参考]\n{chr(10).join(context_str_parts)}")

        # 记忆上下文
        if memory_context:
            builder.memory(f"\n\n{memory_context}")

        # 创作上下文（CreativeAnchors、反馈、最近内容等）
        extra_context = task_input.get("_extra_context", "")
        if extra_context:
            builder.memory(f"\n\n{extra_context}")

        # ── 构建消息 ──
        composed = builder.build()
        messages = composed.messages

        # 转为 LangChain 消息
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        # ── 工具感知的 LLM 调用 ──
        # 透传创作偏好：creativity → temperature，length → max_tokens
        llm_result = await call_llm(
            llm, lc_messages,
            agent_type="WritingAgent",
            stream=True,
            on_chunk=task_input.get("_on_llm_chunk"),
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )
        raw = llm_result.content

        # 提取正文（extract_output 统一处理多种分隔符）
        content = extract_output(raw)
        if not content:
            content = raw.strip()
            if content:
                logger.warning("extract_output 返回空，使用原始响应", agent="WritingAgent")

        # 后处理：剥离 LLM 自我元描述（## 思考 / ## 正文 / 行内"思考：xxx"等）
        from ..output_filter import strip_meta_dump
        if content:
            _stripped = strip_meta_dump(content)
            if _stripped and len(_stripped.strip()) >= 100:
                content = _stripped

        platform_display = f"（{platform_name}）" if platform_name else ""
        from ..result import agent_response

        return agent_response(
            "WritingAgent",
            content=content,
            output_type="text",
            tokens_used=llm_result.total_tokens,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            output_extra={"content_type": content_type, "platform": platform},
            output_fields={
                "article_body": content,
            },
            prompt_versions=composed.used_templates,
            thinking_steps=[{
                "agent": "WritingAgent", "action": "写作完成",
                "reasoning": f"撰写{content_type}内容{platform_display}：{topic[:30]}...",
                "output_summary": f"生成了约 {len(content)} 字内容",
                "tokens_used": llm_result.total_tokens, "status": "done",
            }],
        )
    except Exception as e:
        logger.error("WritingAgent 执行失败: %s", e, exc_info=True)
        raise HandlerError(str(e), handler="WritingAgent") from e
