"""SEOAgent — 搜索优化 + 平台内容改写

两种工作模式：
1. rewrite=False（分析模式）：关键词研究 + SEO 评分（给 ReviewAgent 参考，不进入最终输出）
2. rewrite=True（改写模式）：基于 WritingAgent 的内容，改写为 SEO 友好版本（作为最终内容输出）

SEOAgent 不应向用户暴露内部分析过程，只输出对用户有价值的成果：
  - 分析模式：metadata（keywords, score 等）附加在 ReviewAgent 的上下文中
  - 改写模式：输出改写后的正文内容

所有平台配置统一从 platform_registry.py 读取。
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .tool_utils import call_llm
from ..agent_utils import extract_output, seo_score, seo_suggestions
from .keyword_tool import extract_keywords, suggest_related_keywords, keyword_density_analysis
from src.core.exceptions import HandlerError
from src.prompts import PromptBuilder
from src.logging_shared.config import get_logger

logger = get_logger("agent.seo")

_REWRITE_SYSTEM_PROMPT = """你是一位资深内容优化专家，精通各平台内容营销。

你的职责是根据 SEO 分析结果，对原始内容进行优化改写，使内容：
1. 自然融入目标关键词（不堆砌，密度 1-3%）
2. 标题更有吸引力、更符合平台用户搜索习惯
3. 正文结构更清晰，段落更易读
4. 保留原文的核心信息和口语化风格（不要变成论文）

重要原则：
- 改写而不是重写：保留原文的核心观点和个人风格
- 关键词自然融入：不要生硬插入，要像正常写作一样
- 平台适配：根据目标平台调整表达方式"""

# ── 平台专属改写 Prompt（对应 platform_registry.PLATFORMS 的 rewrite_prompt_key）──
_REWRITE_PROMPTS: dict[str, str] = {
    "xiaohongshu": """## SEO 改写任务

**目标平台：** 小红书
**写作主题：** {topic}

**关键词（自然融入）：**
{keywords}

**待改写内容：**
{original_content}

请直接输出改写后的内容，保持小红书的风格：
- 口语化、真实感强
- 短段落、多换行
- 适当使用 emoji 增加亲和力
- 标题要有吸引力（带 emoji，20字以内）
- 避免硬性 SEO 痕迹，让关键词自然融入
- 结尾加 3-5 个相关标签

直接输出正文，不要加「以下是改写内容」等前缀。""",

    "wechat": """## SEO 改写任务

**目标平台：** 微信公众号
**写作主题：** {topic}

**关键词（自然融入）：**
{keywords}

**待改写内容：**
{original_content}

请对以上内容进行 SEO 优化改写：
- 保持公众号深度文章的文风，温暖有见地
- 层层递进的逻辑结构
- 小标题分段，结构清晰
- 结尾有明确的行动号召或思考引导
- 自然融入关键词，不要堆砌
- 适合微信生态的标题风格

直接输出改写后的正文，不要加前缀。""",

    "zhihu": """## SEO 改写任务

**目标平台：** 知乎
**写作主题：** {topic}

**关键词（自然融入）：**
{keywords}

**待改写内容：**
{original_content}

请对以上内容进行 SEO 优化改写：
- 保持知乎专业但不平庸的调性
- 逻辑严谨、数据说话、有独特见解
- 适当引用数据和研究增加说服力
- 结构清晰，段落层次分明
- 自然融入关键词
- 适合知乎的回答/文章格式

直接输出改写后的正文，不要加前缀。""",

    "copywriting": """## SEO 改写任务

**目标平台：** 营销文案
**写作主题：** {topic}

**关键词（自然融入）：**
{keywords}

**待改写内容：**
{original_content}

请对以上内容进行营销文案优化：
- 直击痛点，激发需求
- 价值主张清晰，卖点突出
- 行动号召有力
- 语言简练有力，避免废话
- 自然融入关键词，不要生硬

直接输出改写后的文案，不要加前缀。""",

    "newsletter": """## SEO 改写任务

**目标平台：** 邮件通讯（Newsletter）
**写作主题：** {topic}

**关键词（自然融入）：**
{keywords}

**待改写内容：**
{original_content}

请对以上内容进行 Newsletter 风格优化：
- 开头抓人，让读者想继续看
- 内容有价值，提供独特见解
- 语气亲切，像朋友间的交流
- 结尾有明确的行动指引
- 自然融入关键词

直接输出改写后的内容，不要加前缀。""",
}


def _get_rewrite_prompt(platform: str) -> str:
    """根据平台获取对应的改写 prompt"""
    return _REWRITE_PROMPTS.get(
        platform,
        _REWRITE_PROMPTS.get(
            "xiaohongshu",
            """## SEO 改写任务

**主题：** {topic}
**关键词：** {keywords}

**待改写内容：**
{original_content}

请对以上内容进行 SEO 优化改写，保留核心信息和风格，自然融入关键词。
直接输出改写后的正文内容。""",
        ),
    )


async def handle_seo(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    """SEO Agent 主入口，根据 rewrite 参数决定模式"""
    rewrite = task_input.get("rewrite", False)
    platform = task_input.get("platform", "web")

    if rewrite:
        return await _handle_seo_rewrite(task_input, llm, mcp_tools, platform)
    else:
        return await _handle_seo_analyze(task_input, llm, mcp_tools)


# ── 改写模式（rewrite=True）────────────────────────────────────────────

async def _handle_seo_rewrite(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None, platform: str) -> dict:
    """改写模式：将 WritingAgent 的内容优化后输出作为最终内容"""
    topic = task_input.get("topic", "")
    original_content = _extract_content(task_input)

    if not original_content:
        return _make_failed_result("SEOAgent", "无内容可改写", topic)

    if llm is None:
        # 无 LLM 时直接返回原始内容（优雅降级）
        return {
            "status": "completed",
            "agent_type": "SEOAgent",
            "output": {
                "content": original_content,
                "type": "text",
                "content_type": f"seo_{platform}",
            },
            "tokens_used": 0,
            "thinking_steps": [{
                "agent": "SEOAgent",
                "action": "改写完成（无LLM，原文保留）",
                "reasoning": f"平台={platform}，内容改写",
                "output_summary": f"保留原文 {len(original_content)} 字",
                "status": "done",
            }],
        }

    try:
        # 生成关键词
        related = suggest_related_keywords(topic, original_content)
        keywords_str = "\n".join([f"- {kw}" for kw in related[:5]])

        # 构建改写 prompt（根据平台选择专属模板）
        prompt_tmpl = _get_rewrite_prompt(platform)
        prompt = prompt_tmpl.format(
            topic=topic,
            keywords=keywords_str,
            original_content=original_content[:3000],
        )

        lc_messages = [
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # 流式回调
        on_chunk = task_input.get("_on_llm_chunk")

        llm_result = await call_llm(
            llm, lc_messages,
            agent_type="SEOAgent",
            stream=True,
            on_chunk=on_chunk,
        )

        content = extract_output(llm_result.content) or llm_result.content.strip()

        # 如果改写结果太短，说明 LLM 可能生成了空内容，用原始内容兜底
        if len(content) < 50:
            logger.warning("SEO 改写结果过短（%d 字符），使用原始内容", len(content))
            content = original_content

        return {
            "status": "completed",
            "agent_type": "SEOAgent",
            "output": {
                "content": content,
                "type": "text",
                "content_type": f"seo_{platform}",
                "keywords": related[:8],
                "meta": {
                    "title": topic[:60],
                    "platform": platform,
                },
            },
            "tokens_used": llm_result.total_tokens,
            "prompt_tokens": llm_result.prompt_tokens,
            "completion_tokens": llm_result.completion_tokens,
            "thinking_steps": [{
                "agent": "SEOAgent",
                "action": "SEO改写完成",
                "reasoning": f"平台={platform}，关键词：{', '.join(related[:3])}",
                "output_summary": f"改写为 SEO 友好版本，约 {len(content)} 字",
                "tokens_used": tokens,
                "status": "done",
            }],
        }

    except Exception as e:
        logger.error("SEO 改写失败: %s", e, exc_info=True)
        # 改写失败时，优雅降级为原文（不暴露错误）
        return {
            "status": "completed",
            "agent_type": "SEOAgent",
            "output": {
                "content": original_content,
                "type": "text",
                "content_type": f"seo_{platform}_fallback",
            },
            "tokens_used": 0,
            "thinking_steps": [{
                "agent": "SEOAgent",
                "action": "改写完成（降级）",
                "reasoning": f"改写失败，保留原文",
                "output_summary": f"保留原文 {len(original_content)} 字",
                "status": "done",
            }],
        }


# ── 分析模式（rewrite=False）───────────────────────────────────────────

async def _handle_seo_analyze(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None) -> dict:
    """分析模式：关键词研究 + SEO 评分（不产正文，进入 ReviewAgent 的上下文）"""
    topic = task_input.get("topic", "")
    content = _extract_content(task_input)

    # 关键词研究
    related = suggest_related_keywords(topic, content) if topic else []
    extracted = extract_keywords(content, 5) if content else []

    keywords = [
        {"keyword": topic, "source": "topic", "density_target": 0.02},
    ]
    for kw in related[1:4]:
        keywords.append({"keyword": kw, "source": "suggestion", "density_target": 0.015})
    for kw in extracted[:3]:
        kw_str = kw["keyword"]
        if kw_str != topic and kw_str not in [k["keyword"] for k in keywords]:
            keywords.append({"keyword": kw_str, "source": "extraction", "density_target": 0.01})

    score = seo_score(content, keywords, topic) if content else None
    suggestions = seo_suggestions(content, keywords, score) if score else []
    density = keyword_density_analysis(content, [k["keyword"] for k in keywords[:5]]) if content and keywords else None

    # 分析模式：content 字段为空字符串，不向用户暴露分析过程
    # 分析结果作为 metadata 传递给 ReviewAgent
    return {
        "status": "completed",
        "agent_type": "SEOAgent",
        "output": {
            "content": "",  # 不产正文，分析结果在 metadata 中
            "type": "metadata",
            "metadata": {
                "keywords": keywords,
                "seo_score": score,
                "density_analysis": density,
                "related_keywords": related[:8],
                "suggestions": suggestions,
                "meta": {
                    "title": topic[:60] if topic else "",
                    "description": (content or f"关于{topic}的深度内容")[:160] if topic else "",
                    "keywords": ", ".join(k["keyword"] for k in keywords[:5]),
                },
            },
        },
        "tokens_used": 0,
        "thinking_steps": [{
            "agent": "SEOAgent",
            "action": "SEO分析完成",
            "reasoning": f"为「{topic}」进行SEO分析",
            "output_summary": f"SEO得分：{score['overall'] if score else 'N/A'}/100 · 关键词 {len(keywords)} 个",
            "status": "done",
        }],
    }


# ── 辅助函数 ──────────────────────────────────────────────────────────

def _extract_content(task_input: dict) -> str:
    """从 task_input 中提取正文内容"""
    content = task_input.get("content", "")
    if content:
        return content

    dep_context = task_input.get("_context", {})
    for cdata in dep_context.values():
        raw = cdata.get("content", "")
        if raw:
            if isinstance(raw, str):
                return raw
            elif isinstance(raw, dict):
                return raw.get("result", "") or raw.get("content", "")
    return ""


def _make_failed_result(agent_type: str, reason: str, topic: str) -> dict:
    """生成一个优雅的失败结果（不暴露内部错误）"""
    return {
        "status": "completed",
        "agent_type": agent_type,
        "output": {
            "content": "",
            "type": "metadata",
            "error": reason,
        },
        "tokens_used": 0,
        "thinking_steps": [{
            "agent": agent_type,
            "action": "完成",
            "reasoning": reason,
            "output_summary": f"主题：{topic}",
            "status": "done",
        }],
    }
