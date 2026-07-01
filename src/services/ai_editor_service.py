"""AI Editor Service — 提供文章改写、扩写、翻译、SEO 等能力"""

from __future__ import annotations

import json
import re
import time
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI

from src.agent.platform_profile import get_profile, format_writing_rules
from src.config import get_settings
from src.core.model_router import ModelRouter
from src.logging_shared.config import get_logger

logger = get_logger("service.ai_editor")


class AIEditorService:
    def __init__(self, router: ModelRouter | None = None):
        self._router = router or ModelRouter()
        self._llm: ChatOpenAI = self._router.get_chat_model("AIEditorService")
        self._mock_mode = False

    # ── 正文操作共用约束 ─────────────────────────────────────
    BODY_ONLY_INSTRUCTION = (
        "注意：标题已独立管理，不要生成或修改标题，不要使用 # 号开头的标题。"
        "只处理正文内容，直接输出正文即可。"
    )

    @staticmethod
    def _strip_title(text: str) -> str:
        """移除 AI 生成的正文中可能混入的 markdown 标题安全网"""
        lines = text.split("\n")
        # 去掉开头的 # 标题行（最多连续 3 行）
        while lines and re.match(r"^#+\s", lines[0]):
            lines.pop(0)
            # 跳过标题后的空行
            while lines and not lines[0].strip():
                lines.pop(0)
        # 去掉结尾的 # 标题行
        while lines and re.match(r"^#+\s", lines[-1]):
            lines.pop()
        return "\n".join(lines).strip()

    def _is_mock_mode(self) -> bool:
        try:
            self._llm.invoke([{"role": "user", "content": "test"}])
            return False
        except Exception:
            self._mock_mode = True
            return True

    async def _call(self, prompt: str) -> dict:
        """调用 LLM 并记录日志"""
        t0 = time.time()
        llm_result = await self._llm.ainvoke([{"role": "user", "content": prompt}])
        elapsed = time.time() - t0
        tokens = llm_result.response_metadata.get("token_usage", {})
        logger.info("ai_editor_llm_call",
                    action="llm_call",
                    prompt_length=len(prompt),
                    response_length=len(llm_result.content),
                    elapsed=round(elapsed, 3),
                    input_tokens=tokens.get("prompt_tokens", 0),
                    output_tokens=tokens.get("completion_tokens", 0),
                    total_tokens=tokens.get("total_tokens", 0),
                    )
        return {"content": llm_result.content, "tokens": tokens}

    # ── P1-2: 平台风格拼接到 system 指令，让改写/扩写/翻译/语调都按平台 DNA 来 ──
    @staticmethod
    def _platform_style_block(platform: str | None) -> str:
        """根据 platform code 返回风格约束块（注入到 user prompt 头部）。"""
        if not platform:
            return ""
        profile = get_profile(platform)
        if not profile:
            return ""
        name = profile.get("name", platform)
        system_prompt = profile.get("system_prompt", "")
        rules = profile.get("writing_rules", {})
        rules_text = format_writing_rules(rules) if rules else ""
        # 只取 tone + structure + avoid 三项关键约束，避免过长
        if rules:
            tone = rules.get("tone", "")
            structure_list = rules.get("structure", [])
            avoid_list = rules.get("avoid", [])
            minimal_rules = []
            if tone:
                minimal_rules.append(f"- 语气：{tone}")
            if structure_list:
                minimal_rules.append("- 结构：")
                for i, step in enumerate(structure_list[:4], 1):
                    minimal_rules.append(f"  {i}. {step}")
            if avoid_list:
                minimal_rules.append("- 避免：")
                for item in avoid_list[:4]:
                    minimal_rules.append(f"  · {item}")
            rules_text = "\n".join(minimal_rules)
        parts = [f"【目标平台风格：{name}】"]
        if system_prompt:
            parts.append(f"角色定位：{system_prompt}")
        if rules_text:
            parts.append(f"风格要求：\n{rules_text}")
        return "\n".join(parts) + "\n\n"

    async def rewrite(self, text: str, instruction: str, platform: str | None = None) -> str:
        if self._is_mock_mode():
            return "[Mock] 改写完成"
        style_block = self._platform_style_block(platform)
        prompt = f"""请根据以下要求改写这段文字，只返回改写后的内容，不要添加额外说明。

{self.BODY_ONLY_INSTRUCTION}

{style_block}原文：{text}

改写要求：{instruction}

改写后的内容："""
        result = await self._call(prompt)
        return self._strip_title(result["content"])

    async def expand(self, text: str, platform: str | None = None) -> str:
        if self._is_mock_mode():
            return "[Mock] 扩写完成"
        style_block = self._platform_style_block(platform)
        prompt = f"""请扩写以下内容，要求：

1. 而是在正文各处展开：丰富细节、补充例子、深入论证、增加数据或场景描述
2. 保持原文的结构和风格一致
3. 至少将篇幅扩展到原来的 1.5-2 倍
4. 保留所有原始信息

{self.BODY_ONLY_INSTRUCTION}

{style_block}原文：{text}

扩写后的完整内容（不要加额外说明）："""
        result = await self._call(prompt)
        return self._strip_title(result["content"])

    async def improve_tone(self, text: str, target_tone: str, platform: str | None = None) -> str:
        if self._is_mock_mode():
            return "[Mock] 语调调整完成"
        tone_desc = {
            "professional": "专业严谨，使用行业术语，引用数据支撑",
            "friendly": "亲切友好，像朋友聊天一样自然",
            "humorous": "轻松幽默，适当加入俏皮话和金句",
        }
        desc = tone_desc.get(target_tone, target_tone)
        style_block = self._platform_style_block(platform)
        prompt = f"""请调整以下文字的语调，使其风格为：{desc}。只返回调整后的内容，不要添加额外说明。

{self.BODY_ONLY_INSTRUCTION}

{style_block}原文：{text}

调整后的内容："""
        result = await self._call(prompt)
        return self._strip_title(result["content"])

    async def summarize(self, text: str, platform: str | None = None) -> str:
        if self._is_mock_mode():
            return "[Mock] 这是对原文的简要总结。"
        style_block = self._platform_style_block(platform)
        prompt = f"""请对以下内容进行简洁总结，保留核心观点。控制在 100 字以内。只返回总结内容。

{self.BODY_ONLY_INSTRUCTION}

{style_block}原文：{text}

总结："""
        result = await self._call(prompt)
        return self._strip_title(result["content"])

    async def translate(self, text: str, target_lang: str, platform: str | None = None) -> str:
        if self._is_mock_mode():
            return f"[Mock] 翻译到 {target_lang}"
        lang_name = {"en": "英语", "zh": "中文", "ja": "日语", "ko": "韩语"}
        name = lang_name.get(target_lang, target_lang)
        style_block = self._platform_style_block(platform)
        prompt = f"""请将以下文字翻译成{name}，保持原意和语气。只返回翻译结果。

{self.BODY_ONLY_INSTRUCTION}

{style_block}原文：{text}

翻译："""
        result = await self._call(prompt)
        return self._strip_title(result["content"])

    async def suggest_tags(self, text: str) -> list[str]:
        if self._is_mock_mode():
            return ["mock_tag_1", "mock_tag_2", "mock_tag_3"]
        prompt = f"""根据以下内容，提取 3-5 个最相关的标签。只返回标签列表，用逗号分隔，不要添加额外说明。

内容：{text}

标签："""
        result = await self._call(prompt)
        tags = [t.strip().lstrip("#") for t in result["content"].replace("，", ",").split(",") if t.strip()]
        return tags[:5]

    async def generate_headlines(self, text: str, count: int = 5) -> list[str]:
        if self._is_mock_mode():
            return [f"[Mock] 标题{i}" for i in range(1, min(count, 5) + 1)]
        prompt = f"""根据以下内容，生成 {count} 个吸引人的标题，每个标题控制在 20 字以内。
        要求：有吸引力、覆盖核心内容、适合社交媒体平台。
        只返回标题列表，每行一个，不要编号和额外说明。

        内容：{text}

        标题列表："""
        result = await self._call(prompt)
        headlines = [line.strip().lstrip("0123456789.、. ") for line in result["content"].split("\n") if line.strip()]
        return headlines[:count]

    async def adapt_content(self, text: str, source_platform: str, target_platform: str) -> str:
        if self._is_mock_mode():
            return f"[Mock] 内容从{source_platform}适配到{target_platform}"
        platform_names = {
            "xiaohongshu": "小红书",
            "wechat": "微信公众号",
            "zhihu": "知乎",
        }
        src_name = platform_names.get(source_platform, source_platform)
        tgt_name = platform_names.get(target_platform, target_platform)

        if source_platform == target_platform:
            return text

        prompt = f"""请将以下内容从{src_name}风格适配为{tgt_name}风格。

适配要求：
- 保持核心信息和观点不变
- 调整语气、格式、长度以适配目标平台
- {"小红书：标题带emoji，段落简短，结尾加标签" if target_platform == "xiaohongshu" else ""}
- {"小红书：多用 emoji 和分段，口语化表达" if target_platform == "xiaohongshu" else ""}
- {"微信公众号：正式专业，逻辑清晰，有章节标题" if target_platform == "wechat" else ""}
- {"知乎：第一人称视角，有个人见解，适当引用数据" if target_platform == "zhihu" else ""}
{self.BODY_ONLY_INSTRUCTION}
只返回适配后的内容，不要添加额外说明。

原文：
{text}

适配后的内容："""
        result = await self._call(prompt)
        return self._strip_title(result["content"])

    async def analyze_seo(self, text: str, platform: str = "xiaohongshu") -> dict:
        if self._is_mock_mode():
            return {
                "overall": 60, "title_score": 50, "keyword_score": 50,
                "readability_score": 70, "engagement_score": 50,
                "suggestions": [
                    {"type": "info", "message": "建议增加标题吸引力"},
                    {"type": "info", "message": "建议优化关键词密度"},
                ]
            }
        prompt = f"""作为 SEO 专家，请分析以下内容的质量，给出 0-100 分的评分和优化建议。

        平台：{platform}
        评分维度：
        - title_score: 标题吸引力（0-100）
        - keyword_score: 关键词密度（0-100）
        - readability_score: 可读性（0-100）
        - engagement_score: 互动潜力（0-100）

        请按以下 JSON 格式返回，不要添加额外说明：
        {{
          "overall": <总分>,
          "title_score": <标题分>,
          "keyword_score": <关键词分>,
          "readability_score": <可读性分>,
          "engagement_score": <互动分>,
          "suggestions": [
            {{"type": "warning|error|info", "message": "建议内容"}}
          ]
        }}

        内容：{text}"""

        result = await self._call(prompt)

        try:
            json_match = re.search(r'\{.*\}', result["content"], re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(
                "LLM 响应 JSON 解析失败，返回默认评分",
                error=str(e),
                _type="service",
                event_name="json_parse_error",
            )

        return {
            "overall": 60, "title_score": 50, "keyword_score": 50,
            "readability_score": 70, "engagement_score": 50,
            "suggestions": [
                {"type": "info", "message": "建议增加标题吸引力"},
                {"type": "info", "message": "建议优化关键词密度"},
            ]
        }

    async def expand_keywords(self, seed_keywords: str) -> list[str]:
        if self._is_mock_mode():
            return [f"mock_{seed_keywords}_keyword_{i}" for i in range(1, 6)]
        prompt = f"""作为 SEO 关键词研究专家，请根据以下种子关键词，扩展出 10-15 个长尾关键词。

要求：
- 长尾关键词应包含 3-5 个词
- 覆盖搜索意图（信息类、购买类、疑问类）
- 与种子关键词高度相关
- 按搜索热度从高到低排列

种子关键词：{seed_keywords}

只返回关键词列表，每行一个，不要编号和额外说明："""
        result = await self._call(prompt)
        keywords = [line.strip().lstrip("0123456789.、. -") for line in result["content"].split("\n") if line.strip() and len(line.strip()) > 2]
        return keywords[:15]

    async def analyze_viral_potential(self, text: str) -> dict:
        if self._is_mock_mode():
            return {
                "overall": 65, "topic_heat": 60, "headline_appeal": 60,
                "content_quality": 70, "engagement_potential": 60, "share_potential": 60,
                "strengths": ["内容完整，逻辑清晰"],
                "weaknesses": ["标题吸引力有提升空间"],
                "tips": ["建议优化标题，增加互动引导"],
            }
        prompt = f"""作为社交媒体爆款内容分析师，请分析以下内容的爆款潜质。

评分维度（每项 0-100）：
- topic_heat: 话题热度潜力 — 是否当下热门话题
- headline_appeal: 标题吸引力 — 标题是否足够吸引点击
- content_quality: 内容质量 — 信息密度、实用价值、逻辑性
- engagement_potential: 互动潜力 — 是否容易引发点赞/评论/收藏
- share_potential: 传播潜力 — 是否容易被转发分享

请按以下 JSON 格式返回，不要添加额外说明：
{{
  "overall": <综合爆款指数>,
  "topic_heat": <话题热度>,
  "headline_appeal": <标题吸引力>,
  "content_quality": <内容质量>,
  "engagement_potential": <互动潜力>,
  "share_potential": <传播潜力>,
  "strengths": ["优势1", "优势2", ...],
  "weaknesses": ["不足1", "不足2", ...],
  "tips": ["优化建议1", "优化建议2", ...]
}}

内容：{text}"""

        result = await self._call(prompt)

        try:
            json_match = re.search(r'\{.*\}', result["content"], re.DOTALL)
            if json_match:
                import json
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(
                "LLM 响应 JSON 解析失败，返回默认分析结果",
                error=str(e),
                _type="service",
                event_name="json_parse_error",
            )

        return {
            "overall": 65, "topic_heat": 60, "headline_appeal": 60,
            "content_quality": 70, "engagement_potential": 60, "share_potential": 60,
            "strengths": ["内容完整，逻辑清晰"],
            "weaknesses": ["标题吸引力有提升空间"],
            "tips": ["建议优化标题，增加互动引导"],
        }

    async def check_banned_words_ai(
        self, text: str, platform: str = "xiaohongshu"
    ) -> dict:
        """AI 检测平台违禁内容（含上下文理解）"""
        if self._is_mock_mode():
            return {"words": [], "total": 0}

        PLATFORM_RULES = {
            "xiaohongshu": "小红书社区规范：禁止医疗广告/代购/刷单/虚假宣传（如'最赚钱'等承诺收益表述）；禁止导流（微信号/QQ群）；禁止政治敏感/色情/赌博内容；禁止封建迷信。",
            "wechat": "微信公众号规范：禁止政治敏感内容（六四/学运/法轮功等）；禁止色情/赌博/暴恐；禁止诱导分享/传销/集资；禁止荐股/内幕消息。",
            "zhihu": "知乎社区规范：禁止政治敏感内容；禁止色情/赌博；禁止代购/刷单/传销；禁止导流行为。",
            "douyin": "抖音社区规范：禁止政治敏感/色情/暴力内容；禁止虚假宣传/夸大功效；禁止导流行为；禁止未成年不当内容。",
            "bilibili": "B站社区规范：禁止政治敏感/色情/暴力内容；禁止抄袭/侵权；禁止恶意引战/人身攻击。",
        }

        prompt = f"""你是一个内容安全审查专家。请检查以下内容是否违反平台发布规范。

{PLATFORM_RULES.get(platform, PLATFORM_RULES['xiaohongshu'])}

请逐条检查并返回所有违规项。对于每个违规项：
1. word: 违规的具体文本片段（原样摘取）
2. reason: 违反了什么规则
3. suggestion: 建议的修改方案
4. severity: "high" | "medium" | "low"

如果完全合规，返回空列表。

注意：
- 不要只做关键词匹配，要理解上下文和语义
- 夸张的营销用语（如"最好"、"第一"）也属于违规
- 隐晦表达、谐音、变体也需识别

请以 JSON 格式返回（不要添加额外说明）：
{{"words": [{{"word": "...", "reason": "...", "suggestion": "...", "severity": "..."}}]}}

内容：
{text}"""

        try:
            result = await self._call(prompt)
            json_match = re.search(r'\{.*\}', result["content"], re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                words = data.get("words", [])
                return {"words": words, "total": len(words)}
        except Exception:
            pass

        return {"words": [], "total": 0}


ai_editor_service = AIEditorService()
