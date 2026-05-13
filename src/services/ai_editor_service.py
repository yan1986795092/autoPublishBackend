from src.core.llm_gateway import LLMGateway
from src.config import get_settings


class AIEditorService:
    def __init__(self):
        self._gateway: LLMGateway | None = None

    def _get_gateway(self) -> LLMGateway:
        if not self._gateway:
            settings = get_settings()
            default_provider = settings.default_llm_provider
            api_key = getattr(settings, f"{default_provider}_api_key", "")
            if not api_key:
                for provider in ["openai", "zhipu", "qwen"]:
                    api_key = getattr(settings, f"{provider}_api_key", "")
                    if api_key:
                        default_provider = provider
                        break
            self._gateway = LLMGateway(api_key=api_key, provider=default_provider)
        return self._gateway

    async def rewrite(self, text: str, instruction: str) -> str:
        prompt = f"""请根据以下要求改写这段文字，只返回改写后的内容，不要添加额外说明。

原文：{text}

改写要求：{instruction}

改写后的内容："""
        return await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )

    async def expand(self, text: str) -> str:
        prompt = f"""请扩写以下段落，丰富细节和论证，保持风格一致。只返回扩写后的内容。

原文：{text}

扩写后的内容："""
        return await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000,
        )

    async def improve_tone(self, text: str, target_tone: str) -> str:
        tone_desc = {
            "professional": "专业严谨，使用行业术语，引用数据支撑",
            "friendly": "亲切友好，像朋友聊天一样自然",
            "humorous": "轻松幽默，适当加入俏皮话和金句",
        }
        desc = tone_desc.get(target_tone, target_tone)
        prompt = f"""请调整以下文字的语调，使其风格为：{desc}。只返回调整后的内容，不要添加额外说明。

原文：{text}

调整后的内容："""
        return await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )

    async def summarize(self, text: str) -> str:
        prompt = f"""请对以下内容进行简洁总结，保留核心观点。控制在 100 字以内。只返回总结内容。

原文：{text}

总结："""
        return await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500,
        )

    async def translate(self, text: str, target_lang: str) -> str:
        lang_name = {"en": "英语", "zh": "中文", "ja": "日语", "ko": "韩语"}
        name = lang_name.get(target_lang, target_lang)
        prompt = f"""请将以下文字翻译成{name}，保持原意和语气。只返回翻译结果。

	原文：{text}

	翻译："""
        return await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

    async def suggest_tags(self, text: str) -> list[str]:
        prompt = f"""根据以下内容，提取 3-5 个最相关的标签。只返回标签列表，用逗号分隔，不要添加额外说明。

	内容：{text}

	标签："""
        result = await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        tags = [t.strip().lstrip("#") for t in result.replace("，", ",").split(",") if t.strip()]
        return tags[:5]

    async def generate_headlines(self, text: str, count: int = 5) -> list[str]:
        prompt = f"""根据以下内容，生成 {count} 个吸引人的标题，每个标题控制在 20 字以内。
    要求：有吸引力、覆盖核心内容、适合社交媒体平台。
    只返回标题列表，每行一个，不要编号和额外说明。

    内容：{text}

    标题列表："""
        result = await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300,
        )
        headlines = [line.strip().lstrip("0123456789.、. ") for line in result.split("\n") if line.strip()]
        return headlines[:count]

    async def adapt_content(self, text: str, source_platform: str, target_platform: str) -> str:
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
只返回适配后的内容，不要添加额外说明。

原文：
{text}

适配后的内容："""
        return await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=3000,
        )

    async def analyze_seo(self, text: str, platform: str = "xiaohongshu") -> dict:
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
        import json
        import re

        result = await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback score
        word_count = len(text)
        return {
            "overall": 60,
            "title_score": 50,
            "keyword_score": 50,
            "readability_score": 70,
            "engagement_score": 50,
            "suggestions": [
                {"type": "info", "message": "建议增加标题吸引力"},
                {"type": "info", "message": "建议优化关键词密度"},
            ]
        }


    async def expand_keywords(self, seed_keywords: str) -> list[str]:
        prompt = f"""作为 SEO 关键词研究专家，请根据以下种子关键词，扩展出 10-15 个长尾关键词。

要求：
- 长尾关键词应包含 3-5 个词
- 覆盖搜索意图（信息类、购买类、疑问类）
- 与种子关键词高度相关
- 按搜索热度从高到低排列

种子关键词：{seed_keywords}

只返回关键词列表，每行一个，不要编号和额外说明："""
        result = await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=400,
        )
        keywords = [line.strip().lstrip("0123456789.、. -") for line in result.split("\n") if line.strip() and len(line.strip()) > 2]
        return keywords[:15]

    async def analyze_viral_potential(self, text: str) -> dict:
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
        import json
        import re

        result = await self._get_gateway().generate_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {
            "overall": 65,
            "topic_heat": 60,
            "headline_appeal": 60,
            "content_quality": 70,
            "engagement_potential": 60,
            "share_potential": 60,
            "strengths": ["内容完整，逻辑清晰"],
            "weaknesses": ["标题吸引力有提升空间"],
            "tips": ["建议优化标题，增加互动引导"],
        }


ai_editor_service = AIEditorService()
