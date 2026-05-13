from typing import Any, Optional
from jinja2 import Environment, BaseLoader, TemplateSyntaxError


class PromptEngine:
    def __init__(self):
        self.env = Environment(loader=BaseLoader())

    def render(self, template_content: str, variables: dict[str, Any]) -> str:
        processed = self._preprocess_variables(variables)
        safe_vars = self._sanitize_variables(processed)
        try:
            template = self.env.from_string(template_content)
            return template.render(**safe_vars)
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error: {e}")

    def _preprocess_variables(self, variables: dict) -> dict:
        processed = variables.copy()
        length_map = {"short": 300, "medium": 600, "long": 1000}
        if "length" in processed:
            processed["length_words"] = length_map.get(processed["length"], 600)
        platform_names = {"xiaohongshu": "小红书", "wechat": "微信公众号", "zhihu": "知乎"}
        if "platform_code" in processed:
            processed["platform_name"] = platform_names.get(processed["platform_code"], processed["platform_code"])
        style_map = {"friendly": "亲切友好，像朋友聊天", "professional": "专业严谨，引用数据", "humorous": "轻松幽默，金句频出"}
        if "tone" in processed:
            processed["style"] = style_map.get(processed["tone"], processed["tone"])

        # --- 新增参数映射 ---
        audience_map = {"beginner": "面向初学者，通俗易懂", "intermediate": "面向有一定基础的用户，深度适中", "expert": "面向专家，深入专业"}
        if "target_audience" in processed:
            processed["target_audience_desc"] = audience_map.get(processed["target_audience"], processed["target_audience"])

        if "keywords" in processed and isinstance(processed["keywords"], str) and processed["keywords"].strip():
            processed["keywords_desc"] = f"文中自然融入以下关键词：{processed['keywords']}"

        ref_style_map = {
            "informative": "信息密度高，干货为主，注重实用价值",
            "storytelling": "故事叙述风格，以真实案例或故事引出观点",
            "case-study": "案例分析风格，深入剖析具体案例",
            "guide": "教程指南风格，步骤清晰，易于跟随",
            # 小红书专有风格
            "collection": "合集分享风格，以「合集」形式罗列推荐，如好物合集、穿搭合集、书单合集，每项附简短理由",
            "immersive": "沉浸式体验风格，以第一视角还原真实体验过程，注重细节描述和感官体验，让读者身临其境",
            "advice-seeking": "听劝互动风格，以「听劝」「求建议」开头，营造向读者求助的互动氛围，结尾引导评论建议",
            "challenge": "挑战记录风格，以「挑战XX天/件事」为框架，记录过程和成果，突出坚持和变化",
        }
        if "reference_style" in processed:
            processed["reference_style_desc"] = ref_style_map.get(processed["reference_style"], processed["reference_style"])

        struct_map = {
            "intro-body-conclusion": "引言-正文-结论的经典结构，开头吸引注意，中间展开论述，结尾总结升华",
            "problem-solution": "问题-解决方案结构，先指出痛点再给出解决方法",
            "listicle": "清单体结构，分点列举，条理清晰",
            "flexible": "灵活结构，根据内容自然展开，不受固定框架限制",
        }
        if "content_structure" in processed:
            processed["content_structure_desc"] = struct_map.get(processed["content_structure"], processed["content_structure"])

        cta_map = {
            "ask-follow": "引导读者关注账号，如「关注我获取更多干货」",
            "ask-share": "鼓励读者分享，如「觉得有用就分享给朋友吧」",
            "ask-save": "引导读者收藏，如「先收藏起来慢慢看」",
            "none": "不强引导，自然结尾",
        }
        if "call_to_action" in processed:
            processed["call_to_action_desc"] = cta_map.get(processed["call_to_action"], processed["call_to_action"])

        lang_map = {"formal": "书面语，正式严谨", "neutral": "中性表达，自然流畅", "casual": "口语化，轻松随意"}
        if "language_style" in processed:
            processed["language_style_desc"] = lang_map.get(processed["language_style"], processed["language_style"])

        pov_map = {"first-person": "第一人称视角，以「我」的亲身经历展开", "third-person": "第三人称视角，客观叙述", "brand": "品牌视角，以「我们」代表品牌发声"}
        if "point_of_view" in processed:
            processed["point_of_view_desc"] = pov_map.get(processed["point_of_view"], processed["point_of_view"])

        if "brand_voice_name" in processed and isinstance(processed["brand_voice_name"], str) and processed["brand_voice_name"].strip():
            processed["brand_voice_desc"] = f"使用品牌语调「{processed['brand_voice_name']}」贯穿全文"

        return processed

    def _sanitize_variables(self, variables: dict) -> dict:
        sanitized = {}
        for key, value in variables.items():
            if isinstance(value, str):
                sanitized[key] = value.replace("{{", "\\{\\{").replace("}}", "\\}\\}")
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_variables({"_": v})["_"] for v in value]
            else:
                sanitized[key] = value
        return sanitized

    def build_full_prompt(self, system_template: Optional[str], user_template: str, variables: dict) -> list[dict]:
        user_content = self.render(user_template, variables)
        messages = []
        if system_template:
            system_content = self.render(system_template, variables)
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        return messages


prompt_engine = PromptEngine()
