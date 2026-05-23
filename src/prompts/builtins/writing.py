"""WritingAgent 模板 — 多风格内容写作"""

from ..engine import PromptTemplateDef
from ..variables import VariableDef

# 导出给 prompts.__init__ 向后兼容
CONTENT_TYPE_PROMPTS: dict[str, str] = {
    "blog":
        "写一篇技术博客文章。要求：有吸引力的标题、引言、2-4个主要章节、结论。使用专业但易懂的语言。",
    "social_post":
        "写一篇社交媒体帖子。要求：吸引眼球的开头、简短有力、有互动引导。",
    "copywriting":
        "写一篇营销文案。要求：抓住痛点的标题、价值主张、行动号召。",
    "script":
        "写一个视频脚本。要求：口播自然、有开场钩子、分镜描述。",
    "newsletter":
        "写一封邮件Newsletter。要求：友好的问候语、2-3个核心内容。",
    "report":
        "写一份分析报告。要求：数据驱动、结构化、有结论和建议。",
    "weibo_post":
        "写一条微博/即刻短帖。要求：140字以内、一针见血、有话题标签、适合快速传播。",
    "twitter_thread":
        "写一条 Twitter/X Thread。要求：英文为主、开头hook吸引点击、每条推文独立可读、有逻辑递进、结尾带CTA和话题标签。",
    "podcast_script":
        "写一期播客脚本。要求：有开场介绍、嘉宾引入（如适用）、2-3个讨论环节、收尾总结和预告。",
}

STYLE_GUIDES: dict[str, str] = {
    "技术": "使用专业术语但解释清楚，代码示例要完整可运行。",
    "种草": "第一人称体验分享，突出真实感和代入感。",
    "干货": "信息密度高，条理清晰，每个观点都有论据支持。",
    "幽默": "轻松诙谐的语气，适当使用比喻和夸张。",
    "深度": "系统全面，有独特见解，引用研究和数据。",
    "教程": "步骤清晰，从零开始，每个步骤都有预期结果说明。",
}

templates = [
    # ── 写作主任务 ──
    PromptTemplateDef(
        name="writing.task",
        template="""## 写作任务

**主题：** {topic}
**内容形式：** {content_type}
{platform_line}
**风格偏好：** {style_guide}

**正文要求：**
{type_guide}

直接开始写吧，标题和正文都要。""",
        description="写作任务指令",
        version="2.0",
        role="WritingAgent",
        group="writing",
        variables=[
            VariableDef("topic", required=True, label="写作主题"),
            VariableDef("content_type", required=False, default="blog", label="内容形式",
                        options=list(CONTENT_TYPE_PROMPTS.keys())),
            VariableDef("style_guide", required=False, default="自然流畅，有自己的观点和态度",
                        label="风格偏好"),
            VariableDef("platform_line", required=False, default="", label="目标平台行"),
            VariableDef("type_guide", required=False, default=CONTENT_TYPE_PROMPTS["blog"],
                        label="内容形式要求"),
        ],
    ),

    # ── 各内容类型的指令（作为独立模板，方便部分覆盖）──
    *[
        PromptTemplateDef(
            name=f"writing.type.{key}",
            template=guide,
            description=f"内容类型指令: {key}",
            role="WritingAgent",
            group="writing.types",
        )
        for key, guide in CONTENT_TYPE_PROMPTS.items()
    ],

    # ── 风格指南（同上）──
    *[
        PromptTemplateDef(
            name=f"writing.style.{key}",
            template=guide,
            description=f"写作风格: {key}",
            role="WritingAgent",
            group="writing.styles",
        )
        for key, guide in STYLE_GUIDES.items()
    ],
]
