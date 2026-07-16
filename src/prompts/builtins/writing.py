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
        "为某平台写一篇图文内容。要求：信息密度高、排版清晰、重点突出。",
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
    # 小红书专项
    "xhs_post":
        "写一篇小红书种草笔记。要求：口语化、真实感强、短段落多换行、适当使用 emoji 增加亲和力。标题要有吸引力，能引发点赞收藏。结尾引导互动，如「大家有什么想法评论区聊聊」「点赞收藏不迷路」等。避免生硬的广告腔，要像朋友推荐一样自然。",
    # 公众号
    "wechat_article":
        "写一篇公众号文章。要求：温暖的文风、深度分析、层层递进。结尾有明确的行动号召或思考引导。适合传播的标题，适当使用小标题分段。",
    # 知乎
    "zhihu_article":
        "写一篇知乎文章/回答。要求：逻辑严谨、数据说话、有独特见解。结构清晰，适当引用数据和研究。用专业但不晦涩的语言，展现思考深度。",
    # 小说创作
    "fiction":
        "写一篇小说。要求：人物形象鲜明，对话自然生动，情节跌宕起伏，节奏把控得当。环境描写细腻，氛围营造到位。有独特的叙事视角和文学质感。",
    # 分镜脚本
    "storyboard":
        "写一份分镜脚本。要求：镜头语言专业，景别标注清晰（远景/中景/近景/特写）。画面描述具体，包含构图、色彩、光线。对白与画面配合精准，时长估算合理。",
    # 技术文档
    "technical_doc":
        "写一份技术文档。要求：结构层次分明，语言准确严谨，术语规范。示例代码/配置完整可用，注重可操作性和实用性。",
    # 教程指南
    "tutorial":
        "写一份教程指南。要求：循序渐进，由浅入深，步骤清晰，每个步骤有明确目标。配合示例，学练结合，提示常见错误和注意事项。",
}

STYLE_GUIDES: dict[str, str] = {
    "技术": "使用专业术语但解释清楚，代码示例要完整可运行。",
    "种草": "第一人称体验分享，突出真实感和代入感。",
    "干货": "信息密度高，条理清晰，每个观点都有论据支持。",
    "幽默": "轻松诙谐的语气，适当使用比喻和夸张。",
    "深度": "系统全面，有独特见解，引用研究和数据。",
    "教程": "步骤清晰，从零开始，每个步骤都有预期结果说明。",
}

# 语调指南：前端"语调"参数 → 写作 LLM 的语气要求
TONE_GUIDES: dict[str, str] = {
    "friendly": "语气友好亲切，像朋友聊天一样自然，避免生硬说教。",
    "professional": "语气专业严谨，措辞正式但不晦涩，适合职场和商务场景。",
    "humorous": "轻松幽默，适度调侃和玩笑，让内容有趣易读。",
    "warm": "温暖柔和，像贴心的长辈或闺蜜，给人被理解和被关怀的感觉。",
    "authoritative": "权威果断，观点鲜明，不模棱两可，给读者明确指引。",
    "casual": "口语化随意，像朋友圈动态或私信一样轻松自在。",
}

# 篇幅指南：前端"篇幅"参数 → 字数要求和内容详略
LENGTH_GUIDES: dict[str, str] = {
    "short": "简洁精炼，约 300-800 字。聚焦一个核心观点，信息密度高，不展开细节，适合快速阅读。",
    "medium": "适中篇幅，约 1000-2000 字。观点有展开，适度举例和解释，节奏适中。",
    "long": "深度长文，约 2500-4000 字。系统全面，论点充分展开，论据丰富，适合深度阅读。",
}

# 目标读者指南：前端"受众"参数 → 内容深度和专业度
AUDIENCE_GUIDES: dict[str, str] = {
    "beginner": "面向零基础初学者，假设读者没有任何前置知识，需要解释每一个概念，步骤要极其详细。",
    "intermediate": "面向有基础的中级读者，假设读者了解基础知识，重点讲解进阶技巧和常见误区。",
    "expert": "面向专业从业者，可以用术语直接交流，重点分享深度洞察、行业趋势和前沿实践。",
}

templates = [
    # ── 写作主任务 ──
    PromptTemplateDef(
        name="writing.task",
        template="""## 写作任务

**主题：** {topic}
**内容形式：** {content_type}
{platform_line}
{genre_line}
**风格偏好：** {style_guide}

**语调要求：** {tone_guide}

**篇幅要求：** {length_guide}

**目标读者：** {audience_guide}

**正文要求：**
{type_guide}

直接输出完整的文章正文，不要加任何 JSON 包装、标题、标签。""",
        description="写作任务指令（纯正文输出）",
        version="5.0",
        role="WritingAgent",
        group="writing",
        variables=[
            VariableDef("topic", required=True, label="写作主题"),
            VariableDef("content_type", required=False, default="blog", label="内容形式",
                        options=list(CONTENT_TYPE_PROMPTS.keys())),
            VariableDef("style_guide", required=False, default="自然流畅，有自己的观点和态度",
                        label="风格偏好"),
            VariableDef("platform_line", required=False, default="", label="目标平台行"),
            VariableDef("genre_line", required=False, default="**内容体裁：** 图文", label="体裁行"),
            VariableDef("type_guide", required=False, default=CONTENT_TYPE_PROMPTS["blog"],
                        label="内容形式要求"),
            VariableDef("tone_guide", required=False,
                        default="语气自然流畅，像和朋友聊天一样亲切友好。",
                        label="语调要求"),
            VariableDef("length_guide", required=False,
                        default="适中篇幅，约 1000-2000 字，观点有展开，节奏适中。",
                        label="篇幅要求"),
            VariableDef("audience_guide", required=False,
                        default="面向有基础的中级读者，假设读者了解基础知识，重点讲解进阶技巧。",
                        label="目标读者"),
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
