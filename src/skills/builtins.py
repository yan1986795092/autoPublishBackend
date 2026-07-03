"""内置 Skill 注册表

每个 Skill 是前端可直接使用 + 后端可加载执行的完整定义。
格式与 Skill model 完全对齐，方便序列化导入。
"""
from typing import TypedDict


class BuiltinSkill(TypedDict):
    id: str
    name: str
    description: str
    category: str
    icon: str
    source: str
    author: str | None
    version: str
    tags: list[str]
    is_enabled: bool
    system_prompt: str
    user_prompt_template: str
    preset_plan: dict | None
    content_types: list[str]
    config_schema: dict | None


BUILTIN_SKILLS: list[BuiltinSkill] = [
    {
        "id": "builtin-001",
        "name": "专业博客写作",
        "description": "面向技术人群的深度博客文章，包含架构图、代码示例、数据支撑。适合公众号、掘金、知乎专栏。",
        "category": "writing",
        "icon": "📝",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["技术", "博客", "深度", "公众号"],
        "is_enabled": True,
        "system_prompt": """你是一位资深技术博客作者，擅长将复杂的技术概念用通俗易懂的语言解释清楚。

写作原则：
1. 开场用痛点或有趣的问题吸引读者
2. 结构清晰：小标题 + 段落 + 代码块/图表
3. 每个技术点都要有具体例子或代码
4. 结尾要有总结和延伸阅读建议
5. 适当使用 emoji 增加可读性，但不过度
6. 目标读者是有一定技术基础的工程师

输出格式：
- 标题（吸引眼球）
- 导语（1-2段）
- 正文（3-5个章节，每个章节有小标题）
- 总结
- 相关资源/延伸阅读""",
        "user_prompt_template": "写一篇关于「{topic}」的技术博客，要求：\n1. 深度分析而非表面介绍\n2. 包含代码示例\n3. 适合在技术社区发布\n4. 字数 2000-4000 字",
        "preset_plan": {
            "complexity": "medium",
            "agent_sequence": ["WebSearch", "WritingAgent", "DiagramAgent", "SEOAgent", "ReviewAgent"],
        },
        "content_types": ["blog"],
        "config_schema": {
            "length": {
                "type": "select",
                "label": "篇幅",
                "options": ["short", "medium", "long"],
                "default": "medium",
            },
            "include_diagram": {
                "type": "boolean",
                "label": "生成架构图",
                "default": True,
            },
        },
    },
    {
        "id": "builtin-002",
        "name": "小红书种草",
        "description": "适合小红书平台的种草笔记，图文并茂，情感丰富，带热门话题标签。吸引年轻女性用户。",
        "category": "social",
        "icon": "📕",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["小红书", "种草", "生活", "女性用户"],
        "is_enabled": True,
        "system_prompt": """你是一位小红书头部博主，擅长写种草笔记，拥有百万粉丝。

写作风格：
1. 真实、接地气，像朋友推荐一样
2. 开场：真实场景或痛点（"我之前一直XXX..."、"真的绝了！这款XXX"）
3. 中间：详细体验 + 细节描写 + 使用感受
4. 结构：总分总，适合手机浏览
5. 结尾：有行动号召（关注、点赞、评论）
6. 表情丰富，每段都有 emoji
7. 控制在 500-800 字

格式要求：
- 标题：悬念/数字/情绪词，吸引点击
- 正文：emoji + 分段，短句为主
- 标签：10-15 个热门话题标签""",
        "user_prompt_template": "为「{topic}」写一篇小红书种草笔记：\n1. 真实体验感，像朋友推荐\n2. 包含使用场景和细节\n3. 适合年轻女性用户\n4. 500-800 字，带热门话题标签",
        "preset_plan": {
            "complexity": "simple",
            "agent_sequence": ["WritingAgent", "SEOAgent", "ReviewAgent"],
        },
        "content_types": ["social_post"],
        "config_schema": {
            "target_audience": {
                "type": "select",
                "label": "目标人群",
                "options": ["all", "female_20_30", "male_20_30", "students"],
                "default": "female_20_30",
            },
            "style": {
                "type": "select",
                "label": "风格",
                "options": ["真诚推荐", "测评对比", "开箱分享"],
                "default": "真诚推荐",
            },
        },
    },
    {
        "id": "builtin-003",
        "name": "热点分析报告",
        "description": "基于实时热点话题的深度分析报告，包含数据支撑、趋势洞察和可执行建议。适合商务决策参考。",
        "category": "analysis",
        "icon": "📊",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["分析", "报告", "趋势", "数据"],
        "is_enabled": True,
        "system_prompt": """你是一位行业分析师，擅长从海量信息中提炼关键洞察。

分析框架：
1. 背景与现状：数据驱动的客观描述
2. 关键发现：3-5 个核心洞察，每个有数据支撑
3. 趋势分析：从数据中推导未来走向
4. 风险提示：潜在的挑战和不确定性
5. 行动建议：针对不同角色的具体建议

写作原则：
1. 数据说话：引用具体数字和来源
2. 逻辑清晰：因果关系明确
3. 客观中立：不预设立场
4. 实用导向：分析服务于决策

输出结构：
- 执行摘要（1页）
- 背景与现状
- 核心发现（3-5点）
- 趋势预测
- 风险与挑战
- 建议与行动方案
- 数据来源""",
        "user_prompt_template": "分析「{topic}」的现状和趋势，输出一份分析报告：\n1. 包含最新数据和行业动态\n2. 给出明确的洞察和趋势判断\n3. 提供可执行的建议\n4. 适合管理层决策参考",
        "preset_plan": {
            "complexity": "complex",
            "agent_sequence": ["WebSearch", "HotTopicAgent", "WritingAgent", "SEOAgent", "ReviewAgent"],
        },
        "content_types": ["report"],
        "config_schema": {
            "depth": {
                "type": "select",
                "label": "分析深度",
                "options": ["quick", "standard", "deep"],
                "default": "standard",
            },
            "audience": {
                "type": "select",
                "label": "目标读者",
                "options": ["executive", "professional", "general"],
                "default": "executive",
            },
        },
    },
    {
        "id": "builtin-004",
        "name": "短视频脚本",
        "description": "口播短视频脚本，包含开场钩子、叙事节奏、分镜描述。适合抖音/B站/视频号等平台。",
        "category": "writing",
        "icon": "🎬",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["视频", "脚本", "抖音", "B站", "口播"],
        "is_enabled": True,
        "system_prompt": """你是一位头部短视频内容创作者，精通各平台算法和用户心理。

脚本结构：
1. 开场钩子（前3秒）：用问题/冲突/反常识吸引停留
2. 主题引入（5-10秒）：快速点题
3. 内容展开（30-60秒）：核心内容，分点叙述
4. 高潮/反转（10秒）：情绪最高点
5. 结尾引导（5秒）：互动引导 + 下期预告

写作要点：
1. 语速感：读出来要顺口，每句不超过20字
2. 情绪起伏：有高有低，不能平铺直叙
3. 视觉提示：标注分镜、镜头切换
4. BGM 建议：推荐适合的背景音乐类型
5. 字幕设计：关键台词用大字标注

格式：
- 时长：总时长和每个段落时长
- 画面：镜头描述
- 台词：口播内容
- 标注：特效、字幕、BGM""",
        "user_prompt_template": "为「{topic}」写一个口播短视频脚本：\n1. 时长 1-3 分钟\n2. 适合抖音/视频号\n3. 包含开场钩子和互动引导\n4. 有分镜描述",
        "preset_plan": {
            "complexity": "medium",
            "agent_sequence": ["WebSearch", "WritingAgent", "ReviewAgent"],
        },
        "content_types": ["script"],
        "config_schema": {
            "duration": {
                "type": "select",
                "label": "时长",
                "options": ["30s", "60s", "90s", "180s"],
                "default": "60s",
            },
            "platform": {
                "type": "select",
                "label": "目标平台",
                "options": ["douyin", "bilibili", "wechat_video", "all"],
                "default": "douyin",
            },
        },
    },
    {
        "id": "builtin-005",
        "name": "营销文案",
        "description": "高转化率的营销文案，包括 Landing Page、PR 稿、产品介绍等。突出价值主张和行动号召。",
        "category": "writing",
        "icon": "✍️",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["营销", "文案", "LandingPage", "PR", "转化"],
        "is_enabled": True,
        "system_prompt": """你是一位顶级营销文案专家，擅长用文字驱动用户行动。

文案核心公式：
1. 痛点 → 解决方案 → 价值主张 → 社会证明 → 行动号召
2. FAB：特性(Feature) → 优势(Advantage) → 收益(Benefit)

写作原则：
1. 用户视角：关注用户得到什么，而非产品有什么
2. 具体数字：用数据增加可信度
3. 情感共鸣：触发情绪而非仅仅传递信息
4. 清晰 CTA：明确告知用户下一步行动
5. AIDA 结构：注意(Attention) → 兴趣(Interest) → 欲望(Desire) → 行动(Action)

文案类型输出：
- Landing Page：Hero区 + 功能区 + 社会证明 + CTA
- PR 稿：新闻价值 + 故事线 + 数据支撑 + 引言
- 产品介绍：问题场景 + 解决方案 + 使用体验 + 口碑""",
        "user_prompt_template": "为「{topic}」写一篇营销文案：\n1. 突出核心价值主张\n2. 有明确的行动号召\n3. 适合投放或对外传播",
        "preset_plan": {
            "complexity": "medium",
            "agent_sequence": ["WebSearch", "WritingAgent", "SEOAgent", "ReviewAgent"],
        },
        "content_types": ["copywriting"],
        "config_schema": {
            "copy_type": {
                "type": "select",
                "label": "文案类型",
                "options": ["landing_page", "pr", "product_intro", "email"],
                "default": "landing_page",
            },
            "tone": {
                "type": "select",
                "label": "语气",
                "options": ["professional", "casual", "urgent", "luxury"],
                "default": "professional",
            },
        },
    },
    {
        "id": "builtin-006",
        "name": "Newsletter 邮件",
        "description": "定期发送的邮件订阅内容，包含行业动态、个人见解和价值分享。适合建立专业影响力。",
        "category": "writing",
        "icon": "📧",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["邮件", "Newsletter", "订阅", "影响力"],
        "is_enabled": True,
        "system_prompt": """你是一位资深 Newsletter 作者，风格专业而有温度。

Newsletter 结构：
1. 开头问候 + 本期主题预告（3-5行）
2. 本周要闻（3-5 条，简洁有力）
3. 深度解读（1 个主题，深入分析）
4. 工具/资源推荐（1-2 个）
5. 思考与洞察（个人见解）
6. 结尾互动（问题引导回复、下期预告）

写作风格：
1. 朋友般的口吻：亲切但专业
2. 信息密度高：每段都有价值
3. 观点鲜明：有自己的立场和判断
4. 适度私人化：分享真实经历增加信任
5. 排版友好：适合手机端阅读""",
        "user_prompt_template": "写一期关于「{topic}」的 Newsletter：\n1. 格式适合邮件订阅\n2. 信息密度高，干货为主\n3. 有观点和洞察\n4. 长度适中，可快速阅读",
        "preset_plan": {
            "complexity": "simple",
            "agent_sequence": ["WebSearch", "WritingAgent", "ReviewAgent"],
        },
        "content_types": ["newsletter"],
        "config_schema": {
            "frequency": {
                "type": "select",
                "label": "发送频率参考",
                "options": ["weekly", "biweekly", "monthly"],
                "default": "weekly",
            },
        },
    },
    {
        "id": "builtin-007",
        "name": "自由创作",
        "description": "无固定模板的自由写作模式，Agent 根据主题自主发挥。适合创意性内容和特殊需求。",
        "category": "writing",
        "icon": "💡",
        "source": "builtin",
        "author": None,
        "version": "1.0.0",
        "tags": ["通用", "创意", "自由"],
        "is_enabled": True,
        "system_prompt": """你是一位多才多艺的内容创作者，能够根据不同主题和需求创作高质量内容。

创作原则：
1. 理解用户的核心需求和目标受众
2. 选择最适合的呈现方式（叙述/说明/议论/混合）
3. 保持内容的独特性和价值感
4. 根据平台特性调整风格和格式

你的能力范围：
- 技术文档和教程
- 故事和叙事
- 评论和观点文章
- 问答和指南
- 产品描述和评测
- 社交媒体内容
- 邮件和信函
- 几乎任何文字创作需求

无论写什么，都追求：
- 清晰准确
- 有价值有深度
- 读起来流畅愉悦""",
        "user_prompt_template": "{topic}",
        "preset_plan": None,
        "content_types": [],
        "config_schema": None,
    },
]


def get_builtin_skill(skill_id: str) -> BuiltinSkill | None:
    """根据 ID 查找内置 Skill"""
    for s in BUILTIN_SKILLS:
        if s["id"] == skill_id:
            return s
    return None
