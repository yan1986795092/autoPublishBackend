from sqlalchemy.orm import Session
from typing import Optional, AsyncGenerator
import json
import time
import asyncio

from src.models import Content, ContentVersion, User, ContentStatus, Persona
from src.schemas.content import GenerateRequest
from src.utils.tokenizer import estimate_tokens
from src.logging_shared.config import get_logger
from src.core.model_router import ModelRouter

logger = get_logger("service.content")


# 预设系统模板
SYSTEM_TEMPLATES = {
    "xiaohongshu": {
        "system": "你是一位专业的小红书内容创作者，擅长撰写知识干货类笔记。你的风格是：\n1. 标题吸引眼球，多用 emoji\n2. 内容实用有料，干货满满\n3. 排版清晰，分段适中\n4. 结尾有行动号召（引导收藏、评论）",
        "user": """请帮我撰写一篇关于【{topic}】的小红书笔记。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 目标平台：小红书

## 输出格式
请按以下格式输出：

### 【标题】（20字以内，带 emoji）

正文内容...（使用 Markdown 格式）

### 推荐标签
#标签1 #标签2 #标签3 #标签4 #标签5

---
提示：开头要用【】标注重要信息，多用 emoji 增加可读性。"""
    },
    "wechat": {
        "system": "你是一位资深的微信公众号内容创作者，擅长撰写深度、有见地的长文。你的风格是：\n1. 逻辑严密，论证充分\n2. 引用数据或案例增加说服力\n3. 语言正式但不晦涩\n4. 结构清晰，有小标题",
        "user": """请帮我撰写一篇关于【{topic}】的微信公众号文章。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 目标平台：微信公众号

## 输出格式
请用 Markdown 格式输出，包含：
1. 一个吸引人的标题
2. 3-5 个小节
3. 每个小节有具体论述
4. 结尾有总结和互动引导"""
    },
    "zhihu": {
        "system": "你是一位知乎上的专业内容创作者，擅长撰写有深度、有见地的回答和文章。你的风格是：\n1. 专业但不装，展示真实经验\n2. 善用数据和案例支撑观点\n3. 逻辑清晰，层次分明\n4. 结尾有独到见解",
        "user": """请帮我撰写一篇关于【{topic}】的知乎回答/文章。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 目标平台：知乎

## 输出格式
请用 Markdown 格式输出，确保内容专业、有深度、有独特见解。"""
    },
    "fiction": {
        "system": "你是一位优秀的小说创作者，擅长各种类型小说的创作。你的风格是：\n1. 人物形象鲜明，对话自然生动\n2. 情节跌宕起伏，节奏把控得当\n3. 环境描写细腻，氛围营造到位\n4. 有独特的叙事视角和文学质感",
        "user": """请创作一篇关于【{topic}】的小说。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 类型：小说创作

## 输出格式
请用 Markdown 格式输出：

### 小说标题

**作者的话：**（简短的创作说明）

**人物介绍：**（主要角色简介）

## 正文

（小说正文，注意分段、对话格式、场景切换）

---
提示：注意人物性格的一致性，对话要符合角色身份，情节推进要有逻辑性。"""
    },
    "storyboard": {
        "system": "你是一位专业的影视分镜师，擅长将文字剧本转化为可视化分镜脚本。你的风格是：\n1. 镜头语言专业，景别标注清晰（远景/中景/近景/特写）\n2. 画面描述具体，包含构图、色彩、光线\n3. 对白与画面配合精准\n4. 时长估算合理，节奏把控到位",
        "user": """请根据【{topic}】创作一份详细的分镜脚本。

## 内容要求
- 风格：{style}
- 预估总时长：{length_words} 字左右的分镜描述
- 类型：分镜脚本

## 输出格式

### 片名/主题

**总时长估算：** XX 分钟

| 场次 | 景别 | 画面描述 | 对白/旁白 | 时长 | 备注 |
|------|------|----------|-----------|------|------|
| 1 | 远景 | ... | ... | 5s | ... |
| 2 | 中景 | ... | ... | 8s | ... |

（表格形式，每行一个镜头）

---
提示：分镜要注重视觉叙事，用画面推进故事，而非依赖对白。"""
    },
    "script": {
        "system": "你是一位专业的剧作家/编剧，擅长各种类型剧本创作。你的风格是：\n1. 场景设置清晰，时间地点明确\n2. 对白自然有力，展现人物性格\n3. 动作描述简洁明了\n4. 遵循标准剧本格式",
        "user": """请创作一部关于【{topic}】的剧本。

## 内容要求
- 风格：{style}
- 预估长度：{length_words} 字
- 类型：剧本创作

## 输出格式

### 剧名

**类型：**
**时长：**
**角色：**

## 第一幕

**场景 1**
[地点] - [时间]

（场景描述）

**角色名**
（动作/表情描述）
对白内容...

**角色名**
对白内容...

---
提示：格式要规范，对白和动作描述要清晰区分。"""
    },
    "copywriting": {
        "system": "你是一位资深的营销文案写手，擅长撰写高转化率的营销内容。你的风格是：\n1. 直击痛点，激发需求\n2. 价值主张清晰，卖点突出\n3. 行动号召有力\n4. 语言简练有力，避免废话",
        "user": """请撰写一份关于【{topic}】的营销文案。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 类型：营销文案

## 输出格式

### 标题（吸引眼球）

**核心卖点：**（1-2句话概括）

**痛点分析：**（目标用户的痛点）

**解决方案：**（产品/服务的价值）

**行动号召：**（引导转化）

---
提示：以用户为中心，突出利益而非功能。"""
    },
    "newsletter": {
        "system": "你是一位优秀的邮件通讯(Newsletter)写手，擅长撰写高打开率和点击率的邮件内容。你的风格是：\n1. 开头抓人，让读者想继续看\n2. 内容有价值，提供独特见解\n3. 语气亲切，像朋友间的交流\n4. 结尾有明确的行动指引",
        "user": """请撰写一份关于【{topic}】的邮件通讯(Newsletter)。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 类型：邮件通讯

## 输出格式

### 主题行（吸引打开）

**亲爱的读者，**

（开头段 - 引人入胜）

## 主要内容
（2-3个核心内容模块）

## 推荐阅读/资源推荐

**结语：**（总结 + 互动引导）

---
提示：邮件要有个人风格，让读者期待你的下一封来信。"""
    },
    "technical_doc": {
        "system": "你是一位专业的技术文档写手，擅长撰写清晰、准确、易读的技术文档。你的风格是：\n1. 结构层次分明，目录清晰\n2. 语言准确严谨，术语规范\n3. 示例代码/配置完整可用\n4. 注重可操作性和实用性",
        "user": """请撰写一份关于【{topic}】的技术文档。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 类型：技术文档

## 输出格式

# 文档标题

## 概述
（背景介绍、适用场景）

## 前置条件
- 条件1
- 条件2

## 详细说明
（分步骤、分模块说明）

## 示例
（代码示例或使用示例）

## 常见问题
（FAQ）

---
提示：技术文档要准确、完整、可验证。"""
    },
    "tutorial": {
        "system": "你是一位经验丰富的教程作者，擅长将复杂知识转化为易于理解和跟随的教程。你的风格是：\n1. 循序渐进，由浅入深\n2. 步骤清晰，每个步骤有明确目标\n3. 配合示例，学练结合\n4. 提示常见错误和注意事项",
        "user": """请撰写一份关于【{topic}】的教程指南。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字
- 类型：教程指南

## 输出格式

# 教程标题

**适合人群：**
**前置知识：**
**你将学到：**

## 第一步：[步骤名称]
（详细说明 + 示例）

## 第二步：[步骤名称]
（详细说明 + 示例）

## 总结
（回顾要点 + 进阶学习建议）

---
提示：教程要让零基础读者也能跟上节奏。"""
    },
}

# 通用回退模板（用于未匹配的平台代码）
DEFAULT_SYSTEM_TEMPLATE = {
    "system": "你是一位专业的内容创作者，擅长根据主题创作高质量内容。请根据用户的要求输出格式规范、内容充实的文章。",
    "user": """请撰写一篇关于【{topic}】的内容。

## 内容要求
- 风格：{style}
- 预估字数：{length_words} 字

## 输出格式
请用 Markdown 格式输出，包含标题和正文内容。内容需结构清晰、逻辑连贯、言之有物。"""
}


# Mock 内容库
MOCK_TITLES = {
    "xiaohongshu": [
        "职场沟通高手都在用的拒绝技巧，学会这3招不再委屈自己",
        "如何优雅地拒绝同事的请求？过来人的5条经验分享",
        "高情商拒绝指南：职场中那些不得不说的小技巧",
        "不是所有人都值得你讨好：职场拒绝的艺术",
        "学会说\"不\"：职场人的必修课",
    ],
    "wechat": [
        "关于「{topic}」的深度分析与思考",
        "如何正确理解「{topic}」：一篇讲透",
        "「{topic}」的完整指南（建议收藏）",
        "深度解析「{topic}」的本质与实践",
        "关于「{topic}」，你需要知道的一切",
    ],
    "zhihu": [
        "如何看待「{topic}」这一现象？",
        "关于「{topic}」，我来分享一些真实经历",
        "深度长文：全面解析「{topic}」",
        "「{topic}」的正确打开方式",
        "那些关于「{topic}」的误解与真相",
    ],
    "generic": [
        "关于「{topic}」的深度分享",
        "「{topic}」完整指南",
        "「{topic}」：从入门到精通",
        "「{topic}」全面解析",
        "「{topic}」的终极指南",
    ]
}

MOCK_BODY_TEMPLATES = {
    "xiaohongshu": """你是不是也有这样的困扰？同事总是把自己的活推给你，做了功劳是别人的，出了问题锅却是你的。

今天分享 **5个高情商拒绝技巧**，帮你摆脱职场老好人的困境。

---

## 1. 先肯定对方，再表达困难

❌ 错误示范："不好意思，我手上工作很多，没空帮你。"

✅ 正确示范："这个想法挺好的！看得出来你花了很多心思。不过我最近手上的项目比较紧，怕耽误你的进度，你觉得找XX来协助会不会更合适？"

**关键点**：把拒绝的理由归结为"为对方考虑"，而不是"我太忙/我不想"。

---

## 2. 给出替代方案

拒绝的同时，提供一个可行的替代方案，会让对方更容易接受。

> 建议找 [相关同事] 咨询一下，他们可能更有经验
> 你可以先查一下 [内部文档]，基本流程都在里面

---

## 3. 设定明确的边界

【重要的事情说三遍】
> 你的时间 = 你的价值
> 你的时间 = 你的价值
> 你的时间 = 你的价值

学会给工作设置优先级，不要让别人随意打断你的节奏。

---

## 4. 用"5秒法则"快速决策

当你犹豫要不要答应的时候，给自己5秒钟时间快速决策：
- 如果5秒内想不出答应的好处 → 拒绝
- 如果5秒内想到了 → 谨慎考虑

---

## 5. 练习拒绝，从小事开始

拒绝是一种可以练习的能力。建议从一些小事情开始：
- 同事让你帮忙带咖啡，你说"今天赶时间，下回请我喝？"
- 临时拉你开会，你说"我有个deadline，能发我会议纪要吗？"

---

💡 **总结**：拒绝不是冷漠，而是对自己和他人时间的尊重。学会优雅地拒绝，你才能把精力放在真正重要的事情上。

#职场 #沟通技巧 #自我提升 #拒绝能力 #职场干货""",

    "wechat": """# {title}

在当今社会，职场沟通已经成为每个职场人必须掌握的核心技能之一。很多人在工作中勤勤恳恳，却因为不懂得如何得体地拒绝而陷入被动。本文将深入探讨这一话题。

## 一、为什么我们害怕拒绝

心理学研究表明，拒绝他人的请求会引发大脑中的"社会排斥反应"，这种感觉与身体疼痛激活的是同一片脑区。这解释了为什么很多人宁可委屈自己，也不愿意说"不"。

然而，长期的过度迁就会带来一系列问题：
- 时间和精力被大量消耗
- 工作质量下降
- 职业发展受阻
- 心理压力增大

## 二、拒绝的艺术：三个核心原则

### 原则一：真诚而不僵硬

拒绝不代表冷漠。真正的拒绝是建立在尊重和诚实的基础上的。当你真诚地表达自己的困难时，对方往往能够理解。

### 原则二：明确而不模糊

很多人在拒绝时喜欢绕弯子，结果反而让对方误解为"还有商量的余地"。正确的做法是清晰、明确地表达你的立场。

### 原则三：完整而不残缺

拒绝一个请求后，最好能提供一个替代方案或后续帮助。这样既表达了你的善意，也维护了你的边界。

## 三、实用话术参考

**场景一：同事希望你帮忙完成他的工作**

> "我很理解这个任务的紧迫性，也很感谢你想到我。不过我目前手头有几个 deadline 都很紧，恐怕没办法全程跟进。你可以先看看 [相关文档]，有具体问题我们再讨论？"

**场景二：领导安排超出你能力范围的任务**

> "感谢您的信任。这个任务对我来说是新的领域，我担心自己做不好影响整体进度。能否安排一个更有经验的同事带我一起做，或者先从小范围试点开始？"

## 四、建立拒绝的信心

1. **从小事开始**：先在小事上练习拒绝，逐步建立信心
2. **记住你的价值**：你的时间和精力是有限的资源，有权合理分配
3. **接受不完美**：不要追求让所有人都满意，那是不可能的

## 结语

学会拒绝，是职场成熟的重要标志。它不是自私，而是对自己和他人负责。当你能够得体地拒绝时，你会发现工作更加高效，人际关系也更加健康。

**愿每个职场人都能从容地说"不"，优雅地守护自己的边界。**""",

    "zhihu": """# 关于职场拒绝，我有一些真实的想法

作为一个在职场摸爬滚打多年的过来人，关于"拒绝"这个话题，我有很多想说的。

## 先说说我自己的经历

刚入职场的头两年，我几乎不会拒绝任何人。同事让我帮忙，我答应；领导临时加任务，我接受；甚至下班后被拉去应酬，我也从不推辞。

结果呢？我的直属领导在年底评估时说："小王工作很努力，但缺乏主见，什么都答应，最后什么都做不精。"

这句话像一盆冷水浇醒了我。

## 拒绝的本质是什么？

很多人以为拒绝就是"得罪人"，其实大错特错。

**拒绝的本质是管理预期和资源。**

当你接受一个超出你能力或时间范围的请求时，你实际上是在做一件对双方都不负责任的事：
- 对对方：你给了一个很低的预期，最终可能失望
- 对自己：你承担了不必要的压力和风险

## 我的拒绝方法论

经过多年实践，我总结出一套"拒绝三部曲"：

### 第一步：停顿
不要立刻回答。说"让我想想"或"我先确认一下我的 schedule"。

### 第二步：评估
问自己三个问题：
1. 我有时间/能力做这件事吗？
2. 做完对我有什么好处？
3. 不做会怎样？

### 第三步：回应
用"yes-and"或"yes-but"的方式回应：
- "Yes, 我很想帮你, BUT 我现在手头有..."
- "Yes, 这个想法很好, AND 我建议你先..."

## 写在最后

职场中，最受欢迎的人不是那个对谁都点头的人，而是那个说到做到、有原则有底线的人。

学会拒绝，你的人生会轻松很多。

---

*以上是我的一些真实经历和思考，希望对你们有帮助。*""",

    "generic": """# {title}

## 关于这个话题

在深入探讨之前，我们不妨先来思考一下这个问题背后的本质。这个话题涉及到多个层面的内容，值得我们从不同角度来分析。

## 核心要点

### 1. 基础知识
首先，我们需要建立对相关概念的基本理解。掌握基础知识是深入探讨的前提。

### 2. 实践方法
了解理论之后，更重要的是如何将其应用到实际中。以下是几个实用的方法和步骤：

- 明确目标，制定计划
- 循序渐进，持之以恒
- 及时总结，持续优化

### 3. 常见误区
在实践中，很多人会陷入以下几个常见的误区：

1. 急于求成，忽视基础
2. 照搬照抄，缺乏变通
3. 半途而废，缺乏坚持

## 经验分享

作为一个长期关注这个领域的人，我想分享一些个人的心得体会。首先，最重要的是保持开放和学习的心态。其次，要善于从实践中总结经验教训。

> 知行合一，方得始终。

## 总结

总的来说，关于这个话题还有很多值得探讨的内容。希望以上分享能对你有所帮助和启发。如果你有更多的想法或经验，欢迎继续交流！

---
*本文为示例内容，仅供展示。*"""
}


# 违禁词替换建议
BANNED_WORDS_SUGGESTIONS: dict[str, str] = {
    "最赚钱": "高收益",
    "月入过万": "稳定收入",
    "暴利": "丰厚回报",
    "代购": "海外购买",
    "刷单": "提升销量",
    "微信号": "联系方式",
    "私聊": "进一步沟通",
    "加我": "联系我",
    "炒股": "投资",
    "荐股": "投资建议",
    "内幕消息": "行业分析",
    "传销": "多级分销",
    "色情": "成人内容",
    "裸聊": "视频聊天",
    "一夜情": "短暂关系",
    "赌博": "博彩",
    "赌场": "娱乐场",
    "算命": "命理分析",
    "改运": "转运",
}

# 平台敏感词库
BANNED_WORDS: dict[str, list[str]] = {
    "xiaohongshu": [
        "最赚钱", "月入过万", "暴利", "刷单", "代购", "代购费",
        "医疗器械", "处方药", "减肥药", "丰胸", "增高",
        "微信号", "QQ群", "私聊", "加我",
        "政治", "习近平", "天安门", "法轮功", "台独", "藏独",
        "赌博", "赌场", "彩票技巧",
        "封建迷信", "算命", "改运",
    ],
    "wechat": [
        "政治", "习近平", "天安门", "法轮功", "台独", "藏独",
        "六四", "学运", "暴恐", "分裂",
        "代购", "刷单", "传销", "集资",
        "炒股", "荐股", "内幕消息",
        "赌博", "赌场",
        "色情", "裸聊", "一夜情",
        "微信号", "QQ群", "私聊", "加我",
    ],
    "zhihu": [
        "政治", "习近平", "天安门", "法轮功", "台独", "藏独",
        "六四", "学运", "暴恐", "分裂",
        "代购", "刷单", "传销",
        "赌博", "赌场",
        "色情", "裸聊",
        "微信号", "QQ群", "加我",
    ],
}


def _render_template(template_text: str, variables: dict) -> str:
    """渲染模板，兼容 {{var}} 和 {var} 两种语法"""
    import re
    converted = re.sub(r'\{\{(\w+)\}\}', r'{\1}', template_text)
    from src.prompts import prompt_engine
    return prompt_engine.render_text(converted, variables)


class ContentService:
    """内容服务"""

    def __init__(self, db: Session, router: ModelRouter | None = None):
        self.db = db
        self._router = router or ModelRouter()

    def _get_template(self, platform_code: str) -> tuple[dict, object | None]:
        """获取模板，优先使用 DB 中的 PromptTemplate，无则回退硬编码"""
        from src.models import PromptTemplate

        db_template = self.db.query(PromptTemplate).filter(
            PromptTemplate.platform_code == platform_code
        ).order_by(PromptTemplate.use_count.desc()).first()

        if db_template and db_template.system_prompt and db_template.user_template:
            return {
                "system": db_template.system_prompt,
                "user": db_template.user_template,
            }, db_template

        return SYSTEM_TEMPLATES.get(platform_code, DEFAULT_SYSTEM_TEMPLATE), None

    def _generate_mock_content(self, topic: str, platform: str) -> dict:
        """生成 Mock 内容"""
        import random

        titles = MOCK_TITLES.get(platform, ["关于「{topic}」的分享"])
        title_template = random.choice(titles)
        title = title_template.format(topic=topic)

        body_template = MOCK_BODY_TEMPLATES.get(platform, MOCK_BODY_TEMPLATES["generic"])
        body = body_template.format(title=title)

        return {"title": title, "content_md": body}

    def _parse_generated_content(self, content: str) -> dict:
        """解析生成的内容，提取标题"""
        lines = content.split("\n")
        title = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title and lines:
            title = lines[0].strip()[:50]
        return {"title": title, "content_md": content}

    def _build_persona_desc(self, persona: Persona) -> str:
        """生成人设描述文本"""
        parts = [f"你当前的人设是：{persona.name}。"]
        if persona.description:
            parts.append(f"角色描述：{persona.description}")
        tone_map = {
            "friendly": "语气亲切友好，像朋友一样自然交流",
            "professional": "语气专业严谨，使用行业术语，数据支撑观点",
            "humorous": "语气轻松幽默，适当使用俏皮话和金句",
        }
        if persona.tone:
            parts.append(f"语调要求：{tone_map.get(persona.tone, persona.tone)}")
        if persona.style_notes:
            parts.append(f"风格指南：{persona.style_notes}")
        return "\n".join(parts)

    def _get_llm_for_request(self, model: str) -> "ChatOpenAI":
        """经 ModelRouter 统一入口获取 LLM（流向 LLM Gateway）"""
        return self._router.get_chat_model_for_model(
            model=model,
            agent_type="ContentService",
        )

    async def generate_content(self, user_id: str, request: GenerateRequest) -> dict:
        """生成内容"""
        from src.config import get_settings

        settings = get_settings()
        start_time = time.time()

        # 获取模板（优先 DB，回退硬编码）
        system_tmpl, db_template = self._get_template(request.platform_code)

        length_pref = request.parameters.get("length", "medium") if request.parameters else "medium"
        length_map = {"short": 500, "medium": 1500, "long": 3000}
        length_words = length_map.get(length_pref, 1500)

        variables = {
            "topic": request.topic,
            "platform_code": request.platform_code,
            "length_words": length_words,
            "creativity": request.parameters.get("creativity", 0.7) if request.parameters else 0.7,
            "style": request.parameters.get("style", "自然流畅") if request.parameters else "自然流畅",
        }

        # 人设查询
        persona_desc = ""
        persona_obj = None
        persona_id = request.parameters.get("persona_id") if request.parameters else None
        if persona_id:
            persona_obj = self.db.query(Persona).filter(
                Persona.id == persona_id
            ).first()
            if persona_obj:
                persona_desc = self._build_persona_desc(persona_obj)
                persona_obj.use_count += 1
                self.db.commit()
        variables["persona_desc"] = persona_desc

        messages = [
            {"role": "system", "content": _render_template(system_tmpl["system"], variables)},
            {"role": "user",   "content": _render_template(system_tmpl["user"],   variables)},
        ]

        # 注入人设到系统消息
        if persona_desc and messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n{persona_desc}"

        # 选择 LLM Provider
        # 优先级：用户指定 model > 智谱 > OpenAI (需配置 API Key)
        api_key = None
        provider = "openai"
        model_name = settings.openai_model

        if request.model:
            # 用户指定了模型，根据模型名推断 provider
            if "glm" in request.model.lower() or "zhipu" in (request.model.lower()):
                api_key = settings.zhipu_api_key
                provider = "zhipu"
                model_name = request.model
            elif "qwen" in request.model.lower():
                api_key = settings.qwen_api_key
                provider = "qwen"
                model_name = request.model
            else:
                api_key = settings.openai_api_key
                provider = "openai"
                model_name = request.model
        elif settings.zhipu_api_key:
            api_key = settings.zhipu_api_key
            provider = "zhipu"
            model_name = settings.zhipu_model
        elif settings.openai_api_key:
            api_key = settings.openai_api_key
            provider = "openai"
            model_name = settings.openai_model

        generation_time_ms = 0
        result = None

        if not api_key:
            # Mock 模式
            result = self._generate_mock_content(request.topic, request.platform_code)
            model_name = "mock"
        else:
            # 真实 API 调用 — 经 ModelRouter 统一入口到 Gateway
            llm = self._get_llm_for_request(model_name)
            temperature = variables["creativity"]
            llm_result = await llm.ainvoke(messages)
            full_content = llm_result.content
            result = self._parse_generated_content(full_content)

        # 保存到数据库
        content = Content(
            user_id=user_id,
            title=result["title"],
            topic=request.topic,
            platform_code=request.platform_code,
            content_md=result["content_md"],
            content_html=result["content_md"].replace("\n", "<br>"),
            status=ContentStatus.DRAFT,
            metadata_json=json.dumps(request.parameters or {}),
        )
        self.db.add(content)
        self.db.commit()
        self.db.refresh(content)

        # 创建第一个版本
        version = ContentVersion(
            content_id=content.id,
            version_number=1,
            content_md=result["content_md"],
            content_html=result["content_md"].replace("\n", "<br>"),
            change_summary="初稿生成",
        )
        self.db.add(version)
        self.db.commit()

        # 更新模板使用次数
        if db_template:
            db_template.use_count += 1
            self.db.commit()

        generation_time_ms = int((time.time() - start_time) * 1000)

        return {
            "content_id": content.id,
            "title": content.title,
            "content_md": content.content_md,
            "content_html": content.content_html,
            "metadata": {
                "model": model_name,
                "tokens_used": estimate_tokens(result["content_md"]),
                "tokens_cost_usd": 0,
                "generation_time_ms": generation_time_ms,
            }
        }

    async def generate_content_stream(
        self, user_id: str, request: GenerateRequest
    ) -> AsyncGenerator[str, None]:
        """SSE 流式生成内容（逐 token 推送）"""
        from src.config import get_settings

        settings = get_settings()
        start_time = time.time()

        # 获取模板（优先 DB，回退硬编码）
        system_tmpl, db_template = self._get_template(request.platform_code)

        length_pref = request.parameters.get("length", "medium") if request.parameters else "medium"
        length_map = {"short": 500, "medium": 1500, "long": 3000}
        length_words = length_map.get(length_pref, 1500)

        variables = {
            "topic": request.topic,
            "platform_code": request.platform_code,
            "length_words": length_words,
            "creativity": request.parameters.get("creativity", 0.7) if request.parameters else 0.7,
            "style": request.parameters.get("style", "自然流畅") if request.parameters else "自然流畅",
        }

        # 人设查询
        persona_desc = ""
        persona_obj = None
        persona_id = request.parameters.get("persona_id") if request.parameters else None
        if persona_id:
            persona_obj = self.db.query(Persona).filter(
                Persona.id == persona_id
            ).first()
            if persona_obj:
                persona_desc = self._build_persona_desc(persona_obj)
                persona_obj.use_count += 1
                self.db.commit()
        variables["persona_desc"] = persona_desc

        messages = [
            {"role": "system", "content": _render_template(system_tmpl["system"], variables)},
            {"role": "user",   "content": _render_template(system_tmpl["user"],   variables)},
        ]

        # 注入人设到系统消息
        if persona_desc and messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n{persona_desc}"

        # 选择 LLM Provider
        api_key = None
        provider = "openai"
        model_name = settings.openai_model

        if request.model:
            if "glm" in request.model.lower() or "zhipu" in (request.model.lower()):
                api_key = settings.zhipu_api_key
                provider = "zhipu"
                model_name = request.model
            elif "qwen" in request.model.lower():
                api_key = settings.qwen_api_key
                provider = "qwen"
                model_name = request.model
            else:
                api_key = settings.openai_api_key
                provider = "openai"
                model_name = request.model
        elif settings.zhipu_api_key:
            api_key = settings.zhipu_api_key
            provider = "zhipu"
            model_name = settings.zhipu_model
        elif settings.openai_api_key:
            api_key = settings.openai_api_key
            provider = "openai"
            model_name = settings.openai_model

        full_content = ""
        title = ""

        try:
            if not api_key:
                # Mock 模式：分块模拟流式
                result = self._generate_mock_content(request.topic, request.platform_code)
                full_content = result["content_md"]
                title = result["title"]
                yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"
                chunk_size = 15
                for i in range(0, len(full_content), chunk_size):
                    chunk = full_content[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.01)
            else:
                llm = self._get_llm_for_request(model_name)
                temperature = variables["creativity"]

                async for event in llm.astream_events(messages, config={"temperature": temperature}):
                    if event["event"] == "chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            full_content += content
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

                parsed = self._parse_generated_content(full_content)
                title = parsed["title"]
                yield f"data: {json.dumps({'type': 'title', 'content': title})}\n\n"

            # 保存到数据库（仅在 session 仍然活跃时）
            if self.db.is_active:
                content = Content(
                    user_id=user_id,
                    title=title,
                    topic=request.topic,
                    platform_code=request.platform_code,
                    content_md=full_content,
                    content_html=full_content.replace("\n", "<br>"),
                    status=ContentStatus.DRAFT,
                    metadata_json=json.dumps(request.parameters or {}),
                )
                self.db.add(content)
                self.db.commit()
                self.db.refresh(content)

                # 创建第一个版本
                version = ContentVersion(
                    content_id=content.id,
                    version_number=1,
                    content_md=full_content,
                    content_html=full_content.replace("\n", "<br>"),
                    change_summary="初稿生成",
                )
                self.db.add(version)
                self.db.commit()

                # 更新模板使用次数
                if db_template:
                    db_template.use_count += 1
                    self.db.commit()

            generation_time_ms = int((time.time() - start_time) * 1000)

            done_data = {
                'type': 'done',
                'content_id': content.id if self.db.is_active else '',
                'title': title,
                'metadata': {
                    'model': model_name,
                    'tokens_used': estimate_tokens(full_content),
                    'tokens_cost_usd': 0,
                    'generation_time_ms': generation_time_ms,
                }
            }
            yield f"data: {json.dumps(done_data)}\n\n"

        except (Exception, asyncio.CancelledError) as e:
            logger.error("流式生成内容异常", error=str(e))
            err_payload = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    def suggest_best_time(self, platform_code: str) -> dict:
        """基于预设规则建议最佳发布时间"""
        platform_times = {
            "xiaohongshu": {
                "time_slots": ["07:00-08:00", "11:30-13:00", "17:30-19:00", "20:00-22:00"],
                "best": "20:00-21:00",
                "weekdays": ["周二", "周四", "周六", "周日"],
                "reason": "小红书用户活跃高峰在晚间和周末，此时发布可获得更多曝光",
            },
            "wechat": {
                "time_slots": ["08:00-09:00", "11:30-12:30", "17:00-18:00", "20:00-22:00"],
                "best": "20:00-21:00",
                "weekdays": ["周三", "周五", "周六"],
                "reason": "微信公众号阅读高峰在晚间和周末早晨，用户习惯在放松时阅读深度内容",
            },
            "zhihu": {
                "time_slots": ["08:00-09:30", "12:00-13:30", "18:00-19:30", "21:00-22:30"],
                "best": "21:00-22:00",
                "weekdays": ["周四", "周五", "周六", "周日"],
                "reason": "知乎用户常在晚间深度阅读，周末下午也是问答互动高峰",
            },
        }
        return platform_times.get(platform_code, platform_times["xiaohongshu"])

    def check_banned_words(self, content: str, platform_code: str) -> list[dict]:
        """检测内容中的敏感词（含替换建议）"""
        words = BANNED_WORDS.get(platform_code, [])
        if not content or not words:
            return []

        content_lower = content.lower()
        found = []
        for word in words:
            idx = content_lower.find(word.lower())
            while idx != -1:
                found.append({
                    "word": word,
                    "position": idx,
                    "platform": platform_code,
                    "suggestion": BANNED_WORDS_SUGGESTIONS.get(word, ""),
                })
                idx = content_lower.find(word.lower(), idx + 1)

        return found

    async def batch_generate(self, user_id: str, platforms: list[str], topic: str, parameters: dict | None = None, model: str | None = None) -> list[dict]:
        """批量生成多平台内容（每个平台使用独立 DB session，避免并发竞态）"""
        from src.database import SessionLocal
        from src.schemas.content import GenerateRequest

        async def _gen(platform: str) -> dict:
            db = SessionLocal()
            try:
                svc = ContentService(db)
                req = GenerateRequest(
                    topic=topic,
                    platform_code=platform,
                    model=model,
                    parameters=parameters,
                )
                return await svc.generate_content(user_id, req)
            finally:
                db.close()

        tasks = [_gen(p) for p in platforms]
        results = await asyncio.gather(*tasks)
        return list(results)

    def diff_versions(self, content_id: str, version_id_1: str, version_id_2: str) -> dict:
        """对比两个版本的差异（校验版本属于指定内容）"""
        import difflib

        v1 = self.db.query(ContentVersion).filter(
            ContentVersion.id == version_id_1,
            ContentVersion.content_id == content_id,
        ).first()
        v2 = self.db.query(ContentVersion).filter(
            ContentVersion.id == version_id_2,
            ContentVersion.content_id == content_id,
        ).first()

        if not v1 or not v2:
            raise ValueError("版本不存在")

        lines1 = (v1.content_md or '').splitlines(keepends=True)
        lines2 = (v2.content_md or '').splitlines(keepends=True)

        diff = list(difflib.ndiff(lines1, lines2))

        changes = []
        for line in diff:
            if line.startswith('  '):
                changes.append({'type': 'equal', 'text': line[2:]})
            elif line.startswith('- '):
                changes.append({'type': 'delete', 'text': line[2:]})
            elif line.startswith('+ '):
                changes.append({'type': 'insert', 'text': line[2:]})

        stats = {
            'added': sum(1 for c in changes if c['type'] == 'insert'),
            'removed': sum(1 for c in changes if c['type'] == 'delete'),
            'unchanged': sum(1 for c in changes if c['type'] == 'equal'),
        }

        return {'changes': changes, 'stats': stats}

    def list_contents(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        status: str = None,
        platform_code: str = None,
        search: str = None,
        tag: str = None,
    ):
        """获取内容列表"""
        query = self.db.query(Content).filter(Content.user_id == user_id)

        if status:
            try:
                query = query.filter(Content.status == ContentStatus(status))
            except ValueError:
                query = query.filter(Content.status == status)
        if platform_code:
            query = query.filter(Content.platform_code == platform_code)
        if search:
            query = query.filter(
                (Content.title.contains(search)) |
                (Content.topic.contains(search))
            )
        if tag:
            # 精确匹配标签，避免 LIKE '%tag%' 子串误报
            query = query.filter(
                (Content.tags == tag) |
                (Content.tags.like(f"{tag},%")) |
                (Content.tags.like(f"%,{tag}")) |
                (Content.tags.like(f"%,{tag},%"))
            )

        total = query.count()
        items = query.order_by(Content.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

        return items, total
