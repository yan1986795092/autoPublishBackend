"""Agent 共享常量和工具函数 — 从 nodes.py 提取

违禁词检查相关函数 (check_banned_words / BANNED_WORDS / 正则模式)
统一由 src.utils.evaluator 维护，此处通过 import 引用避免重复。

System Prompt 模板现在统一由 src.prompts 管理，此处保留引用以向后兼容。
"""

from __future__ import annotations

from .registry import format_agent_descriptions

from src.utils.evaluator import (
    BANNED_WORDS,
    check_banned_words,
    extract_output,
    seo_score,
    seo_suggestions,
)

# ── Prompt 模板（从 prompts 模块导入，保持向后兼容）──
from src.prompts import (
    CONTENT_TYPE_PROMPTS,
    STYLE_GUIDES,
)

# ── 这些常量将逐步迁移到 prompts/builtins/ 中 ──
# 保留在此处确保现有 import 不中断

QUALITY_CHECKS = [
    "内容是否完整，有开头和结尾",
    "是否有明显的逻辑断层",
    "语言是否流畅，语感是否自然",
    "是否有违禁词或敏感词",
    "段落长度是否适中，是否便于阅读",
]

SYSTEM_PROMPT_TEMPLATE = """你是一位经验丰富的内容创作者，擅长根据不同场景和需求，创作出自然流畅、有温度的文字。

写作原则：
1. 语言要自然流畅，像和人聊天一样，不要生硬刻板
2. 结构清晰但不教条，让读者感觉舒适
3. 根据内容和受众灵活调整风格，不要机械套模板
4. 适当使用口语化表达，拉近和读者的距离
5. 有自己的观点和态度，不要面面俱到却四平八稳

输出要求：
- 先写下你的思路和判断（用「思考：」开头）
- 然后直接输出正文内容（用「正文：」开头），不要加任何「以下是...」之类的过渡
- 正文内容直接开始，不需要「好的，我来为您...」这样的开场白"""

# ── 提示词：理解 Agent（纯 LLM）────────────────────────────────────

UNDERSTANDING_SYSTEM_PROMPT = """你是一个意图理解助手。请分析用户请求，输出结构化的 JSON。

## 意图分类

粗粒度（互斥，选一个）：
- CREATE：从零生成新内容
- REVISE：基于已有内容修改、润色、扩写、缩写
- CONTINUE：继续生成未完成的内容
- SUMMARIZE：总结、提炼要点
- TRANSFORM：改变格式或风格（如文章→表格，正式→幽默）
- ASK：询问关于内容的知识或建议（非生成动作）
- MIXED：包含多个意图

## 输出字段

{
  "intent": {
    "coarse": "CREATE",
    "fine": "2-5个词描述具体动作"
  },
  "slots": {
    "content_type": "文章|代码|邮件|脚本|文案|报告|笔记|social_post|fiction|technical_doc|tutorial|广告|品牌故事|其他",
    "domain": "科技|金融|医疗|教育|娱乐|电商|生活|职场|其他",
    "creativity_level": "low|medium|high",
    "length_hint": "short|medium|long",
    "tone": "正式|友好|紧迫|幽默|同理心|专业|温暖|null",
    "audience": "专家|大众|儿童|决策者|技术人员|普通用户|null"
  },
  "missing_slots": ["列出用户没提供但合理应该问的槽位"],
  "cleaned_prompt": "清理/优化后的请求（保留原意，修复错别字）",
  "safety": true,
  "safety_reason": "",
  "platform": "用户明确提到的平台代码，没提则为 null",
  "platform_name": "平台中文名",
  "language": "zh|en"
}

规则：
1. 用户没明确提平台就不要推断，设 null
2. missing_slots 只放确实需要但不明确的关键信息（如没提篇幅就加 length_hint）
3. 如果请求明显不完整，在 missing_slots 中说明
4. 只输出 JSON，不要 markdown，不要解释"""


# ── 提示词：规划 Agent（无 suggested_agents 干扰）──────────────────

_PLANNING_AGENTS_LIST = format_agent_descriptions()

PLANNING_SYSTEM_PROMPT = f"""你将用户的请求分解为子任务，分配给专门的 AI Agent 执行。

可用 Agent 及其能力：
{_PLANNING_AGENTS_LIST}

## 依赖规则
- dependencies 是数组，内容是前面任务的索引（从 0 开始）
- 任务 A 依赖任务 B → A.dependencies = [B的索引]
- 无依赖的任务写 dependencies = []

## 示例

示例 1：写作任务（需搜索+热点分析）
{{
  "goal": "查看当前热点，为一款智能笔记应用写产品推广文案",
  "tasks": [
    {{
      "agent_type": "WebSearch",
      "input": {{"query": "智能笔记应用 2025 趋势", "count": 5}},
      "dependencies": []
    }},
    {{
      "agent_type": "HotTopicAgent",
      "input": {{"topic": "智能笔记应用热门趋势分析"}},
      "dependencies": [0]
    }},
    {{
      "agent_type": "WritingAgent",
      "input": {{"topic": "为一款智能笔记应用写产品推广文案，结合热点趋势", "content_type": "blog", "tone": "professional"}},
      "dependencies": [1]
    }},
    {{
      "agent_type": "ReviewAgent",
      "input": {{"topic": "为一款智能笔记应用写产品推广文案"}},
      "dependencies": [2]
    }},
    {{
      "agent_type": "FixAgent",
      "input": {{"topic": "为一款智能笔记应用写产品推广文案"}},
      "dependencies": [2, 3]
    }}
  ]
}}

示例 2：写作任务（无需搜索）
{{
  "goal": "写一篇关于时间管理的文章",
  "tasks": [
    {{
      "agent_type": "WritingAgent",
      "input": {{"topic": "写一篇关于时间管理的文章", "content_type": "article"}},
      "dependencies": []
    }},
    {{
      "agent_type": "ReviewAgent",
      "input": {{"topic": "写一篇关于时间管理的文章"}},
      "dependencies": [0]
    }}
  ]
}}

示例 3：代码任务
{{
  "goal": "用 Python 写一个文件监控脚本",
  "tasks": [
    {{
      "agent_type": "CodeAgent",
      "input": {{"task": "用 Python 写一个文件监控脚本", "language": "python"}},
      "dependencies": []
    }},
    {{
      "agent_type": "ReviewAgent",
      "input": {{"task": "用 Python 写一个文件监控脚本"}},
      "dependencies": [0]
    }}
  ]
}}

## 规则
1. 根据实际需求选择 Agent——没有固定模板。
2. **必须用 WebSearch 的场景**：用户请求包含"当前/最新/实时/热搜/流行趋势/2025/2026"等实时性关键词时，必须将 WebSearch 作为第一个任务。HotTopicAgent **不能替代** WebSearch 的搜索功能。
3. **热点分析的 pipeline**：如果使用了 WebSearch 且用户需要分析趋势/热点/流行方向，必须在 WebSearch 之后紧跟 HotTopicAgent 来分析搜索结果。Pipeline 顺序为：WebSearch → HotTopicAgent → WritingAgent。
4. 内容创作类任务必须包含 WritingAgent → ReviewAgent → FixAgent 链。
5. 所有依赖用索引引用前面的任务，不要用其他格式。
5. 输出必须是 JSON 对象（以 {{ 开头），不要直接输出数组。不要 markdown，不要解释。"""