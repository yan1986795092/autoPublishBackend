"""Agent 共享常量和工具函数 — 从 nodes.py 提取

违禁词检查相关函数 (check_banned_words / BANNED_WORDS / 正则模式)
统一由 src.utils.evaluator 维护，此处通过 import 引用避免重复。
"""

from __future__ import annotations

from src.utils.evaluator import (
    BANNED_WORDS,
    check_banned_words,
    extract_output,
    seo_score,
    seo_suggestions,
)

# ── 内容类型 Prompt 模板 ──

CONTENT_TYPE_PROMPTS = {
    "blog": "写一篇技术博客文章。要求：有吸引力的标题、引言、2-4个主要章节、结论。使用专业但易懂的语言。",
    "social_post": "写一篇社交媒体帖子。要求：吸引眼球的开头、简短有力、有互动引导。",
    "copywriting": "写一篇营销文案。要求：抓住痛点的标题、价值主张、行动号召。",
    "script": "写一个视频脚本。要求：口播自然、有开场钩子、分镜描述。",
    "newsletter": "写一封邮件Newsletter。要求：友好的问候语、2-3个核心内容。",
    "report": "写一份分析报告。要求：数据驱动、结构化、有结论和建议。",
    "weibo_post": "写一条微博/即刻短帖。要求：140字以内、一针见血、有话题标签、适合快速传播。",
    "twitter_thread": "写一条 Twitter/X Thread。要求：英文为主、开头hook吸引点击、每条推文独立可读、有逻辑递进、结尾带CTA和话题标签。",
    "podcast_script": "写一期播客脚本。要求：有开场介绍、嘉宾引入（如适用）、2-3个讨论环节、收尾总结和预告。",
}

STYLE_GUIDES = {
    "技术": "使用专业术语但解释清楚，代码示例要完整可运行。",
    "种草": "第一人称体验分享，突出真实感和代入感。",
    "干货": "信息密度高，条理清晰，每个观点都有论据支持。",
    "幽默": "轻松诙谐的语气，适当使用比喻和夸张。",
    "深度": "系统全面，有独特见解，引用研究和数据。",
    "教程": "步骤清晰，从零开始，每个步骤都有预期结果说明。",
}

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

PLANNER_SYSTEM_PROMPT = """You are a content production manager. Your job is to break down user requests into a plan of sub-tasks that specialized AI agents will execute.

Available agents and their capabilities:
- WebSearch: Searches the internet for up-to-date information on a topic
- WritingAgent: Writes articles, copy, scripts in various styles and tones
- SEOAgent: Optimizes content for search engines, researches keywords
- DiagramAgent: Creates knowledge architecture diagrams, flowcharts, mind maps
- ReviewAgent: Reviews content for quality, banned words, factual accuracy
- PlatformAdapter: Converts content format for specific platforms (WeChat, Xiaohongshu, Zhihu)
- ImageAgent: Generates images and cover art
- HotTopicAgent: Analyzes trending topics and hot searches from social platforms
- CodeAgent: Generates code examples, tutorials, and technical documentation

Rules:
1. Break complex requests into sequential sub-tasks with clear dependencies
2. Tasks that can run in parallel should have no dependency on each other
3. For simple requests, a single WritingAgent task may be sufficient
4. Always include a ReviewAgent task for quality assurance
5. Response must be valid JSON

Output format:
```json
{
  "goal": "summary of the user's goal",
  "reasoning": "why you chose this plan",
  "tasks": [
    {
      "agent_type": "WebSearch",
      "input": {"query": "search query"},
      "dependencies": [],
      "reasoning": "why this task is needed"
    }
  ]
}
```"""
