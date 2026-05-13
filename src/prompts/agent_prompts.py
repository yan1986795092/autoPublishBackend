"""Agent 系统 Prompt 模板和写作指南"""

CONTENT_TYPE_PROMPTS = {
    "blog": "写一篇技术博客文章。要求：有吸引力的标题、引言、2-4个主要章节、结论。使用专业但易懂的语言。",
    "social_post": "写一篇社交媒体帖子。要求：吸引眼球的开头、简短有力、有互动引导。",
    "copywriting": "写一篇营销文案。要求：抓住痛点的标题、价值主张、行动号召。",
    "script": "写一个视频脚本。要求：口播自然、有开场钩子、分镜描述。",
    "newsletter": "写一封邮件Newsletter。要求：友好的问候语、2-3个核心内容。",
    "report": "写一份分析报告。要求：数据驱动、结构化、有结论和建议。",
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

BANNED_WORDS = ["首个", "独家", "唯一", "绝对", "顶级", "百分百", "100%", "全网", "国家级", "最佳"]

SYSTEM_PROMPT_TEMPLATE = """You are an AI {role}. Your task is to {description}.

Follow these guidelines:
1. Think step by step before producing your output.
2. Structure your output clearly.
3. If the input is ambiguous, state your assumptions.

Output format:
- First, write your reasoning (prefixed with "REASONING:")
- Then, write your final output (prefixed with "OUTPUT:")"""

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
