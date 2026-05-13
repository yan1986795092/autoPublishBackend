"""Agent 共享常量和工具函数 — 从 nodes.py 提取"""

from __future__ import annotations
import re

# ── 内容类型 Prompt 模板 ──

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

_BANNED_PATTERNS = [
    re.compile(r'(?:排名|销量|口碑|销量|业绩|行业|全国|全球|全网)\s*第一'),
    re.compile(r'第一\s*(?:品牌|名|好|大)'),
    re.compile(r'第\s*一\s*(?:位|名)'),
]
for w in BANNED_WORDS:
    _BANNED_PATTERNS.append(re.compile(re.escape(w)))

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


# ── 工具函数 ──

def extract_output(response: str) -> str:
    """从 LLM 响应中提取 OUTPUT: 后面的内容"""
    if "OUTPUT:" in response:
        return response[response.index("OUTPUT:") + len("OUTPUT:"):].strip()
    return response.strip()


def check_banned_words(content: str) -> list[dict]:
    """违禁词检查，返回匹配列表"""
    banned_found = []
    seen = set()
    for pattern in _BANNED_PATTERNS:
        for match in pattern.finditer(content):
            word = match.group()
            if word not in seen:
                seen.add(word)
                banned_found.append({"word": word, "severity": "warning"})
    return banned_found


def seo_score(content: str, keywords: list[dict], topic: str) -> dict | None:
    """计算 SEO 评分"""
    if not content:
        return None
    content_lower = content.lower()
    word_count = len(content.split())
    keyword_scores = {}
    for k in keywords:
        kw = k["keyword"].lower()
        count = content_lower.count(kw)
        density = count / max(word_count, 1)
        target = k.get("density_target", 0.02)
        keyword_scores[kw] = {
            "count": count, "density": round(density, 4),
            "score": min(100, int((1 - abs(density - target) / target) * 100)) if target else 50,
        }
    avg_kw = sum(v["score"] for v in keyword_scores.values()) / max(len(keyword_scores), 1)
    title_score = 80 if 20 <= len(topic) <= 60 else 50
    readability = 90 if 15 <= (word_count / max(len(content.split("。")), 1)) <= 30 else 70
    return {
        "overall": int(avg_kw * 0.4 + title_score * 0.3 + readability * 0.3),
        "word_count": word_count,
        "details": {"title_score": title_score, "keyword_score": int(avg_kw), "readability_score": readability},
    }


def seo_suggestions(content: str, keywords: list[dict], score: dict | None) -> list[str]:
    """生成 SEO 建议"""
    suggestions = []
    if score and score.get("details", {}).get("title_score", 100) < 70:
        suggestions.append("标题长度建议在 20-60 字符之间")
    if score and score.get("details", {}).get("keyword_score", 100) < 70:
        suggestions.append("关键词使用不足，建议在标题和首段中包含核心关键词")
    suggestions.append("建议在标题和首段中包含核心关键词")
    return suggestions
