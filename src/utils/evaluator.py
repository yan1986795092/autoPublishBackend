"""评估工具 — SEO 评分、违禁词检查、输出提取"""

from __future__ import annotations
import re

BANNED_WORDS = ["首个", "独家", "唯一", "绝对", "顶级", "百分百", "100%", "全网", "国家级", "最佳"]

_BANNED_PATTERNS = [
    re.compile(r'(?:排名|销量|口碑|销量|业绩|行业|全国|全球|全网)\s*第一'),
    re.compile(r'第一\s*(?:品牌|名|好|大)'),
    re.compile(r'第\s*一\s*(?:位|名)'),
]
for w in BANNED_WORDS:
    _BANNED_PATTERNS.append(re.compile(re.escape(w)))


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
