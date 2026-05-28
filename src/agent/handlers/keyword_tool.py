"""KeywordTool — SEOAgent 的关键词研究与建议工具"""

from __future__ import annotations
import re
from collections import Counter
from typing import Optional


def extract_keywords(text: str, max_keywords: int = 10) -> list[dict]:
    """从文本中提取关键词及统计信息"""
    if not text:
        return []

    cleaned = re.sub(r"[^\w一-鿿\s]", " ", text)
    words = [w for w in cleaned.split() if len(w) > 1]
    word_freq = Counter(words)
    total = max(sum(word_freq.values()), 1)

    keywords = []
    for i, (word, count) in enumerate(word_freq.most_common(max_keywords)):
        keywords.append({
            "keyword": word,
            "count": count,
            "density": round(count / total, 4),
            "rank": i + 1,
        })
    return keywords


def suggest_related_keywords(topic: str, content: Optional[str] = None) -> list[str]:
    """基于主题生成关键词建议

    这是一个静态规则版本。实际使用时可以替换为 LLM 调用获取更丰富的建议。
    """
    suggestions = [topic]

    prefixes = ["什么是", "如何", "最佳", "2026"]
    for p in prefixes:
        suggestions.append(f"{p}{topic}")

    suffixes = ["教程", "指南", "案例", "工具", "趋势", "对比", "优缺点"]
    for s in suffixes:
        suggestions.append(f"{topic}{s}")

    if content:
        words = extract_keywords(content, 5)
        for kw in words[:3]:
            kw_str = kw["keyword"]
            if kw_str != topic:
                suggestions.append(kw_str)

    seen: set[str] = set()
    deduped = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:12]


def keyword_density_analysis(content: str, keywords: list[str]) -> dict:
    """分析内容中关键词密度"""
    if not content or not keywords:
        return {"keyword_stats": [], "overall_score": 100}

    content_lower = content.lower()
    total_words = max(len(content.split()), 1)
    stats = []

    for kw in keywords:
        kw_lower = kw.lower()
        count = content_lower.count(kw_lower)
        density = round(count / total_words, 4)
        score = 100
        if density < 0.005:
            score = 40
        elif density < 0.01:
            score = 60
        elif density > 0.05:
            score = 50
        stats.append({"keyword": kw, "count": count, "density": density, "score": score})

    avg_score = int(sum(s["score"] for s in stats) / max(len(stats), 1))
    return {
        "keyword_stats": stats,
        "overall_score": avg_score,
        "total_words": total_words,
    }
