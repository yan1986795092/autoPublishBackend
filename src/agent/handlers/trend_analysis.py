"""TrendAnalysis — HotTopicAgent 的热点趋势分析工具"""

from __future__ import annotations
from collections import Counter
from datetime import datetime
from typing import Optional
import re


def analyze_topics(topics: list[dict]) -> dict:
    """对热点列表进行趋势分析

    Args:
        topics: [{"title": "...", "heat": "...", "tag": "..."}, ...]
    """
    if not topics:
        return {"trend_found": False, "summary": "无热点数据可分析"}

    # 按标签分类
    tag_groups: dict[str, list[str]] = {}
    all_titles: list[str] = []

    for t in topics:
        tag = t.get("tag", "未分类")
        if tag not in tag_groups:
            tag_groups[tag] = []
        tag_groups[tag].append(t["title"])
        all_titles.append(t["title"])

    # 标签分布
    tag_distribution = {tag: len(items) for tag, items in sorted(
        tag_groups.items(), key=lambda x: len(x[1]), reverse=True
    )}

    # 识别热点关键词
    all_words: list[str] = []
    for title in all_titles:
        cleaned = re.sub(r"[#\[\]【】]", "", title)
        words = [w for w in cleaned.split() if len(w) >= 2]
        all_words.extend(words)
    word_freq = Counter(all_words)
    hot_keywords = [{"keyword": w, "count": c} for w, c in word_freq.most_common(15)]

    # 内容主题识别
    tech_keywords = {"AI", "大模型", "编程", "科技", "算法", "数据", "芯片", "智能", "数字"}
    life_keywords = {"生活", "美食", "旅游", "健康", "教育", "情感", "时尚"}
    keywords_set = set(w for w, _ in word_freq.most_common(30))

    tech_overlap = keywords_set & tech_keywords
    life_overlap = keywords_set & life_keywords

    dominant_theme = "general"
    if len(tech_overlap) >= len(life_overlap):
        dominant_theme = "tech"
    if len(life_overlap) > len(tech_overlap):
        dominant_theme = "lifestyle"

    # 选题建议
    content_ideas = []
    for t in topics[:5]:
        title = t.get("title", "").lstrip("#【")
        tag = t.get("tag", "")
        if "科技" in tag or "教育" in tag:
            content_ideas.append({
                "title": f"深度解读：{title}",
                "format": "blog",
                "reasoning": f"热门科技话题，适合撰写分析博客",
            })
        elif "生活" in tag or "健康" in tag:
            content_ideas.append({
                "title": f"{title}的实用指南",
                "format": "social_post",
                "reasoning": f"生活类话题，适合制作分享帖",
            })
        else:
            content_ideas.append({
                "title": f"热点解读：{title}",
                "format": "blog",
                "reasoning": f"热门话题，适合内容创作",
            })

    return {
        "trend_found": True,
        "analysis_time": datetime.utcnow().isoformat(),
        "summary": f"共分析 {len(topics)} 条热点，覆盖 {len(tag_groups)} 个领域，热点关键词 {len(hot_keywords)} 个",
        "tag_distribution": tag_distribution,
        "hot_keywords": hot_keywords[:10],
        "dominant_theme": dominant_theme,
        "content_ideas": content_ideas[:5],
        "topic_count": len(topics),
        "tag_count": len(tag_groups),
    }
