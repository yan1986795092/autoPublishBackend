"""评估工具 — SEO 评分、违禁词检查、输出提取"""

from __future__ import annotations
import re

BANNED_WORDS = ["首个", "独家", "唯一", "绝对", "顶级", "百分百", "100%", "全网", "国家级", "最佳"]

_BANNED_PATTERNS = [
    re.compile(r'(?:排名|销量|口碑|行业|全国|全球|全网)\s*第一'),
    re.compile(r'第一\s*(?:品牌|名|好|大)'),
    re.compile(r'第\s*一\s*(?:位|名)'),
]
for w in BANNED_WORDS:
    _BANNED_PATTERNS.append(re.compile(re.escape(w)))


def extract_output(response: str) -> str:
    """从 LLM 响应中提取正文内容（支持多种分隔符，按优先级依次尝试）

    1. "正文：" —— WritingAgent 的中文输出格式
    2. "OUTPUT:" —— 英文输出格式
    3. "### OUTPUT" / "```output" —— Markdown 代码块格式
    4. 整体返回原始内容（strip 后）

    即使没有任何分隔符也返回有效内容，不返回空字符串。
    """
    if not response:
        return ""

    # 1. 尝试中文分隔符
    if "正文：" in response:
        return response.split("正文：", 1)[1].strip()
    if "正文:" in response:
        return response.split("正文:", 1)[1].strip()

    # 2. 尝试英文分隔符
    if "OUTPUT:" in response:
        return response.split("OUTPUT:", 1)[1].strip()
    if "Output:" in response:
        return response.split("Output:", 1)[1].strip()

    # 3. 尝试 Markdown 代码块（```output ... ```）
    if "```output" in response:
        parts = response.split("```output", 1)
        if len(parts) > 1 and "```" in parts[1]:
            return parts[1].split("```", 1)[0].strip()
    if "```json" in response:
        # 尝试解析 JSON
        import json
        json_str = response.split("```json", 1)[1].split("```", 1)[0].strip()
        try:
            data = json.loads(json_str)
            return data.get("content", data.get("output", json.dumps(data, ensure_ascii=False, indent=2)))
        except (json.JSONDecodeError, Exception):
            return response.strip()

    # 4. 去掉推理前缀后返回
    stripped = response.strip()
    for prefix in ["思考：", "思考:", "Reasoning:", "reasoning:"]:
        if prefix.lower() in stripped.lower():
            idx = stripped.lower().index(prefix.lower())
            # 只在 prefix 出现在开头时裁剪
            if idx == 0:
                stripped = stripped[len(prefix):].strip()
                break

    return stripped if stripped else ""


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
    return suggestions
