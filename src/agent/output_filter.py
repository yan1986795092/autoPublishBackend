"""Agent 输出过滤器 — 识别"伪内容"和不合格输出

背景：glm-4-flash 在某些场景会输出带"## 思考：…## 正文：…"格式的迷你短文，
这些不是用户想要的实际内容，会污染 pipeline 输出。本模块提供：
- looks_like_meta_dump: 识别 LLM 自我元描述
- is_too_short_content: 识别太短的输出
- strip_meta_dump: 剥离元描述行
- content_quality_check: 综合质量检查
"""

from __future__ import annotations
import re

# 识别"自我元描述"——这些是 LLM 内部结构化输出（## 思考 / ## 正文 / ## 章节），
# 区别于正常文章的 H2 标题（如 ## 标题、## 引言、## 结论等结构）。
# 核心特征：前缀"思考"通常紧跟"正文"出现（写完思路立刻写正文），
# 而"标题/引言/结论"是博客/文章的合法结构。
_QUALITY_BLACKLIST = re.compile(
    r'^\s*#{1,3}\s*(思考|正文|章节)\s*[:：]?',
    re.MULTILINE,
)

# 多个 Markdown 标题（## xxx）—— 仍保留，但需要至少 2 个"可疑"元描述才拦截
_MULTI_HEADER = re.compile(r'(^|\n)#{1,3}\s+\S')

# 行内"思考：xxx"或"正文：xxx"模式（不需要 # 前缀，LLM 也常这么写）
_INLINE_META = re.compile(
    r'(?:^|\n)\s*(思考|正文|章节)\s*[:：][^\n]{0,200}',
)


def looks_like_meta_dump(text: str) -> bool:
    """检测文本是否包含 LLM 自我元描述（## 思考 / ## 正文 等）"""
    if not text:
        return False
    matches = _QUALITY_BLACKLIST.findall(text)
    # 出现 2 个以上元描述前缀 → 视为元描述 dump
    return len(matches) >= 2


def is_too_short_content(text: str, min_chars: int = 100) -> bool:
    """检测文本是否太短（无意义输出）"""
    cleaned = re.sub(r'\s+', '', text or "")
    return len(cleaned) < min_chars


def strip_meta_dump(text: str) -> str:
    """剥离元描述行，返回纯正文"""
    if not text:
        return text
    lines = text.split('\n')
    out = []
    for line in lines:
        if _QUALITY_BLACKLIST.match(line):
            continue
        out.append(line)
    cleaned = '\n'.join(out).strip()
    # 再处理行内"思考：xxx"（常出现在段落里）
    cleaned = _INLINE_META.sub('', cleaned).strip()
    return cleaned


def content_quality_check(text: str, min_chars: int = 100) -> tuple[bool, str]:
    """综合内容质量检查

    Returns:
        (is_valid, reason)
    """
    if is_too_short_content(text, min_chars=min_chars):
        return False, f"content too short (< {min_chars} chars)"
    if looks_like_meta_dump(text):
        return False, "content contains LLM meta-dump (## 思考/## 正文/...)"
    return True, ""
