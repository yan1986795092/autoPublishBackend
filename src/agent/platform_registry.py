"""PlatformRegistry — 平台元数据中央注册表

DB 的 `Platform` 表是平台元数据的来源（code、name、category）。
本模块提供：
- 每个平台的 content_type / SEO rewrite 模式（供 LLM prompt 和 WritingAgent 参考）
- detect_platform() 从用户请求中识别平台
- get_names_map() 提供平台名称映射

重要：Pipeline（用哪些 Agent、什么顺序）由 LLM 自主决定。
本模块只提供平台元数据作为上下文提示，不硬编码任何流程。

平台代码与 DB Platform.code 一致：
  social_media:   xiaohongshu, wechat, zhihu
  creative_writing: fiction, storyboard, script
  marketing:      copywriting, newsletter
  technical:      technical_doc, tutorial
"""

from __future__ import annotations
from typing import NamedTuple


class PlatformMeta(NamedTuple):
    """平台元数据 — 仅供 LLM 上下文提示用，不控制流程"""
    code: str           # 与 DB Platform.code 一致
    name: str           # 中文显示名
    category: str        # social_media | creative_writing | marketing | technical
    content_type: str    # WritingAgent 的 content_type
    seo_rewrite: bool    # SEOAgent 是否用 rewrite 模式（True=改写内容，False=分析评分）
    # 以下供 SEO 改写 prompt 选择
    rewrite_prompt_key: str  # seo.py _REWRITE_PROMPTS 的 key


# ── 平台元数据（LLM 上下文用）──────────────────────────────────────────
PLATFORMS: dict[str, PlatformMeta] = {
    # ── 社交媒体 ─────────────────────────────────────────
    "xiaohongshu": PlatformMeta(
        code="xiaohongshu", name="小红书", category="social_media",
        content_type="xhs_post", seo_rewrite=True, rewrite_prompt_key="xiaohongshu",
    ),
    "wechat": PlatformMeta(
        code="wechat", name="微信公众号", category="social_media",
        content_type="wechat_article", seo_rewrite=False, rewrite_prompt_key="wechat",
    ),
    "zhihu": PlatformMeta(
        code="zhihu", name="知乎", category="social_media",
        content_type="zhihu_article", seo_rewrite=False, rewrite_prompt_key="zhihu",
    ),
    # ── 创意写作 ───────────────────────────────────────
    "fiction": PlatformMeta(
        code="fiction", name="小说创作", category="creative_writing",
        content_type="fiction", seo_rewrite=False, rewrite_prompt_key="",
    ),
    "storyboard": PlatformMeta(
        code="storyboard", name="分镜脚本", category="creative_writing",
        content_type="storyboard", seo_rewrite=False, rewrite_prompt_key="",
    ),
    "script": PlatformMeta(
        code="script", name="剧本创作", category="creative_writing",
        content_type="script", seo_rewrite=False, rewrite_prompt_key="",
    ),
    # ── 营销内容 ──────────────────────────────────────
    "copywriting": PlatformMeta(
        code="copywriting", name="营销文案", category="marketing",
        content_type="copywriting", seo_rewrite=True, rewrite_prompt_key="copywriting",
    ),
    "newsletter": PlatformMeta(
        code="newsletter", name="邮件通讯", category="marketing",
        content_type="newsletter", seo_rewrite=True, rewrite_prompt_key="newsletter",
    ),
    # ── 技术内容 ──────────────────────────────────────
    "technical_doc": PlatformMeta(
        code="technical_doc", name="技术文档", category="technical",
        content_type="technical_doc", seo_rewrite=False, rewrite_prompt_key="",
    ),
    "tutorial": PlatformMeta(
        code="tutorial", name="教程指南", category="technical",
        content_type="tutorial", seo_rewrite=False, rewrite_prompt_key="",
    ),
}

# ── 检测关键词（用于从用户请求识别平台）──────────────────────────────
_PLATFORM_KEYWORDS: list[tuple[str, list[str]]] = [
    ("xiaohongshu", ["小红书", "xhs", "种草", "好物分享"]),
    ("wechat",     ["公众号", "微信文章", "微信图文"]),
    ("zhihu",     ["知乎", "zhihu", "问答"]),
    ("fiction",    ["小说", "故事", "fiction"]),
    ("storyboard", ["分镜", "storyboard"]),
    ("script",    ["剧本", "编剧"]),
    ("copywriting", ["营销", "文案", "推广", "转化", "广告"]),
    ("newsletter", ["newsletter", "邮件通讯", "邮件订阅"]),
    ("technical_doc", ["技术文档", "technical_doc"]),
    ("tutorial",   ["教程", "指南", "tutorial"]),
]


def detect_platform(user_request: str) -> str | None:
    """从用户请求中检测目标平台，返回 code 或 None（未检测到）"""
    text = user_request.lower()
    for code, keywords in _PLATFORM_KEYWORDS:
        if any(kw in text for kw in keywords):
            return code
    return None


def get_names_map() -> dict[str, str]:
    """返回 code → name 的映射"""
    return {code: p.name for code, p in PLATFORMS.items()}


def get_all_codes() -> list[str]:
    """返回所有平台代码"""
    return list(PLATFORMS.keys())
