from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional

from src.database import get_db
from src.models import Platform
from src.config import get_settings

router = APIRouter(prefix="/platforms", tags=["平台"])


DEFAULT_PLATFORMS = [
    # ═══════════════════════════════════════════════════════════
    # 社交平台 — 可一键发布到外部渠道
    # ═══════════════════════════════════════════════════════════
    {"code": "xiaohongshu", "name": "小红书", "category": "social_media", "content_genre": "text", "is_mock": True},
    {"code": "wechat", "name": "微信公众号", "category": "social_media", "content_genre": "text", "is_mock": True},
    {"code": "zhihu", "name": "知乎", "category": "social_media", "content_genre": "text", "is_mock": True},
    {"code": "weibo", "name": "微博", "category": "social_media", "content_genre": "text", "is_mock": True},
    {"code": "douyin", "name": "抖音", "category": "social_media", "content_genre": "text", "is_mock": True},
    {"code": "bilibili", "name": "B站", "category": "social_media", "content_genre": "text", "is_mock": True},
    # ═══════════════════════════════════════════════════════════
    # 营销体裁 — 按对应结构/语气生成内容
    # ═══════════════════════════════════════════════════════════
    {"code": "copywriting", "name": "产品文案", "category": "content", "content_genre": "text", "is_mock": True},
    {"code": "brand-story", "name": "品牌故事", "category": "content", "content_genre": "text", "is_mock": True},
    {"code": "ad-copy", "name": "广告文案", "category": "content", "content_genre": "text", "is_mock": True},
    {"code": "fiction", "name": "小说创作", "category": "content", "content_genre": "text", "is_mock": True},
    {"code": "script", "name": "剧本脚本", "category": "content", "content_genre": "text", "is_mock": True},
    {"code": "storyboard", "name": "分镜脚本", "category": "content", "content_genre": "video", "is_mock": True},
    {"code": "newsletter", "name": "邮件通讯", "category": "content", "content_genre": "text", "is_mock": True},
    # ═══════════════════════════════════════════════════════════
    # 专业文体 — 结构严谨的内容格式
    # ═══════════════════════════════════════════════════════════
    {"code": "tech-doc", "name": "技术文档", "category": "professional", "content_genre": "text", "is_mock": True},
    {"code": "tutorial", "name": "教程指南", "category": "professional", "content_genre": "text", "is_mock": True},
    {"code": "email", "name": "邮件营销", "category": "professional", "content_genre": "text", "is_mock": True},
    {"code": "seo", "name": "SEO文章", "category": "professional", "content_genre": "text", "is_mock": True},
]


def _ensure_default_platforms(db: Session):
    """确保所有默认平台都存在（upsert 逻辑，不覆盖已有修改）"""
    defaults = {p["code"]: p for p in DEFAULT_PLATFORMS}
    existing = {p.code: p for p in db.query(Platform).filter(Platform.code.in_(defaults.keys())).all()}

    # ══ 后向兼容：把旧版 technical_doc 重命名为 tech-doc ══
    old_techdoc = db.query(Platform).filter(Platform.code == "technical_doc").first()
    if old_techdoc and "tech-doc" not in existing:
        old_techdoc.code = "tech-doc"
        old_techdoc.name = "技术文档"
        old_techdoc.category = "professional"
        old_techdoc.content_genre = "text"
        old_techdoc.is_mock = True
        existing["tech-doc"] = old_techdoc
        print(f"  迁移旧平台: technical_doc → tech-doc")
        db.commit()

    for code, data in defaults.items():
        if code in existing:
            # 更新分类和体裁（新加的字段可能为空）
            if not existing[code].category and data.get("category"):
                existing[code].category = data["category"]
            if not existing[code].content_genre and data.get("content_genre"):
                existing[code].content_genre = data["content_genre"]
        else:
            db.add(Platform(**data))

    if len(existing) < len(defaults):
        db.commit()


@router.get("", response_model=dict)
def list_platforms(db: Session = Depends(get_db)):
    """获取支持的内容类型/平台列表"""
    _ensure_default_platforms(db)
    platforms = db.query(Platform).filter(Platform.is_active == True).all()

    return {
        "success": True,
        "data": [
            {
                "code": p.code,
                "name": p.name,
                "icon_url": p.icon_url,
                "category": p.category or "social_media",
                "content_genre": p.content_genre or "text",
                "is_mock": p.is_mock,
                "is_active": p.is_active,
            }
            for p in platforms
        ]
    }


@router.get("/models", response_model=dict)
def list_available_models():
    """返回按 provider 分组的可用模型（根据 API key 配置情况标记可用性）"""
    settings = get_settings()

    providers = [
        {
            "provider": "openai",
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-16k"],
            "available": bool(settings.openai_api_key),
        },
        {
            "provider": "zhipu",
            "name": "智谱",
            "models": ["glm-4-flash", "glm-4-plus", "glm-4-long"],
            "available": bool(settings.zhipu_api_key),
        },
        {
            "provider": "qwen",
            "name": "通义千问",
            "models": ["qwen-plus", "qwen-max", "qwen-turbo"],
            "available": bool(settings.qwen_api_key),
        },
    ]

    return {"success": True, "data": {"providers": providers}}
