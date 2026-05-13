from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional

from src.database import get_db
from src.models import Platform
from src.config import get_settings
from src.services.hot_topic_service import hot_topic_service

router = APIRouter(prefix="/platforms", tags=["平台"])


@router.get("", response_model=dict)
def list_platforms(db: Session = Depends(get_db)):
    """获取支持的平台列表"""
    platforms = db.query(Platform).filter(Platform.is_active == True).all()

    if not platforms:
        default_platforms = [
            Platform(code="xiaohongshu", name="小红书", is_mock=True),
            Platform(code="wechat", name="微信公众号", is_mock=True),
            Platform(code="zhihu", name="知乎", is_mock=True),
        ]
        for p in default_platforms:
            db.add(p)
        db.commit()
        platforms = default_platforms

    return {
        "success": True,
        "data": [
            {
                "code": p.code,
                "name": p.name,
                "icon_url": p.icon_url,
                "is_mock": p.is_mock,
                "is_active": p.is_active,
            }
            for p in platforms
        ]
    }


@router.get("/hot-topics", response_model=dict)
def get_hot_topics(
    platform: Optional[str] = None,
    limit: int = Query(10, ge=1, le=20),
):
    """获取各平台热门话题"""
    data = hot_topic_service.get_hot_topics(platform, limit)
    return {"success": True, "data": data}


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
