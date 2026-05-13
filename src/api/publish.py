from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
import random
import json


class BatchPublishRequest(BaseModel):
    content_id: str
    platforms: list[str]
    publish_mode: str = "draft"
    schedule_time: str | None = None

from src.database import get_db
from src.models import Content, PublishRecord, Platform, ContentStatus, PublishStatus
from src.schemas.publish import PublishRequest
from src.api.auth import get_current_user
from src.services.content_service import ContentService

router = APIRouter(prefix="/publish", tags=["发布"])


@router.post("", response_model=dict)
def publish_content(
    request: PublishRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """发布内容到指定平台"""
    content = db.query(Content).filter(
        Content.id == request.content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    # 定时发布
    if request.schedule_time:
        try:
            scheduled_at = datetime.fromisoformat(request.schedule_time)
            if scheduled_at > datetime.utcnow():
                record = PublishRecord(
                    content_id=content.id,
                    platform_code=request.platform_code,
                    status=PublishStatus.PENDING,
                    response_data=json.dumps({"scheduled_at": request.schedule_time}),
                )
                db.add(record)
                db.commit()
                return {
                    "success": True,
                    "data": {
                        "publish_record_id": record.id,
                        "status": "scheduled",
                        "platform": request.platform_code,
                        "mode": request.publish_mode,
                        "scheduled_at": request.schedule_time,
                    }
                }
        except ValueError:
            pass  # fall through to immediate publish

    # Mock 模式：模拟发布结果（90% 成功率）
    success = random.random() > 0.1
    platform_name_map = {
        "xiaohongshu": "小红书",
        "wechat": "微信公众号",
        "zhihu": "知乎",
    }

    # 创建发布记录（必须在引用 record.id 之前）
    record = PublishRecord(
        content_id=content.id,
        platform_code=request.platform_code,
        status=PublishStatus.PENDING,
    )
    db.add(record)
    db.flush()

    url_patterns = {
        "xiaohongshu": f"https://www.xiaohongshu.com/explore/{record.id}",
        "wechat": f"https://mp.weixin.qq.com/s/{record.id}",
        "zhihu": f"https://zhuanlan.zhihu.com/p/{record.id}",
    }

    if success:
        record.status = PublishStatus.SUCCESS
        record.platform_post_id = f"mock_{record.id}"
        record.platform_url = url_patterns.get(request.platform_code, url_patterns["xiaohongshu"])
        record.published_at = datetime.utcnow()

        if request.publish_mode == "publish":
            content.status = ContentStatus.PUBLISHED

        db.commit()
        return {
            "success": True,
            "data": {
                "publish_record_id": record.id,
                "status": record.status.value,
                "platform": request.platform_code,
                "mode": request.publish_mode,
                "platform_post_id": record.platform_post_id,
                "platform_url": record.platform_url,
            }
        }
    else:
        record.status = PublishStatus.FAILED
        record.error_message = "模拟网络异常，请重试"
        record.response_data = json.dumps({"error": "mock_network_error"})
        db.commit()
        return {
            "success": False,
            "error": {
                "code": "PUBLISH_FAILED",
                "message": "发布失败",
                "detail": record.error_message,
            }
        }


@router.get("/{record_id}", response_model=dict)
def get_publish_status(
    record_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取发布状态"""
    record = db.query(PublishRecord).join(Content).filter(
        PublishRecord.id == record_id,
        Content.user_id == current_user.id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail="发布记录不存在")

    return {
        "success": True,
        "data": {
            "id": record.id,
            "content_id": record.content_id,
            "platform_code": record.platform_code,
            "status": record.status.value,
            "platform_post_id": record.platform_post_id,
            "platform_url": record.platform_url,
            "error_message": record.error_message,
            "published_at": record.published_at.isoformat() if record.published_at else None,
        }
    }


@router.get("/best-time", response_model=dict)
def get_best_publish_time(
    platform_code: str = "xiaohongshu",
    db: Session = Depends(get_db)
):
    """获取最佳发布时间建议"""
    service = ContentService(db)
    suggestion = service.suggest_best_time(platform_code)
    return {"success": True, "data": suggestion}


@router.get("/content/{content_id}", response_model=dict)
def get_content_publish_history(
    content_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取内容的所有发布记录"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    records = db.query(PublishRecord).filter(
        PublishRecord.content_id == content_id
    ).order_by(PublishRecord.created_at.desc()).all()

    return {
        "success": True,
        "data": {
            "records": [
                {
                    "id": r.id,
                    "platform_code": r.platform_code,
                    "status": r.status.value,
                    "platform_post_id": r.platform_post_id,
                    "platform_url": r.platform_url,
                    "error_message": r.error_message,
                    "published_at": r.published_at.isoformat() if r.published_at else None,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]
        }
    }


@router.post("/batch", response_model=dict)
def batch_publish(
    request: BatchPublishRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """批量发布到多个平台"""
    content = db.query(Content).filter(
        Content.id == request.content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    results = []

    # 定时发布场景
    if request.schedule_time:
        try:
            scheduled_at = datetime.fromisoformat(request.schedule_time)
            if scheduled_at > datetime.utcnow():
                for platform_code in request.platforms:
                    record = PublishRecord(
                        content_id=content.id,
                        platform_code=platform_code,
                        status=PublishStatus.PENDING,
                        response_data=json.dumps({"scheduled_at": request.schedule_time}),
                    )
                    db.add(record)
                    db.flush()
                    results.append({
                        "publish_record_id": record.id,
                        "status": "scheduled",
                        "platform": platform_code,
                        "mode": request.publish_mode,
                        "scheduled_at": request.schedule_time,
                    })
                db.commit()
                return {"success": True, "data": {"results": results}}
        except ValueError:
            pass

    for platform_code in request.platforms:
        record = PublishRecord(
            content_id=content.id,
            platform_code=platform_code,
            status=PublishStatus.PENDING,
        )
        db.add(record)
        db.flush()

        success = random.random() > 0.1

        if success:
            record.status = PublishStatus.SUCCESS
            record.platform_post_id = f"mock_{record.id}"
            url_patterns = {
                "xiaohongshu": f"https://www.xiaohongshu.com/explore/{record.id}",
                "wechat": f"https://mp.weixin.qq.com/s/{record.id}",
                "zhihu": f"https://zhuanlan.zhihu.com/p/{record.id}",
            }
            record.platform_url = url_patterns.get(platform_code)
            record.published_at = datetime.utcnow()

            if request.publish_mode == "publish":
                content.status = ContentStatus.PUBLISHED

            results.append({
                "publish_record_id": record.id,
                "status": "success",
                "platform": platform_code,
                "mode": request.publish_mode,
                "platform_post_id": record.platform_post_id,
                "platform_url": record.platform_url,
            })
        else:
            record.status = PublishStatus.FAILED
            record.error_message = "模拟网络异常，请重试"
            results.append({
                "publish_record_id": record.id,
                "status": "failed",
                "platform": platform_code,
                "mode": request.publish_mode,
                "error_message": record.error_message,
            })

    db.commit()
    return {"success": True, "data": {"results": results}}
