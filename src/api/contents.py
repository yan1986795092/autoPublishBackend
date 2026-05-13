from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import json

from src.database import get_db
from src.models import Content, ContentVersion, ContentStatus, PublishRecord
from src.schemas.content import (
    GenerateRequest, GenerateResponse, GenerateMetadata,
    ContentListItem, ContentListResponse, PaginationInfo,
    ContentDetailResponse, ContentVersionItem,
    UpdateContentRequest, UpdateContentResponse,
    BatchGenerateRequest,
)
from src.api.auth import get_current_user
from src.services.content_service import ContentService
from src.core.exceptions import GenerateError

router = APIRouter(prefix="/contents", tags=["内容管理"])


@router.post("/generate", response_model=dict)
async def generate_content(
    request: GenerateRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI 内容生成"""
    try:
        service = ContentService(db)
        result = await service.generate_content(current_user.id, request)
        return {"success": True, "data": result}
    except Exception as e:
        import traceback
        print(f"[ERROR] generate_content failed: {e}")
        traceback.print_exc()
        raise GenerateError(message=f"生成失败: {str(e)}")


@router.post("/generate/stream")
async def generate_content_stream(
    request: GenerateRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """SSE 流式生成内容"""
    service = ContentService(db)
    return StreamingResponse(
        service.generate_content_stream(current_user.id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/generate/batch", response_model=dict)
async def batch_generate_content(
    request: BatchGenerateRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """批量生成多平台内容"""
    try:
        service = ContentService(db)
        results = await service.batch_generate(
            user_id=current_user.id,
            platforms=request.platforms,
            topic=request.topic,
            parameters=request.parameters,
            model=request.model,
        )
        return {"success": True, "data": {"contents": results}}
    except Exception as e:
        import traceback
        print(f"[ERROR] batch_generate failed: {e}")
        traceback.print_exc()
        raise GenerateError(message=f"批量生成失败: {str(e)}")


@router.get("", response_model=dict)
def list_contents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = None,
    platform_code: Optional[str] = None,
    search: Optional[str] = None,
    tag: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取内容列表"""
    service = ContentService(db)
    items, total = service.list_contents(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status,
        platform_code=platform_code,
        search=search,
        tag=tag,
    )

    return {
        "success": True,
        "data": {
            "items": [
                {
                    "id": c.id,
                    "title": c.title,
                    "topic": c.topic,
                    "platform_code": c.platform_code,
                    "status": c.status.value if hasattr(c.status, 'value') else c.status,
                    "tags": c.tags.split(",") if c.tags else [],
                    "view_count": c.view_count or 0,
                    "like_count": c.like_count or 0,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
                for c in items
            ],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size,
            }
        }
    }


@router.get("/{content_id}", response_model=dict)
def get_content(
    content_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取内容详情"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    versions = db.query(ContentVersion).filter(
        ContentVersion.content_id == content_id
    ).order_by(ContentVersion.version_number.desc()).all()

    return {
        "success": True,
        "data": {
            "id": content.id,
            "title": content.title,
            "topic": content.topic,
            "platform_code": content.platform_code,
            "content_md": content.content_md,
            "content_html": content.content_html,
            "status": content.status.value if hasattr(content.status, 'value') else content.status,
            "tags": content.tags.split(",") if content.tags else [],
            "view_count": content.view_count or 0,
            "like_count": content.like_count or 0,
            "metadata": json.loads(content.metadata_json) if content.metadata_json else {},
            "versions": [
                {
                    "id": v.id,
                    "version_number": v.version_number,
                    "change_summary": v.change_summary,
                    "created_at": v.created_at.isoformat(),
                }
                for v in versions
            ],
            "created_at": content.created_at.isoformat(),
            "updated_at": content.updated_at.isoformat(),
        }
    }


@router.put("/{content_id}", response_model=dict)
def update_content(
    content_id: str,
    request: UpdateContentRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新内容"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    if request.title is not None:
        content.title = request.title
    if request.content_md is not None:
        content.content_md = request.content_md
        content.content_html = request.content_md.replace("\n", "<br>")
    if request.tags is not None:
        content.tags = request.tags

    last_version = db.query(ContentVersion).filter(
        ContentVersion.content_id == content_id
    ).order_by(ContentVersion.version_number.desc()).first()

    new_version_number = (last_version.version_number + 1) if last_version else 1
    version = ContentVersion(
        content_id=content_id,
        version_number=new_version_number,
        content_md=request.content_md or content.content_md,
        content_html=content.content_html,
        change_summary=request.change_summary or f"版本 {new_version_number}",
    )
    db.add(version)
    db.commit()

    return {
        "success": True,
        "data": {
            "id": content.id,
            "version_number": new_version_number,
            "updated_at": content.updated_at.isoformat(),
        }
    }


@router.delete("/{content_id}", response_model=dict)
def delete_content(
    content_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除内容（软删除）"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    content.status = ContentStatus.ARCHIVED
    db.commit()

    return {
        "success": True,
        "data": {
            "id": content.id,
            "deleted_at": content.updated_at.isoformat(),
        }
    }


@router.put("/{content_id}/autosave", response_model=dict)
def autosave_content(
    content_id: str,
    request: UpdateContentRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """自动保存（不创建版本快照）"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    if request.content_md is not None:
        content.content_md = request.content_md
        content.content_html = request.content_md.replace("\n", "<br>")
    db.commit()

    return {
        "success": True,
        "data": {
            "id": content.id,
            "updated_at": content.updated_at.isoformat(),
        }
    }


@router.get("/calendar", response_model=dict)
def get_content_calendar(
    year: int = None,
    month: int = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取内容日历数据（按天分组的计划内容）"""
    from datetime import datetime, date
    import json

    now = datetime.utcnow()
    if not year:
        year = now.year
    if not month:
        month = now.month

    # 当月所有内容
    contents = db.query(Content).filter(
        Content.user_id == current_user.id,
    ).all()

    # 当月所有发布记录
    publish_records = db.query(PublishRecord).join(Content).filter(
        Content.user_id == current_user.id,
    ).all()

    # 按日期分组
    calendar: dict[str, list[dict]] = {}

    for c in contents:
        d = c.created_at.strftime("%Y-%m-%d") if c.created_at else ""
        if d and d.startswith(f"{year}-{month:02d}"):
            if d not in calendar:
                calendar[d] = []
            calendar[d].append({
                "id": c.id,
                "title": c.title or c.topic or "未命名",
                "type": "created",
                "platform_code": c.platform_code,
                "status": c.status.value if hasattr(c.status, 'value') else c.status,
            })

    for r in publish_records:
        if r.response_data:
            try:
                data = json.loads(r.response_data)
                scheduled = data.get("scheduled_at", "")
                if scheduled:
                    d = scheduled[:10]
                    if d.startswith(f"{year}-{month:02d}"):
                        if d not in calendar:
                            calendar[d] = []
                        calendar[d].append({
                            "id": r.content_id,
                            "title": r.content.title or r.content.topic or "未命名",
                            "type": "scheduled",
                            "platform_code": r.platform_code,
                            "status": r.status.value if hasattr(r.status, 'value') else r.status,
                        })
            except (json.JSONDecodeError, AttributeError):
                pass

    return {"success": True, "data": {"calendar": calendar, "year": year, "month": month}}


@router.post("/{content_id}/check-banned-words", response_model=dict)
def check_banned_words(
    content_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """检测内容中的敏感词"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    service = ContentService(db)
    words = service.check_banned_words(
        content.content_md or "",
        content.platform_code or "",
    )

    return {
        "success": True,
        "data": {
            "words": words,
            "total": len(words),
            "platform": content.platform_code,
        }
    }


@router.get("/{content_id}/versions/diff", response_model=dict)
def diff_versions(
    content_id: str,
    v1: str = Query(..., description="版本1 ID"),
    v2: str = Query(..., description="版本2 ID"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """对比两个版本的内容差异"""
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    service = ContentService(db)
    try:
        result = service.diff_versions(v1, v2)
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
