from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload
from typing import Optional
import json

from src.database import get_db
from src.models import Content, ContentVersion, ContentStatus, PublishRecord, PublishStatus
from src.schemas.content import (
    GenerateRequest,
    UpdateContentRequest,
    BatchGenerateRequest,
    BatchIdsRequest,
    CreateContentRequest,
)
from src.api.auth import get_current_user
from src.services.content_service import ContentService
from src.core.exceptions import GenerateError
from src.logging_shared.config import get_logger

logger = get_logger("api.contents")

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
        logger.error("generate_content 失败", error=str(e))
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
        logger.error("batch_generate 失败", error=str(e))
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
    """获取内容列表（含全量统计）"""
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

    # 全量统计（受显式 user_id 过滤，确保 tenant 隔离）
    from src.models import ContentStatus
    from sqlalchemy import func as sa_func
    all_query = db.query(
        Content.status,
        sa_func.count(Content.id)
    ).filter(Content.user_id == current_user.id).group_by(Content.status).all()
    stats = {"total": 0, "draft": 0, "published": 0, "archived": 0}
    for s, cnt in all_query:
        key = s.value if hasattr(s, 'value') else str(s)
        stats[key] = cnt
        stats["total"] += cnt

    return {
        "success": True,
        "data": {
            "items": [
                {
                    "id": c.id,
                    "title": c.title,
                    "topic": c.topic,
                    "platform_code": c.platform_code,
                    "status": c.status.value if c.status else "unknown",
                    "tags": c.tags.split(",") if c.tags else [],
                    "content_md": (c.content_md or "")[:200],
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
            },
            "stats": stats,
        }
    }


@router.get("/calendar", response_model=dict)
def get_content_calendar(
    year: int | None = None,
    month: int | None = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取发布排期日历数据——按天分组的发布记录"""
    from datetime import datetime
    import json

    now = datetime.utcnow()
    if not year:
        year = now.year
    if not month:
        month = now.month

    # 查当月有 publish_records 的所有内容
    contents_with_records = db.query(Content).filter(
        Content.user_id == current_user.id,
        Content.publish_records.any(),
    ).all()
    content_ids = {c.id for c in contents_with_records}

    if not content_ids:
        return {"success": True, "data": {"calendar": {}, "year": year, "month": month}}

    # 取当月所有 publish_records（基于 published_at 或 created_at 过滤）
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year + 1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"

    publish_records = db.query(PublishRecord).options(
        selectinload(PublishRecord.content)
    ).filter(
        PublishRecord.content_id.in_(content_ids),
    ).all()

    # 按日期分组
    calendar: dict[str, list[dict]] = {}

    for r in publish_records:
        if r.content is None:
            continue

        # 确定该条记录展示在哪天
        item_date = None
        item_type = r.status.value  # "pending" / "success" / "failed"

        if r.status == PublishStatus.PENDING and r.response_data:
            # 有定时排期 → 按排期日期展示
            try:
                data = json.loads(r.response_data)
                scheduled = data.get("scheduled_at", "")
                if scheduled:
                    item_date = scheduled[:10]
            except Exception:
                pass
        elif r.status == PublishStatus.SUCCESS and r.published_at:
            # 已发布 → 按实际发布时间展示
            item_date = r.published_at.strftime("%Y-%m-%d")
        elif r.status == PublishStatus.FAILED:
            # 失败 → 按 published_at（如果有的话）或 created_at 展示
            if r.published_at:
                item_date = r.published_at.strftime("%Y-%m-%d")
            else:
                item_date = r.created_at.strftime("%Y-%m-%d") if r.created_at else None

        if not item_date or not item_date.startswith(f"{year}-{month:02d}"):
            continue

        if item_date not in calendar:
            calendar[item_date] = []

        calendar[item_date].append({
            "id": r.id,
            "content_id": r.content_id,
            "title": r.content.title or r.content.topic or "未命名",
            "type": item_type,
            "platform_code": r.platform_code,
            "content_status": r.content.status.value if r.content.status else "draft",
        })

    # 每天内部按状态排序：failed > pending > success
    _TYPE_ORDER = {"failed": 0, "pending": 1, "success": 2}
    for date_key in calendar:
        calendar[date_key].sort(key=lambda x: _TYPE_ORDER.get(x["type"], 9))

    return {"success": True, "data": {"calendar": calendar, "year": year, "month": month}}


@router.get("/{content_id}", response_model=dict)
def get_content(
    content_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取内容详情"""
    # 显式 user_id 过滤（ORM hook 在某些 FastAPI 执行路径下 ContextVar 传递不可靠）
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id,
    ).first()

    if not content:
        logger.warning(
            "内容不存在",
            content_id=content_id,
            _type="api",
            event_name="content_not_found",
        )
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
            "status": content.status.value if content.status else "unknown",
            "tags": content.tags.split(",") if content.tags else [],
            "view_count": content.view_count or 0,
            "like_count": content.like_count or 0,
            "metadata": json.loads(content.metadata_json) if content.metadata_json else {},
            "versions": [
                {
                    "id": v.id,
                    "version_number": v.version_number,
                    "content_md": v.content_md,
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
    # 显式 user_id 过滤
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id,
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
    # 显式 user_id 过滤
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id,
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


@router.post("/batch-delete", response_model=dict)
def batch_delete_contents(
    request: BatchIdsRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """批量删除内容（软删除，单事务）"""
    # 显式 user_id 过滤
    contents = db.query(Content).filter(
        Content.id.in_(request.ids),
        Content.user_id == current_user.id,
    ).all()

    found_ids = {c.id for c in contents}
    for c in contents:
        c.status = ContentStatus.ARCHIVED
    db.commit()

    return {
        "success": True,
        "data": {
            "deleted": list(found_ids),
            "not_found": [id for id in request.ids if id not in found_ids],
        }
    }


@router.post("/batch-archive", response_model=dict)
def batch_archive_contents(
    request: BatchIdsRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """批量归档内容（单事务）"""
    # 显式 user_id 过滤
    contents = db.query(Content).filter(
        Content.id.in_(request.ids),
        Content.user_id == current_user.id,
    ).all()

    found_ids = {c.id for c in contents}
    for c in contents:
        c.status = ContentStatus.ARCHIVED
    db.commit()

    return {
        "success": True,
        "data": {
            "archived": list(found_ids),
            "not_found": [id for id in request.ids if id not in found_ids],
        }
    }


@router.post("", response_model=dict)
def create_content(
    request: CreateContentRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """直接创建内容（非 AI 生成）"""
    from src.models import ContentVersion

    content = Content(
        user_id=current_user.id,
        title=request.title,
        topic=request.title[:80],
        platform_code=request.platform_code or "xiaohongshu",
        content_md=request.content_md,
        content_html=request.content_md.replace("\n", "<br>"),
        status=ContentStatus.DRAFT,
        tags=",".join(request.tags or []),
    )
    db.add(content)
    db.commit()
    db.refresh(content)

    # 创建第一个版本
    version = ContentVersion(
        content_id=content.id,
        version_number=1,
        content_md=request.content_md,
        content_html=request.content_md.replace("\n", "<br>"),
        change_summary="手动创建",
    )
    db.add(version)
    db.commit()

    return {
        "success": True,
        "data": {
            "id": content.id,
            "title": content.title,
            "created_at": content.created_at.isoformat(),
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
    # 显式 user_id 过滤
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id,
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    if request.content_md is not None:
        content.content_md = request.content_md
        content.content_html = request.content_md.replace("\n", "<br>")
    if request.title is not None:
        content.title = request.title
    if request.tags is not None:
        content.tags = request.tags
    db.commit()

    return {
        "success": True,
        "data": {
            "id": content.id,
            "updated_at": content.updated_at.isoformat(),
        }
    }




@router.post("/{content_id}/check-banned-words", response_model=dict)
def check_banned_words(
    content_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """检测内容中的敏感词"""
    # 显式 user_id 过滤
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id,
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
    # 显式 user_id 过滤
    content = db.query(Content).filter(
        Content.id == content_id,
        Content.user_id == current_user.id,
    ).first()

    if not content:
        raise HTTPException(status_code=404, detail="内容不存在")

    service = ContentService(db)
    try:
        result = service.diff_versions(content_id, v1, v2)
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
