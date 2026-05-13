from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
import json

from src.database import get_db
from src.models import PromptTemplate
from src.api.auth import get_current_user

router = APIRouter(prefix="/templates", tags=["模板"])


@router.get("", response_model=dict)
def list_templates(
    platform_code: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取模板列表（系统模板 + 用户自定义模板）"""
    query = db.query(PromptTemplate).filter(
        (PromptTemplate.user_id == current_user.id) | (PromptTemplate.is_public == True)
    )

    if platform_code:
        query = query.filter(PromptTemplate.platform_code == platform_code)

    templates = query.order_by(PromptTemplate.use_count.desc()).all()

    return {
        "success": True,
        "data": [
            {
                "id": t.id,
                "name": t.name,
                "platform_code": t.platform_code,
                "description": t.description,
                "is_public": t.is_public,
                "use_count": t.use_count,
            }
            for t in templates
        ]
    }


@router.get("/{template_id}", response_model=dict)
def get_template(
    template_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取模板详情"""
    template = db.query(PromptTemplate).filter(
        PromptTemplate.id == template_id,
        (PromptTemplate.user_id == current_user.id) | (PromptTemplate.is_public == True)
    ).first()

    if not template:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="模板不存在")

    return {
        "success": True,
        "data": {
            "id": template.id,
            "name": template.name,
            "platform_code": template.platform_code,
            "description": template.description,
            "system_prompt": template.system_prompt,
            "user_template": template.user_template,
            "variables": json.loads(template.variables_json) if template.variables_json else [],
            "model_preferred": template.model_preferred,
            "temperature": template.temperature,
            "max_tokens": template.max_tokens,
            "is_public": template.is_public,
            "use_count": template.use_count,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
        }
    }


@router.post("", response_model=dict)
def create_template(
    data: dict,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """创建自定义模板"""
    from uuid import uuid4
    from datetime import datetime, timezone

    template = PromptTemplate(
        id=str(uuid4()),
        user_id=current_user.id,
        name=data.get("name", "未命名模板"),
        platform_code=data.get("platform_code"),
        description=data.get("description"),
        system_prompt=data.get("system_prompt"),
        user_template=data.get("user_template"),
        variables_json=json.dumps(data.get("variables", [])),
        model_preferred=data.get("model_preferred"),
        temperature=data.get("temperature", 0.7),
        max_tokens=data.get("max_tokens", 2000),
        is_public=data.get("is_public", False),
    )
    db.add(template)
    db.commit()
    db.refresh(template)

    return {
        "success": True,
        "data": {
            "id": template.id,
            "name": template.name,
            "platform_code": template.platform_code,
        }
    }


@router.put("/{template_id}", response_model=dict)
def update_template(
    template_id: str,
    data: dict,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新自定义模板（仅限自己的）"""
    template = db.query(PromptTemplate).filter(
        PromptTemplate.id == template_id,
        PromptTemplate.user_id == current_user.id,
    ).first()

    if not template:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="模板不存在或无权限")

    for field in ["name", "platform_code", "description", "system_prompt", "user_template",
                  "model_preferred", "temperature", "max_tokens", "is_public"]:
        if field in data:
            setattr(template, field, data[field])

    if "variables" in data:
        template.variables_json = json.dumps(data["variables"])

    db.commit()
    db.refresh(template)

    return {"success": True, "data": {"id": template.id}}


@router.delete("/{template_id}", response_model=dict)
def delete_template(
    template_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除自定义模板（仅限自己的）"""
    template = db.query(PromptTemplate).filter(
        PromptTemplate.id == template_id,
        PromptTemplate.user_id == current_user.id,
    ).first()

    if not template:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="模板不存在或无权限")

    db.delete(template)
    db.commit()

    return {"success": True, "data": None}
