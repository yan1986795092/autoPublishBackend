from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import json
from uuid import uuid4

from src.database import get_db
from src.models import Preset
from src.api.auth import get_current_user
from src.schemas.preset import PresetCreate, PresetUpdate

router = APIRouter(prefix="/presets", tags=["预设"])


@router.get("", response_model=dict)
def list_presets(
    platform_code: str | None = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取用户预设列表"""
    # ORM Event Hook 自动按 tenant 过滤 user_id
    query = db.query(Preset)
    if platform_code:
        query = query.filter(Preset.platform_code == platform_code)
    presets = query.order_by(Preset.updated_at.desc()).all()
    return {
        "success": True,
        "data": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "platform_code": p.platform_code,
                "use_count": p.use_count,
                "created_at": p.created_at.isoformat(),
            }
            for p in presets
        ],
    }


@router.post("", response_model=dict)
def create_preset(
    data: PresetCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建预设"""
    preset = Preset(
        id=str(uuid4()),
        user_id=current_user.id,
        name=data.name,
        description=data.description,
        parameters_json=json.dumps(data.parameters),
        platform_code=data.platform_code,
        template_id=data.template_id,
    )
    db.add(preset)
    db.commit()
    db.refresh(preset)
    return {
        "success": True,
        "data": {
            "id": preset.id,
            "name": preset.name,
        },
    }


@router.get("/{preset_id}", response_model=dict)
def get_preset(
    preset_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取预设详情"""
    # ORM Event Hook 自动按 tenant 过滤 user_id
    preset = db.query(Preset).filter(
        Preset.id == preset_id,
    ).first()
    if not preset:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="预设不存在")
    return {
        "success": True,
        "data": {
            "id": preset.id,
            "name": preset.name,
            "description": preset.description,
            "parameters": json.loads(preset.parameters_json),
            "platform_code": preset.platform_code,
            "template_id": preset.template_id,
            "use_count": preset.use_count,
            "created_at": preset.created_at.isoformat(),
        },
    }


@router.put("/{preset_id}", response_model=dict)
def update_preset(
    preset_id: str,
    data: PresetUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新预设"""
    # ORM Event Hook 自动按 tenant 过滤 user_id
    preset = db.query(Preset).filter(
        Preset.id == preset_id,
    ).first()
    if not preset:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="预设不存在")
    if data.name is not None:
        preset.name = data.name
    if data.description is not None:
        preset.description = data.description
    if data.parameters is not None:
        preset.parameters_json = json.dumps(data.parameters)
    if data.platform_code is not None:
        preset.platform_code = data.platform_code
    if data.template_id is not None:
        preset.template_id = data.template_id
    db.commit()
    return {"success": True, "data": {"id": preset.id}}


@router.delete("/{preset_id}", response_model=dict)
def delete_preset(
    preset_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除预设"""
    # ORM Event Hook 自动按 tenant 过滤 user_id
    preset = db.query(Preset).filter(
        Preset.id == preset_id,
    ).first()
    if not preset:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="预设不存在")
    db.delete(preset)
    db.commit()
    return {"success": True, "data": None}
