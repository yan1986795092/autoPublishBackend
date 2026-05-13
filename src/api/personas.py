from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4

from src.database import get_db
from src.models import Persona
from src.api.auth import get_current_user
from src.schemas.persona import PersonaCreate, PersonaUpdate

router = APIRouter(prefix="/personas", tags=["人设"])


@router.get("", response_model=dict)
def list_personas(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取用户人设列表"""
    personas = db.query(Persona).filter(
        Persona.user_id == current_user.id,
    ).order_by(Persona.updated_at.desc()).all()
    return {
        "success": True,
        "data": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "tone": p.tone,
                "style_notes": p.style_notes,
                "is_active": p.is_active,
                "use_count": p.use_count,
                "created_at": p.created_at.isoformat(),
            }
            for p in personas
        ],
    }


@router.post("", response_model=dict)
def create_persona(
    data: PersonaCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建人设"""
    persona = Persona(
        id=str(uuid4()),
        user_id=current_user.id,
        name=data.name,
        description=data.description,
        tone=data.tone,
        style_notes=data.style_notes,
    )
    db.add(persona)
    db.commit()
    db.refresh(persona)
    return {
        "success": True,
        "data": {
            "id": persona.id,
            "name": persona.name,
        },
    }


@router.get("/{persona_id}", response_model=dict)
def get_persona(
    persona_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取人设详情"""
    persona = db.query(Persona).filter(
        Persona.id == persona_id,
        Persona.user_id == current_user.id,
    ).first()
    if not persona:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="人设不存在")
    return {
        "success": True,
        "data": {
            "id": persona.id,
            "name": persona.name,
            "description": persona.description,
            "tone": persona.tone,
            "style_notes": persona.style_notes,
            "is_active": persona.is_active,
            "use_count": persona.use_count,
            "created_at": persona.created_at.isoformat(),
        },
    }


@router.put("/{persona_id}", response_model=dict)
def update_persona(
    persona_id: str,
    data: PersonaUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新人设"""
    persona = db.query(Persona).filter(
        Persona.id == persona_id,
        Persona.user_id == current_user.id,
    ).first()
    if not persona:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="人设不存在")
    if data.name is not None:
        persona.name = data.name
    if data.description is not None:
        persona.description = data.description
    if data.tone is not None:
        persona.tone = data.tone
    if data.style_notes is not None:
        persona.style_notes = data.style_notes
    if data.is_active is not None:
        persona.is_active = data.is_active
    db.commit()
    return {"success": True, "data": {"id": persona.id}}


@router.delete("/{persona_id}", response_model=dict)
def delete_persona(
    persona_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除人设"""
    persona = db.query(Persona).filter(
        Persona.id == persona_id,
        Persona.user_id == current_user.id,
    ).first()
    if not persona:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="人设不存在")
    db.delete(persona)
    db.commit()
    return {"success": True, "data": None}
