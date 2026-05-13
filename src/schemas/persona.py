from pydantic import BaseModel, Field
from typing import Optional


class PersonaCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    tone: Optional[str] = None
    style_notes: Optional[str] = None


class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tone: Optional[str] = None
    style_notes: Optional[str] = None
    is_active: Optional[bool] = None
