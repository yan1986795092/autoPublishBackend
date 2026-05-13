from pydantic import BaseModel, Field
from typing import Optional


class PresetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    platform_code: Optional[str] = None
    template_id: Optional[str] = None


class PresetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict] = None
    platform_code: Optional[str] = None
    template_id: Optional[str] = None


class PresetListItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    platform_code: Optional[str] = None
    use_count: int = 0
    created_at: str


class PresetDetail(PresetListItem):
    parameters: dict
    template_id: Optional[str] = None
