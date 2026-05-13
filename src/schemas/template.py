from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TemplateVariable(BaseModel):
    name: str
    type: str = "string"
    required: bool = False
    label: Optional[str] = None
    default: Optional[str] = None
    options: Optional[list[str]] = None


class TemplateListItem(BaseModel):
    id: str
    name: str
    platform_code: Optional[str] = None
    description: Optional[str] = None
    is_public: bool = False
    use_count: int = 0

    class Config:
        from_attributes = True


class TemplateDetail(BaseModel):
    id: str
    name: str
    platform_code: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: str
    variables: list[TemplateVariable] = []
    model_preferred: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    is_public: bool = False
    use_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CreateTemplateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    platform_code: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    user_template: str = Field(..., min_length=1)
    variables: Optional[list[TemplateVariable]] = []
    model_preferred: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=16000)
    is_public: bool = False
