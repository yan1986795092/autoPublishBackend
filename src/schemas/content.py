from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="内容主题")
    platform_code: str = Field(default="xiaohongshu", description="目标平台代码")
    template_id: Optional[str] = Field(None, description="使用的模板 ID")
    model: Optional[str] = Field(None, description="指定的大模型")
    parameters: Optional[dict] = Field(
        default_factory=lambda: {
            "length": "medium",
            "creativity": 0.7,
            "tone": "friendly"
        },
        description="生成参数"
    )


class BatchGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500, description="内容主题")
    platforms: list[str] = Field(default=["xiaohongshu"], description="目标平台列表")
    model: Optional[str] = Field(None, description="指定的大模型")
    parameters: Optional[dict] = Field(
        default_factory=lambda: {
            "length": "medium",
            "creativity": 0.7,
            "tone": "friendly"
        },
        description="生成参数"
    )


class GenerateMetadata(BaseModel):
    model: str
    tokens_used: int
    tokens_cost_usd: float
    generation_time_ms: int
    template_id: Optional[str] = None
    version: int = 1


class GenerateResponse(BaseModel):
    content_id: str
    title: str
    content_md: str
    content_html: str
    metadata: GenerateMetadata


class ContentListItem(BaseModel):
    id: str
    title: Optional[str] = None
    topic: Optional[str] = None
    platform_code: Optional[str] = None
    status: str
    tags: list[str] = []
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class PaginationInfo(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class ContentListResponse(BaseModel):
    items: list[ContentListItem]
    pagination: PaginationInfo


class ContentVersionItem(BaseModel):
    id: str
    version_number: int
    change_summary: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


class ContentDetailResponse(BaseModel):
    id: str
    title: Optional[str] = None
    topic: Optional[str] = None
    platform_code: Optional[str] = None
    content_md: Optional[str] = None
    content_html: Optional[str] = None
    status: str
    tags: list[str] = []
    metadata: dict = {}
    versions: list[ContentVersionItem] = []
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class UpdateContentRequest(BaseModel):
    title: Optional[str] = None
    content_md: Optional[str] = None
    tags: Optional[str] = None
    change_summary: Optional[str] = None


class UpdateContentResponse(BaseModel):
    id: str
    version_number: int
    updated_at: str
