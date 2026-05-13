from pydantic import BaseModel, Field
from typing import Optional


class PublishRequest(BaseModel):
    content_id: str = Field(..., description="内容 ID")
    platform_code: str = Field(..., description="目标平台代码")
    publish_mode: str = Field(default="draft", pattern="^(draft|publish)$", description="发布模式")
    schedule_time: Optional[str] = Field(None, description="定时发布时间 (ISO 格式)")


class PublishResponse(BaseModel):
    publish_record_id: str
    status: str
    platform: str
    mode: str
    platform_post_id: Optional[str] = None
    platform_url: Optional[str] = None
    estimated_time_seconds: Optional[int] = None


class PlatformInfo(BaseModel):
    code: str
    name: str
    icon_url: Optional[str] = None
    is_mock: bool = True
    is_active: bool = True

    class Config:
        from_attributes = True


class PlatformListResponse(BaseModel):
    platforms: list[PlatformInfo]
