from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
import json


class PublishMode(str, Enum):
    DRAFT = "draft"
    PUBLISH = "publish"


class PublishStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class PlatformContent:
    """平台无关的内容格式"""
    title: str
    body_md: str
    body_html: str
    cover_image_url: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class PublishResult:
    """统一的发布结果"""
    status: PublishStatus
    platform_post_id: Optional[str] = None
    platform_url: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Optional[dict] = None


class PlatformAdapter(ABC):
    """
    平台适配器基类
    所有平台适配器必须实现此接口
    """

    platform_code: str = "base"
    platform_name: str = "未知平台"

    def __init__(self, credentials: dict):
        """
        初始化适配器

        Args:
            credentials: 包含平台认证信息的字典，如 access_token, cookies 等
        """
        self.credentials = credentials
        self.is_mock = credentials.get("is_mock", True)

    @abstractmethod
    async def authenticate(self) -> bool:
        """
        验证凭证有效性

        Returns:
            bool: 凭证是否有效
        """
        pass

    @abstractmethod
    async def validate_content(self, content: PlatformContent) -> tuple[bool, str]:
        """
        验证内容是否符合平台规范

        Args:
            content: 待验证的内容

        Returns:
            (是否通过, 错误信息)
        """
        pass

    @abstractmethod
    async def publish(
        self,
        content: PlatformContent,
        publish_mode: str = "draft"
    ) -> PublishResult:
        """
        发布内容到平台

        Args:
            content: 待发布的内容
            publish_mode: 发布模式 ("draft" 或 "publish")

        Returns:
            PublishResult: 发布结果
        """
        pass

    async def get_publish_status(self, platform_post_id: str) -> PublishResult:
        """
        查询发布状态（可选实现）

        Args:
            platform_post_id: 平台返回的帖子 ID

        Returns:
            PublishResult: 查询结果
        """
        return PublishResult(
            status=PublishStatus.SUCCESS,
            platform_post_id=platform_post_id,
        )

    def to_platform_format(self, content: PlatformContent) -> dict:
        """
        将通用格式转换为平台特定格式

        Args:
            content: 通用格式内容

        Returns:
            dict: 平台特定格式的字典
        """
        return {
            "title": content.title,
            "body": content.body_md,
            "cover_image": content.cover_image_url,
            "tags": content.tags,
        }


class MockPlatformAdapter(PlatformAdapter):
    """
    Mock 平台适配器基类
    用于 MVP 阶段的模拟发布
    """

    platform_code = "mock"
    platform_name = "Mock 平台"

    async def authenticate(self) -> bool:
        return True

    async def validate_content(self, content: PlatformContent) -> tuple[bool, str]:
        if not content.title:
            return False, "标题不能为空"
        if len(content.body_md) < 10:
            return False, "正文内容过少"
        return True, ""

    async def publish(
        self,
        content: PlatformContent,
        publish_mode: str = "draft"
    ) -> PublishResult:
        import uuid
        import random
        import asyncio

        # 模拟 API 延迟
        await asyncio.sleep(1)

        # 模拟 90% 成功率
        if random.random() > 0.1:
            post_id = f"mock_{uuid.uuid4().hex[:8]}"
            return PublishResult(
                status=PublishStatus.SUCCESS,
                platform_post_id=post_id,
                platform_url=f"https://mock.example.com/post/{post_id}",
                raw_response={"mock": True, "mode": publish_mode}
            )
        else:
            return PublishResult(
                status=PublishStatus.FAILED,
                error_message="模拟网络异常，请重试"
            )


class PlatformRegistry:
    """
    平台适配器注册中心
    负责管理所有平台适配器
    """

    _adapters: dict[str, type[PlatformAdapter]] = {}

    @classmethod
    def register(cls, adapter_class: type[PlatformAdapter]):
        """注册平台适配器"""
        cls._adapters[adapter_class.platform_code] = adapter_class

    @classmethod
    def get_adapter(cls, platform_code: str, credentials: dict) -> PlatformAdapter:
        """获取平台适配器实例"""
        adapter_class = cls._adapters.get(platform_code)
        if not adapter_class:
            raise ValueError(f"不支持的平台: {platform_code}")
        return adapter_class(credentials)

    @classmethod
    def list_platforms(cls) -> list[dict]:
        """列出所有已注册的平台"""
        return [
            {
                "code": cls._adapters[code].platform_code,
                "name": cls._adapters[code].platform_name,
            }
            for code in cls._adapters
        ]

    @classmethod
    def is_registered(cls, platform_code: str) -> bool:
        """检查平台是否已注册"""
        return platform_code in cls._adapters
