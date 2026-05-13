from src.adapters.base import (
    PlatformAdapter,
    PlatformContent,
    PublishResult,
    PublishStatus,
    PlatformRegistry,
)


class ZhihuAdapter(PlatformAdapter):
    """
    知乎平台适配器
    MVP 阶段使用 Mock 模式
    """

    platform_code = "zhihu"
    platform_name = "知乎"

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self.is_mock = credentials.get("is_mock", True)

    async def authenticate(self) -> bool:
        """验证凭证有效性"""
        if self.is_mock:
            return True

        # 知乎使用 OAuth2 认证
        # TODO: 实现真实的知乎 OAuth2 认证
        raise NotImplementedError("知乎真实 API 对接待实现")

    async def validate_content(self, content: PlatformContent) -> tuple[bool, str]:
        """验证内容是否符合知乎规范"""
        errors = []

        if not content.title:
            errors.append("标题不能为空")
        elif len(content.title) > 100:
            errors.append("标题不能超过100字")

        if not content.body_md:
            errors.append("正文不能为空")
        elif len(content.body_md) < 100:
            errors.append("正文内容过少，至少100字")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    async def publish(
        self,
        content: PlatformContent,
        publish_mode: str = "draft"
    ) -> PublishResult:
        """发布到知乎"""
        import uuid
        import random
        import asyncio

        # Mock 模式
        if self.is_mock:
            await asyncio.sleep(1)  # 模拟 API 延迟

            if random.random() > 0.1:
                post_id = f"zh_{uuid.uuid4().hex[:8]}"
                return PublishResult(
                    status=PublishStatus.SUCCESS,
                    platform_post_id=post_id,
                    platform_url=f"https://zhuanlan.zhihu.com/p/{post_id}",
                    raw_response={"mock": True, "mode": publish_mode}
                )
            else:
                return PublishResult(
                    status=PublishStatus.FAILED,
                    error_message="模拟网络异常，请重试"
                )

        # 真实 API 调用流程：
        # 1. 获取 access_token
        # 2. 调用回答/文章发布 API
        raise NotImplementedError("知乎真实 API 对接待实现")

    def to_platform_format(self, content: PlatformContent) -> dict:
        """转换为知乎 API 格式"""
        return {
            "title": content.title,
            "content": content.body_md,
            "topics": content.tags[:5],  # 知乎最多5个话题
            "is_anonymous": content.extra.get("is_anonymous", False),
        }


# 注册适配器
PlatformRegistry.register(ZhihuAdapter)
