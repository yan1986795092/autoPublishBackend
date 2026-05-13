from src.adapters.base import (
    PlatformAdapter,
    PlatformContent,
    PublishResult,
    PublishStatus,
    PlatformRegistry,
)


class WechatAdapter(PlatformAdapter):
    """
    微信公众号平台适配器
    MVP 阶段使用 Mock 模式
    """

    platform_code = "wechat"
    platform_name = "微信公众号"

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self.is_mock = credentials.get("is_mock", True)

    async def authenticate(self) -> bool:
        """验证凭证有效性"""
        if self.is_mock:
            return True

        # 微信使用 AppID + AppSecret 获取 access_token
        # TODO: 实现真实的微信公众号认证
        raise NotImplementedError("微信公众号真实 API 对接待实现")

    async def validate_content(self, content: PlatformContent) -> tuple[bool, str]:
        """验证内容是否符合微信公众号规范"""
        errors = []

        if not content.title:
            errors.append("标题不能为空")
        elif len(content.title) > 64:
            errors.append("标题不能超过64字")

        if not content.body_md:
            errors.append("正文不能为空")
        elif len(content.body_md) < 50:
            errors.append("正文内容过少，至少50字")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    async def publish(
        self,
        content: PlatformContent,
        publish_mode: str = "draft"
    ) -> PublishResult:
        """发布到微信公众号"""
        import uuid
        import random
        import asyncio

        # Mock 模式
        if self.is_mock:
            await asyncio.sleep(2)  # 模拟 API 延迟

            if random.random() > 0.1:
                post_id = f"wx_{uuid.uuid4().hex[:8]}"
                return PublishResult(
                    status=PublishStatus.SUCCESS,
                    platform_post_id=post_id,
                    platform_url=f"https://mp.weixin.qq.com/s/{post_id}",
                    raw_response={"mock": True, "mode": publish_mode}
                )
            else:
                return PublishResult(
                    status=PublishStatus.FAILED,
                    error_message="模拟网络异常，请重试"
                )

        # 真实 API 调用流程：
        # 1. 获取 access_token
        # 2. 上传封面图获取 media_id
        # 3. 上传正文图片获取 media_id
        # 4. 创建图文消息
        # 5. 发布或存为草稿
        raise NotImplementedError("微信公众号真实 API 对接待实现")

    def to_platform_format(self, content: PlatformContent) -> dict:
        """转换为微信公众号 API 格式"""
        return {
            "title": content.title,
            "content": content.body_html,
            "thumb_media_id": content.extra.get("thumb_media_id"),
            "digest": content.body_md[:120],  # 摘要
            "author": content.extra.get("author", ""),
            "need_open_comment": content.extra.get("need_open_comment", 1),
            "only_fans_can_comment": content.extra.get("only_fans_can_comment", 0),
        }


# 注册适配器
PlatformRegistry.register(WechatAdapter)
