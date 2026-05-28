from src.adapters.base import (
    PlatformAdapter,
    PlatformContent,
    PublishResult,
    PublishStatus,
    PlatformRegistry,
)


class XiaohongshuAdapter(PlatformAdapter):
    """
    小红书平台适配器
    MVP 阶段使用 Mock 模式
    """

    platform_code = "xiaohongshu"
    platform_name = "小红书"

    def __init__(self, credentials: dict):
        super().__init__(credentials)
        self.is_mock = credentials.get("is_mock", True)

    async def authenticate(self) -> bool:
        """验证凭证有效性"""
        if self.is_mock:
            return True

        # 真实 OAuth2 认证
        # TODO: 实现真实 OAuth2 认证
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://creator.xiaohongshu.com/api/va/auth/token",
                json={"access_token": self.credentials.get("access_token")}
            )
            return resp.status_code == 200

    async def validate_content(self, content: PlatformContent) -> tuple[bool, str]:
        """验证内容是否符合小红书规范"""
        errors = []

        if not content.title:
            errors.append("标题不能为空")
        elif len(content.title) > 20:
            errors.append("标题不能超过20字")

        if not content.body_md:
            errors.append("正文不能为空")
        elif len(content.body_md) < 50:
            errors.append("正文内容过少，至少50字")

        if len(content.tags) > 10:
            errors.append("标签最多10个")

        # 敏感词检测
        combined = f"{content.title} {content.body_md}".lower()
        sensitive_words = [
            "赌博", "彩票", "毒品", "枪支", "色情", "低俗",
            "政治", "领导人", "敏感", "暴力", "恐怖",
        ]
        found = [w for w in sensitive_words if w in combined]
        if found:
            errors.append(f"内容包含敏感词: {', '.join(found)}")

        if errors:
            return False, "; ".join(errors)
        return True, ""

    async def publish(
        self,
        content: PlatformContent,
        publish_mode: str = "draft"
    ) -> PublishResult:
        """发布到小红书"""
        import uuid
        import random
        import asyncio

        # Mock 模式
        if self.is_mock:
            await asyncio.sleep(1.5)  # 模拟 API 延迟

            if random.random() > 0.1:
                post_id = f"xhs_{uuid.uuid4().hex[:8]}"
                return PublishResult(
                    status=PublishStatus.SUCCESS,
                    platform_post_id=post_id,
                    platform_url=f"https://www.xiaohongshu.com/explore/{post_id}",
                    raw_response={"mock": True, "mode": publish_mode}
                )
            else:
                return PublishResult(
                    status=PublishStatus.FAILED,
                    error_message="模拟网络异常，请重试"
                )

        # 真实 API 调用
        validated, err = await self.validate_content(content)
        if not validated:
            return PublishResult(status=PublishStatus.FAILED, error_message=err)

        xhs_content = self.to_platform_format(content)

        # TODO: 实现真实小红书 API 调用
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post(
        #         "https://creator.xiaohongshu.com/api/va/content/publish",
        #         headers={"Authorization": f"Bearer {self.credentials['access_token']}"},
        #         json={
        #             "note_content": xhs_content,
        #             "post_mode": publish_mode
        #         },
        #         timeout=30.0
        #     )
        #     if resp.status_code == 200:
        #         data = resp.json()
        #         return PublishResult(
        #             status=PublishStatus.SUCCESS,
        #             platform_post_id=data.get("note_id"),
        #             platform_url=data.get("note_url"),
        #             raw_response=data
        #         )
        #     else:
        #         return PublishResult(
        #             status=PublishStatus.FAILED,
        #             error_message=resp.text
        #         )

        raise NotImplementedError("真实 API 对接待实现")

    def to_platform_format(self, content: PlatformContent) -> dict:
        """转换为小红书 API 格式"""
        return {
            "title": content.title,
            "desc": content.body_md,
            "image_list": [content.cover_image_url] if content.cover_image_url else [],
            "tag_list": content.tags,
            "at_user_list": content.extra.get("at_users", []),
        }


# 注册适配器
PlatformRegistry.register(XiaohongshuAdapter)
