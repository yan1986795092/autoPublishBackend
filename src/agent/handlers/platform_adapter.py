"""PlatformAdapter Agent — 多平台格式转换 (小红书/公众号/知乎)"""

import logging
import os
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _load_credentials(target_platform: str) -> dict:
    """从环境变量加载平台凭证，兜底 mock 模式"""
    env_map = {
        "xiaohongshu": ("XHS_COOKIE",),
        "wechat": ("WECHAT_APP_ID", "WECHAT_APP_SECRET"),
        "zhihu": ("ZHIHU_COOKIE",),
    }
    keys = env_map.get(target_platform, [])
    creds = {k.lower(): os.getenv(k, "") for k in keys}
    if any(creds.values()):
        creds["is_mock"] = False
        return creds
    return {"is_mock": True}


async def handle_platform_adapter(task_input: dict, llm: ChatOpenAI, mcp_tools: list | None = None) -> dict:
    target_platform = task_input.get("platform", task_input.get("target_platform", "xiaohongshu"))
    content = task_input.get("content", "")
    if not content:
        dep_context = task_input.get("_context", {})
        for cdata in dep_context.values():
            if cdata.get("content"):
                content = cdata["content"]
                break

    try:
        from src.adapters.xiaohongshu import XiaohongshuAdapter
        from src.adapters.wechat import WechatAdapter
        from src.adapters.zhihu import ZhihuAdapter
        from src.adapters.base import PlatformContent

        PLATFORM_ADAPTERS = {
            "xiaohongshu": XiaohongshuAdapter,
            "wechat": WechatAdapter,
            "zhihu": ZhihuAdapter,
        }

        adapter_cls = PLATFORM_ADAPTERS.get(target_platform)
        if adapter_cls:
            adapter = adapter_cls(_load_credentials(target_platform))
            platform_content = PlatformContent(
                title=content[:60].strip().split("\n")[0],
                body_md=content,
                body_html="",
            )
            adapted = adapter.to_platform_format(platform_content)
            adapted_text = adapted.get("desc") or adapted.get("content") or adapted.get("body") or content
            return {
                "status": "completed", "agent_type": "PlatformAdapter",
                "output": {
                    "content": adapted_text,
                    "platform": target_platform,
                    "format_notes": adapted.get("format_notes", []),
                    "hashtag_suggestions": adapted.get("tag_list", []),
                },
                "tokens_used": 0,
                "thinking_steps": [{"agent": "PlatformAdapter", "action": "adapt", "reasoning": f"Adapting for {target_platform}", "output_summary": f"Adapted for {target_platform}", "status": "done"}],
            }
    except Exception as e:
        logger.warning(f"PlatformAdapter 处理 {target_platform} 失败: {e}")

    return {
        "status": "completed", "agent_type": "PlatformAdapter",
        "output": {"content": content, "platform": target_platform, "note": "PlatformAdapter 降级：返回原始内容"},
        "tokens_used": 0,
        "thinking_steps": [{"agent": "PlatformAdapter", "action": "adapt", "reasoning": f"Adapting for {target_platform} (fallback)", "output_summary": f"Returned original content for {target_platform}", "status": "done"}],
    }
