from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from fastapi import HTTPException, status

from src.agent.workflow.task_queue import AsyncTask
from src.models import ChatConversation, User


def ensure_task_owner(
    task: AsyncTask | None,
    current_user: User,
    not_found_detail: str = "未找到任务",
) -> AsyncTask:
    if task is None or task.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail)
    return task


def ensure_conversation_owner(
    conversation: ChatConversation | None,
    current_user: User,
    not_found_detail: str = "对话不存在",
) -> ChatConversation:
    if conversation is None or conversation.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=not_found_detail)
    return conversation


def validate_public_http_url(url: str) -> str:
    safe_url = (url or "").strip()
    parsed = urlparse(safe_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 http/https 公网 URL")

    host = parsed.hostname
    if host.lower() == "localhost":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不允许访问本地或内网地址")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return safe_url

    if any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    ):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不允许访问本地或内网地址")

    return safe_url
