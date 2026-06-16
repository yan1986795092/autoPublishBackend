"""Agent 类型 → 前端 Stage 映射表

由 registry.py 驱动，新增 Agent 只需注册卡片，无需修改此文件。
保持与旧代码的 API 兼容（get_stage_info / get_stage_label / get_stage_icon）。
"""

from .registry import get

_DEFAULT_STAGE = {"label": "处理中", "icon": "⚙️", "color": "gray"}


def get_stage_info(agent_type: str) -> dict:
    """获取 Agent 类型对应的 Stage 信息，未知类型返回 fallback"""
    card = get(agent_type)
    if card:
        return {"label": card.stage_label, "icon": card.stage_icon, "color": card.stage_color}
    return _DEFAULT_STAGE


def get_stage_label(agent_type: str) -> str:
    card = get(agent_type)
    return card.stage_label if card else _DEFAULT_STAGE["label"]


def get_stage_icon(agent_type: str) -> str:
    card = get(agent_type)
    return card.stage_icon if card else _DEFAULT_STAGE["icon"]
