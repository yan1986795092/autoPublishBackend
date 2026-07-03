"""Agent 类型 → 前端 Stage 映射表

由 registry.py 驱动，新增 Agent 只需注册卡片，无需修改此文件。
保持与旧代码的 API 兼容（get_stage_info / get_stage_label / get_stage_icon）。
"""

from .registry import get

_DEFAULT_STAGE = {"label": "处理中", "icon": "⚙️", "color": "gray"}


def get_stage_label(agent_type: str) -> str:
    card = get(agent_type)
    return card.stage_label if card else _DEFAULT_STAGE["label"]


def get_stage_icon(agent_type: str) -> str:
    card = get(agent_type)
    return card.stage_icon if card else _DEFAULT_STAGE["icon"]
