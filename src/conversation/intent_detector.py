"""TriggerExtractor — 从 LLM 回复中提取 [GENERATE] 标记"""

from __future__ import annotations
import json
import re
from typing import Optional

from .schemas import GenerateTrigger

# 匹配 [GENERATE] ... [/GENERATE] 之间的 JSON
_TRIGGER_PATTERN = re.compile(
    r'\[GENERATE\]\s*(\{.*?\})\s*\[/GENERATE\]',
    re.DOTALL,
)


class TriggerExtractor:
    """从 LLM 回复文本中提取生成触发指令"""

    @classmethod
    def extract(cls, text: str) -> Optional[GenerateTrigger]:
        """从文本中提取第一个 [GENERATE] 标记并解析为结构化指令

        Args:
            text: LLM 回复的完整文本

        Returns:
            解析成功的 GenerateTrigger，或 None
        """
        match = _TRIGGER_PATTERN.search(text)
        if not match:
            return None

        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

        # 基本校验
        action = data.get("action", "create")
        if action not in ("create", "revise", "expand"):
            return None

        return GenerateTrigger.from_dict(data)

    @classmethod
    def strip(cls, text: str) -> str:
        """移除文本中的 [GENERATE] 标记及其内容，返回纯显示文本"""
        return _TRIGGER_PATTERN.sub('', text).strip()

    @classmethod
    def has_marker(cls, text: str) -> bool:
        """检查文本中是否包含 [GENERATE] 标记"""
        return bool(_TRIGGER_PATTERN.search(text))
