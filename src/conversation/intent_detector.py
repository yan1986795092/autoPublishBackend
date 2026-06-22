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

# 越权自写检测：识别 LLM 输出的元描述行（## 思考 / ## 正文 / ## 标题 等）
_UNAUTH_PATTERN = re.compile(
    r'(^|\n)#{1,3}\s*(思考|正文|标题|引言|结论|章节)\s*[:：]?',
    re.MULTILINE,
)
# 多个 Markdown 标题
_MULTI_HEADER_PATTERN = re.compile(r'(^|\n)#{1,3}\s+\S', re.MULTILINE)
# "标题：" / "引言：" 形式的伪元描述
_LABEL_COLON_PATTERN = re.compile(r'(标题|引言)\s*[:：]\s*\S')


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

        raw = match.group(1).strip()
        # 兼容 LLM 输出的双大括号 {{...}} 格式
        if raw.startswith("{{") and raw.endswith("}}"):
            raw = raw[1:-1]
        try:
            data = json.loads(raw)
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

    @classmethod
    def looks_like_unauthorized_write(cls, text: str, has_marker: bool) -> bool:
        """检测 LLM 是否在 trigger 之外越权写了正文

        判断条件（任一）：
        1. 文本长度 > 300 且包含 2 个以上 Markdown 标题
        2. 文本以"## 思考：" / "## 正文：" 等元描述开头
        3. 文本含有关键词（"标题："/"引言："）且无 trigger 标记
        4. 出现 2 个以上 ## 思考/## 正文 等元描述行
        """
        if not text:
            return False
        if has_marker:
            return False
        # 条件 2: 元描述开头
        if _UNAUTH_PATTERN.match(text):
            return True
        # 条件 4: 出现 >= 2 个元描述行
        if len(_UNAUTH_PATTERN.findall(text)) >= 2:
            return True
        # 条件 1: 长文本 + 多个 Markdown 标题
        if len(text) > 300 and len(_MULTI_HEADER_PATTERN.findall(text)) >= 2:
            return True
        # 条件 3: "标题：xxx" / "引言：xxx" 形式
        if _LABEL_COLON_PATTERN.search(text):
            return True
        return False
