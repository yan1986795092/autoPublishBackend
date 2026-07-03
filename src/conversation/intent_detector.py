"""对话 LLM 回复意图识别器

只保留两类判断：
- looks_like_unauthorized_write：检测 LLM 是否在文本里越权写了正文（## 标题、## 思考等）
- looks_like_followup_question：检测 LLM 是否在主动追问（信息不足）

⚠️ 已移除 [GENERATE]{json}[/GENERATE] 文本标记的提取逻辑——
  生成触发统一通过 OpenAI function calling (trigger_generation 工具) 完成。
"""

from __future__ import annotations

import re


# ─────────────── 越权自写检测 ───────────────

# 越权自写检测：识别 LLM 输出的元描述行（## 思考 / ## 正文 / ## 标题 等）
_UNAUTH_PATTERN = re.compile(
    r'(^|\n)#{1,3}\s*(思考|正文|标题|引言|结论|章节)\s*[:：]?',
    re.MULTILINE,
)
# 多个 Markdown 标题
_MULTI_HEADER_PATTERN = re.compile(r'(^|\n)#{1,3}\s+\S', re.MULTILINE)
# "标题：" / "引言：" 形式的伪元描述
_LABEL_COLON_PATTERN = re.compile(r'(标题|引言)\s*[:：]\s*\S')


# ─────────────── 追问检测 ───────────────

# 对话 LLM 表达"信息不足/需要追问"时的语言特征
# 这些短语都意味着 LLM 主动选择等待用户补充信息，**不应**被兜底分支覆盖
_FOLLOWUP_QUESTION_PATTERNS = [
    re.compile(r'请(先|您|你|简单|具体|详细)?(告诉|说|讲|补充|提供|确认|回答|回复)'),
    re.compile(r'(想|要|需要|先|请问)(了解|知道|确认|明确|弄清|听听)'),
    re.compile(r'(简单|具体|详细)?(说说|聊聊|讲讲|描述|介绍|分享)'),
    re.compile(r'(什么|哪[个些]|哪[里儿])\s*[:？?]'),
    re.compile(r'[？\?]\s*$'),  # 句末问号（至少有 1 个）
    re.compile(r'(可以|能|方便)?(说一?下|说一?句|聊一?下|告诉我)'),
    re.compile(r'(您|你)(看|觉得|想要|想|希望|打算|准备)'),
    re.compile(r'(目标|受[众众]|读者|平台|风格|主题)[^。]*?(?:是|在|发|呢|吗|？|\?)'),
    re.compile(r'我们(再|来|一起|一步|慢慢)'),
    re.compile(r'(再|继续|进一步)(讨论|聊|问|交流|打磨|确认)'),
]
_FOLLOWUP_QUESTION_MIN_HITS = 2  # 至少命中 2 个特征才认定为追问


class IntentDetector:
    """对话 LLM 回复文本意图识别器

    v3 改造后已移除 [GENERATE] 文本标记相关方法。
    生成触发统一通过 trigger_generation 工具调用完成，
    本类只保留两类纯文本意图判断。
    """

    @classmethod
    def looks_like_unauthorized_write(cls, text: str) -> bool:
        """检测 LLM 是否在 trigger 之外越权写了正文

        判断条件（任一）：
        1. 文本以"## 思考：" / "## 正文：" 等元描述开头
        2. 文本含"标题："/"引言：" 等关键词
        3. 出现 2 个以上 ## 思考/## 正文 等元描述行
        4. 文本长度 > 300 且包含 2 个以上 Markdown 标题
        """
        if not text:
            return False
        # 条件 1: 元描述开头
        if _UNAUTH_PATTERN.match(text):
            return True
        # 条件 3: 出现 >= 2 个元描述行
        if len(_UNAUTH_PATTERN.findall(text)) >= 2:
            return True
        # 条件 4: 长文本 + 多个 Markdown 标题
        if len(text) > 300 and len(_MULTI_HEADER_PATTERN.findall(text)) >= 2:
            return True
        # 条件 2: "标题：xxx" / "引言：xxx" 形式
        if _LABEL_COLON_PATTERN.search(text):
            return True
        return False

    @classmethod
    def looks_like_followup_question(cls, text: str) -> bool:
        """检测对话 LLM 是否在主动追问（信息不足）

        用于"用户消息含生成关键词但 LLM 没调用 trigger_generation"场景下，
        判断 LLM 是不是在正确遵守规则 A（在追问而非茫然 / 越权自写）。

        判定：出现 >= 2 个追问语言特征（如"请告诉我""您的平台是""简单说说"等）。
        """
        if not text or len(text.strip()) < 10:
            return False
        hits = sum(1 for p in _FOLLOWUP_QUESTION_PATTERNS if p.search(text))
        return hits >= _FOLLOWUP_QUESTION_MIN_HITS


# 向后兼容：旧代码仍可能 import TriggerExtractor 作为类名。
# 旧版 TriggerExtractor 提供的 [GENERATE] 文本标记相关方法（extract/has_marker/strip）
# 已全部移除——如果代码还在调用，会得到 AttributeError，提示需要迁移到 trigger_generation 工具。
TriggerExtractor = IntentDetector