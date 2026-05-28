"""Token 估算工具 — 使用 tiktoken 精确计算，未安装时自动降级到字符除法估算。

用法：
    from src.utils.tokenizer import estimate_tokens

    tokens = estimate_tokens("Hello, world!")
    tokens = estimate_tokens(["Hello", "world"], encoding_name="cl100k_base")
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)

_tiktoken_encoder = None


def _get_tiktoken_encoder(encoding_name: str = "cl100k_base"):
    """延迟加载 tiktoken encoder，全局缓存"""
    global _tiktoken_encoder
    if _tiktoken_encoder is not None:
        return _tiktoken_encoder
    try:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding(encoding_name)
        logger.debug("tiktoken encoder loaded: %s", encoding_name)
        return _tiktoken_encoder
    except ImportError:
        logger.warning("tiktoken not installed, falling back to char/4 estimation")
        return None
    except Exception as e:
        logger.warning("tiktoken init failed (%s), falling back to char/4 estimation", e)
        return None


def estimate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """估算文本的 token 数量。

    优先使用 tiktoken（OpenAI cl100k_base 编码，与 GPT-4/3.5-turbo 一致）。
    tiktoken 不可用时降级为 len(text) / 4（中文约 1 token = 1.5 字符，英文约 4 字符）。

    对于中文内容，调整系数到 2.5（更接近实际 token 消耗）。
    """
    if not text:
        return 0

    encoder = _get_tiktoken_encoder(encoding_name)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception as e:
            logger.warning("tiktoken encode failed (%s), using char fallback", e)

    # 降级估算：混合中英文内容
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    total_chars = len(text)
    # tiktoken 对中文按 UTF-8 编码，约 2-3 个字节/token
    # 中文约 1 token = 1.5-2 字符；英文约 1 token = 4 字符
    # 加权平均：chinese * 2 + (total - chinese) * 0.25
    if chinese_chars > 0:
        return int(chinese_chars / 1.5 + (total_chars - chinese_chars) / 4)
    return int(total_chars / 4)


def estimate_tokens_batch(texts: Sequence[str], encoding_name: str = "cl100k_base") -> list[int]:
    """批量估算 token 数量"""
    return [estimate_tokens(t, encoding_name) for t in texts]
