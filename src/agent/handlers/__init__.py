"""Agent 处理函数 — 按类型拆分为独立模块"""

from .writing import handle_writing
from .seo import handle_seo
from .review import handle_review
from .diagram import handle_diagram
from .image import handle_image
from .hot_topic import handle_hot_topic
from .code import handle_code
from .web_search import handle_web_search
from .platform_adapter import handle_platform_adapter
from .fallback import handle_fallback
from .translation import handle_translation

HANDLER_MAP = {
    "WritingAgent": handle_writing,
    "SEOAgent": handle_seo,
    "ReviewAgent": handle_review,
    "DiagramAgent": handle_diagram,
    "ImageAgent": handle_image,
    "HotTopicAgent": handle_hot_topic,
    "CodeAgent": handle_code,
    "WebSearch": handle_web_search,
    "PlatformAdapter": handle_platform_adapter,
    "TranslationAgent": handle_translation,
}
