"""Agent 处理函数 — 按类型拆分为独立模块"""

from .writing import handle_writing
from .review import handle_review
from .diagram import handle_diagram
from .code import handle_code
from .web_search import handle_web_search, handle_url_crawl
from .fix import handle_fix
from .hot_topic import handle_hot_topic
from .fallback import handle_fallback

HANDLER_MAP = {
    "WritingAgent": handle_writing,
    "ReviewAgent": handle_review,
    "FixAgent": handle_fix,
    "DiagramAgent": handle_diagram,
    "CodeAgent": handle_code,
    "WebSearch": handle_web_search,
    "URLCrawler": handle_url_crawl,
    "HotTopicAgent": handle_hot_topic,
}
