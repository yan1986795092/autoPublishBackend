"""Agent 处理函数 — 按类型拆分为独立模块"""

from .writing import handle_writing
from .seo import handle_seo
from .review import handle_review
from .diagram import handle_diagram
from .code import handle_code
from .web_search import handle_web_search, handle_url_crawl
from .fallback import handle_fallback

HANDLER_MAP = {
    "WritingAgent": handle_writing,
    "SEOAgent": handle_seo,
    "ReviewAgent": handle_review,
    "DiagramAgent": handle_diagram,
    "CodeAgent": handle_code,
    "WebSearch": handle_web_search,
    "URLCrawler": handle_url_crawl,
}
