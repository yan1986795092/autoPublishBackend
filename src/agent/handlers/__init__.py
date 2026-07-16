"""Agent 处理函数 — 按类型拆分为独立模块"""

from .writing import handle_writing
from .review import handle_review
# from .diagram import handle_diagram  # TEMP: DiagramAgent disabled
from .code import handle_code
from .web_search import handle_web_search
from .fix import handle_fix
from .search_analyzer import handle_search_analyzer
from .title import handle_title
from .tags import handle_tags
from .fallback import handle_fallback

HANDLER_MAP = {
    "WritingAgent": handle_writing,
    "ReviewAgent": handle_review,
    "FixAgent": handle_fix,
    "TitleAgent": handle_title,
    "TagAgent": handle_tags,
    # "DiagramAgent": handle_diagram,  # TEMP: DiagramAgent disabled
    "CodeAgent": handle_code,
    "WebSearch": handle_web_search,
    # "URLCrawler": handle_url_crawl,  # TEMP: URLCrawler disabled
    "SearchAnalyzer": handle_search_analyzer,
}
