"""内建模板 — 启动时自动注册到 PromptEngine

每个子模块定义 register() 函数，返回 PromptTemplateDef 列表。
"""

from ..engine import prompt_engine


def register_all():
    """注册所有内建模板"""
    from .system import templates as sys_templates
    from .writing import templates as writing_templates
    from .manager import templates as manager_templates
    from .review import templates as review_templates
    from .code import templates as code_templates
    from .diagram import templates as diagram_templates

    all_templates = (
        sys_templates +
        writing_templates +
        manager_templates +
        review_templates +
        code_templates +
        diagram_templates
    )

    prompt_engine.register_many(all_templates)
    return len(all_templates)
