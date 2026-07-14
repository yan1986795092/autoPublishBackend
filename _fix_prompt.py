"""Replace conversation engine system prompt and tool schema"""
with open('src/conversation/engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ── 1. Replace SYSTEM_PROMPT ──
old_system = content[content.find('SYSTEM_PROMPT = """'):]
old_system = old_system[:old_system.find('"""') + 3]
old_system_text = old_system

new_system = '''SYSTEM_PROMPT = """你是专业的 AI 内容创作助手，通过自然对话引导用户完成内容创作。

## 职责

1. 了解用户想创作什么（主题、平台、风格、篇幅、语调、受众等）
2. 信息不足时自然追问，一次 1-2 个问题
3. 信息充足后调用 trigger_generation 启动创作

## 约束

- 不要自行撰写正文、标题、章节——那是创作 pipeline 的工作
- 不要使用 [GENERATE] 标记——此格式已废弃
- 如果用户只是修改措辞（改词、调一句），可以直接改好回复，无需触发创作

## 触发时机

当用户明确要求生成内容，且必要信息（至少主题）已收集完毕时：
1. 先输出一句简短的确认（如"好，开始创作"）
2. 在同一轮调用 trigger_generation 工具，传入已收集的参数

用户已选择的配置会以系统消息提供，直接使用即可，无需重复询问。"""'''

if old_system_text in content:
    content = content.replace(old_system_text, new_system)
    print('OK: SYSTEM_PROMPT replaced')
else:
    print('ERROR: SYSTEM_PROMPT not found')

# ── 2. Replace TRIGGER_TOOL_SCHEMA ──
old_tool_start = content.find('TRIGGER_TOOL_SCHEMA: list[dict] = [')
old_tool_section = content[old_tool_start:]
# Find the closing bracket of the list
depth = 0
end_idx = 0
for i, c in enumerate(old_tool_section):
    if c == '[':
        depth += 1
    elif c == ']':
        depth -= 1
        if depth == 0:
            end_idx = old_tool_start + i + 1
            break

old_tool = content[old_tool_start:end_idx]

new_tool = '''TRIGGER_TOOL_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "trigger_generation",
            "description": "信息收集完毕，开始执行创作 pipeline。调用前先输出文本确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "revise", "expand"],
                        "description": "create=创作新内容, revise=修改已有, expand=扩写",
                    },
                    "topic": {
                        "type": "string",
                        "description": "创作主题，填写实际的主题或标题",
                    },
                    "platform": {
                        "type": "string",
                        "description": "目标平台, 可选: xiaohongshu/wechat/douyin/weibo/bilibili/zhihu",
                    },
                    "style": {"type": "string", "description": "写作风格"},
                    "length": {"type": "string", "enum": ["short", "medium", "long"], "description": "篇幅"},
                    "tone": {"type": "string", "enum": ["friendly", "professional", "humorous", "warm"], "description": "语调"},
                    "audience": {"type": "string", "enum": ["beginner", "intermediate", "expert"], "description": "目标读者"},
                    "keywords": {"type": "array", "items": {"type": "string"}, "description": "核心关键词"},
                    "revision_focus": {"type": "string", "description": "修改重点（仅 revise/expand）"},
                },
                "required": ["action", "topic"],
            },
        },
    },
]'''

if old_tool in content:
    content = content.replace(old_tool, new_tool)
    print('OK: TRIGGER_TOOL_SCHEMA replaced')
else:
    print('ERROR: TRIGGER_TOOL_SCHEMA not found')

with open('src/conversation/engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

import ast
ast.parse(content)
print('Syntax OK')
