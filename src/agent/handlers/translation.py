"""TranslationAgent — 多语言翻译，保留格式与语气"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from ..agent_utils import extract_output

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "pt": "Português",
    "ru": "Русский",
    "ar": "العربية",
    "th": "ไทย",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
}

TRANSLATION_SYSTEM = """You are a professional translator. Translate the following content to {target_lang}.

Rules:
- Preserve all markdown formatting, code blocks, and links
- Maintain the original tone and style
- Keep technical terms accurate
- For UI/UX content, use natural, idiomatic expressions
- Output ONLY the translated content, no explanations"""


async def handle_translation(task_input: dict, llm: ChatOpenAI) -> dict:
    content = task_input.get("content", "")
    target_lang = task_input.get("target_lang", "en")
    source_lang = task_input.get("source_lang", "auto")

    if not content:
        dep_context = task_input.get("_context", {})
        for cdata in dep_context.values():
            if cdata.get("content"):
                content = cdata["content"]
                break

    if not content:
        return {
            "status": "failed",
            "agent_type": "TranslationAgent",
            "error": "No content to translate",
            "output": {},
            "tokens_used": 0,
            "thinking_steps": [],
        }

    lang_name = LANG_NAMES.get(target_lang, target_lang)

    response = await llm.ainvoke([
        SystemMessage(content=TRANSLATION_SYSTEM.format(target_lang=lang_name)),
        HumanMessage(content=content[:8000]),
    ])

    translated = extract_output(response.content)
    tokens = len(response.content) // 4

    return {
        "status": "completed",
        "agent_type": "TranslationAgent",
        "output": {
            "content": translated,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original_length": len(content),
            "translated_length": len(translated),
        },
        "tokens_used": tokens,
        "thinking_steps": [{
            "agent": "TranslationAgent",
            "action": "translate",
            "reasoning": f"Translating to {lang_name}",
            "output_summary": f"Translated {len(content)} → {len(translated)} chars ({lang_name})",
            "tokens_used": tokens,
            "status": "done",
        }],
    }
