"""LLM 网关 — LangChain 适配器

保持 generate() / generate_sync() 接口签名不变，
内部使用 langchain_openai.ChatOpenAI 替代原始 httpx 调用。
"""
from typing import AsyncGenerator, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class LLMGateway:
    PROVIDERS = {
        "openai": "https://api.openai.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "zhipu": "https://open.bigmodel.cn/api/coding/paas/v4",
    }

    def __init__(self, api_key: str, provider: str = "openai", model: str = "", base_url: str = ""):
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.base_url = base_url or self.PROVIDERS.get(provider, self.PROVIDERS["openai"])
        self._llm: Optional[ChatOpenAI] = None

    def _get_llm(self, temperature: float = 0.7, max_tokens: int = 2000) -> ChatOpenAI:
        """延迟创建 ChatOpenAI 实例，支持每次调用自定义参数"""
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.model or "gpt-3.5-turbo",
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

    @staticmethod
    def _to_langchain_messages(messages: list[dict]) -> list:
        """将 [{role, content}, ...] 转为 LangChain Message 对象"""
        lc_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = True,
    ) -> AsyncGenerator[str, None]:
        """流式生成 — 保持原名和签名

        Yields:
            文本块，与原始 httpx SSE 实现行为一致
        """
        llm = self._get_llm(temperature=temperature, max_tokens=max_tokens)
        lc_messages = self._to_langchain_messages(messages)

        if stream:
            async for chunk in llm.astream(lc_messages):
                if chunk.content:
                    yield chunk.content
        else:
            response = await llm.ainvoke(lc_messages)
            yield response.content

    async def generate_sync(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """同步生成 — 收集完整响应后返回"""
        llm = self._get_llm(temperature=temperature, max_tokens=max_tokens)
        lc_messages = self._to_langchain_messages(messages)
        response = await llm.ainvoke(lc_messages)
        return response.content
