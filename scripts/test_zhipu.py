import httpx
import asyncio
import json

async def test_zhipu_stream():
    api_key = "your-zhipu-api-key"  # 请替换为实际的 API 密钥
    base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
    model = "glm-4-flash"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    messages = [{"role": "user", "content": "用一句话介绍你自己"}]

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        try:
            async with client.stream(
                "POST", "/chat/completions",
                json={"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 100, "stream": True}
            ) as response:
                print(f"Status: {response.status_code}")
                if response.status_code != 200:
                    body = await response.aread()
                    print(f"Error body: {body.decode()}")
                else:
                    line_count = 0
                    async for line in response.aiter_lines():
                        line_count += 1
                        if line:
                            print(f"LINE[{line_count}]: {line!r}")
                            if line == "data: [DONE]":
                                print("DONE!")
                                break
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        if "content" in delta:
                                            print(f"CONTENT: {delta['content']}")
                                except Exception as e:
                                    print(f"Parse error: {e}")
        except Exception as e:
            print(f"Request error: {e}")
            import traceback
            traceback.print_exc()

asyncio.run(test_zhipu_stream())
