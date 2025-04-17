import asyncio
import httpx

async def main():
    ollama_url = "http://localhost:11434/api/chat"
    model_name = "deepseek-r1:7b"

    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以根据用户的请求，必要时调用工具来完成任务。\n"
                       "工具名: get_alerts\n说明: Get weather alerts for a US state.\n"
                       "参数结构: {'properties': {'state': {'title': 'State', 'type': 'string'}}, 'required': ['state'], 'title': 'get_alertsArguments', 'type': 'object'}\n\n"
                       "工具名: get_forecast\n说明: Get weather forecast for a location.\n"
                       "参数结构: {'properties': {'latitude': {'title': 'Latitude', 'type': 'number'}, 'longitude': {'title': 'Longitude', 'type': 'number'}}, 'required': ['latitude', 'longitude'], 'title': 'get_forecastArguments', 'type': 'object'}"
        },
        {
            "role": "user",
            "content": "你好"
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()
        result = response.json()

    reply = result.get("message", {}).get("content", "[未获取到回复]")
    print("🤖 模型回复：\n", reply)

if __name__ == "__main__":
    asyncio.run(main())