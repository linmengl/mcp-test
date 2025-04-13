import requests
import json

# 定义可用函数
functions = [
    {
        "name": "get_weather",
        "description": "获取某地的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市名称，比如 Beijing、Shanghai"
                }
            },
            "required": ["location"]
        }
    }
]

# 用户问题
user_question = "北京今天的天气怎么样？"

# 构造 prompt 模拟 function calling
system_prompt = f"""你是一个 AI 助手，可以调用函数来完成任务。
已注册的函数如下：
{json.dumps(functions, indent=2, ensure_ascii=False)}

当用户提出问题时，你需要选择一个函数，并给出函数调用的 JSON 格式。
只输出函数调用的 JSON 结构，不要输出其他文字。
"""

prompt = f"{system_prompt}\n用户：{user_question}"

response = requests.post("http://localhost:11434/api/generate", json={
    "model": "deepseek-r1:7b",
    "prompt": prompt,
    "stream": False
})

print("模型回复：")
print(response.json()["response"])