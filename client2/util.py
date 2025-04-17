import json
import json
from json.decoder import JSONDecodeError
# from json_repair import repair_json
from typing import Optional
import re
import httpx

ollama_url = "http://localhost:11434/api/chat"
model_name = "deepseek-r1:7b"

# 改进的JSON提取逻辑
def extract_json(content: str) -> Optional[dict]:
    try:
        # 尝试直接解析整个内容
        return json.loads(content)
    except JSONDecodeError:
        # 增强版正则匹配（支持嵌套和转义）
        pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
        if match := re.search(pattern, content, re.DOTALL):

            try:
                fixed = match.group().replace("'", '"')  # 单引号转双引号
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    return None

def format_tools_prompt(tools):
    prompt = ""
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        input_schema = tool.get("input_schema", {})
        prompt += f"\ntool-name: {name}\n说明: {desc}\n参数结构: {input_schema}\n"
    print("prompt:", prompt)
    return prompt

def should_terminate( response):
    # 优先级1：检测终止标记
    if "Final Answer:" in response:
        return True

    # 优先级2：验证JSON有效性
    try:
        import json
        json.loads(response)
        return True  # 有效JSON视为可终止
    except ValueError:
        # 尝试提取嵌入式JSON
        if extracted_json := extract_json(response):
            return True
    return False

async def ollama_chat(messages):
    print("\nPrompt 给模型：\n", messages)
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False  # 关闭流式返回
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(ollama_url, json=payload)
        response.raise_for_status()
    return response.json()

def build_system_prompt(tools):
    prompt = ""
    for tool in tools:
        prompt += (
            f"\n🔧 tool-name: {tool['name']}\n"
            f"📖 描述: {tool['description']}\n"
            f"📥 参数结构: {json.dumps(tool['input_schema'], ensure_ascii=False)}\n"
        )
    return (
        f"""You are a helpful assistant with access to the following tools:
{prompt}

---

📌 任务说明：

- 当你可以独立完成任务时，直接回复格式如下：
Final Answer: [你的回答]

- 当你需要使用工具时，请仅输出以下格式的 JSON（严格格式）：
```json
{{
  "tool": "tool-name",
  "arguments": {{
    "arg1": "xxx",
    "arg2": "yyy"
  }}
}}
"""
    )
