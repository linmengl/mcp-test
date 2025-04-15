import json
import requests
import re

OLLAMA_API = "http://localhost:11434/api/chat"
MCP_TOOLS_API = "http://localhost:8000/tool_call"

client_id = "user-001"

history = [
    {
        "role": "system",
        "content": (
            "你是一个智能助手，如果用户的问题需要调用工具，你必须以如下格式输出：\n"
            "TOOL_CALL: {\n"
            "  \"name\": \"tool_name\",\n"
            "  \"parameters\": { \"param1\": \"value1\" }\n"
            "}\n"
            "不要自行回答工具能解决的问题，只输出 TOOL_CALL 格式。支持工具包括：\n"
            "- get_weather(city)\n"
            "- add(a, b)"
        )
    }
]

def call_llm(message: str):
    global history
    history.append({"role": "user", "content": message})
    res = requests.post(OLLAMA_API, json={
        "model": "deepseek-r1:7b",  # or deepseek-llm:7b
        "messages": history,
        "stream": False
    })
    print(res.json())
    content = res.json()["message"]["content"]
    history.append({"role": "assistant", "content": content})
    return content

def detect_and_call_tool(llm_output: str):
    if llm_output.strip().__contains__("TOOL_CALL:"):
        try:
            json_part = llm_output.replace("TOOL_CALL:", "").strip()

            match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if match:
                try:
                    json_part = str(match.group(0))
                except json.JSONDecodeError as e:
                    print("JSON 解析失败，内容如下：", match.group(0))
                    raise e
            else:
                raise ValueError("未能从模型输出中提取出 JSON。原始内容：" + llm_output)
            print("json部分", json_part)

            tool_call = json.loads(json_part)
            tool_response = requests.post(MCP_TOOLS_API, json=tool_call)
            tool_result = tool_response.json()["result"]
            # 将结果反馈给 LLM
            history.append({"role": "user", "content": f"TOOL_RESULT: {tool_result}"})
            res = requests.post(OLLAMA_API, json={
                "model": "deepseek-r1:7b",
                "messages": history,
                "stream": False
            })
            final_answer = res.json()["message"]["content"]
            history.append({"role": "assistant", "content": final_answer})
            return final_answer
        except Exception as e:
            return f"[ERROR: tool call failed: {e}]"
    return llm_output

def chat():
    while True:
        user_input = input("You: ")
        output = call_llm(user_input)
        final_output = detect_and_call_tool(output)
        print("Bot:", final_output)

if __name__ == "__main__":
    chat()


