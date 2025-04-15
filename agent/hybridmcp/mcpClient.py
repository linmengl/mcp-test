import json

import requests
from mcp.client import

# 初始化 MCP 客户端
client = stdio_client(server_url="http://localhost:8000")


# 获取工具列表
def get_tool_registry():
    tools = client.list_tools()
    return tools


# 生成系统提示词
def generate_system_prompt(tools):
    prompt = (
        "你是一个智能助手，如果用户的问题需要调用工具，你必须以如下格式输出：\n"
        "TOOL_CALL: {\n"
        "  \"name\": \"tool_name\",\n"
        "  \"parameters\": { \"param1\": \"value1\" }\n"
        "}\n"
        "支持以下工具：\n"
    )
    for tool in tools:
        params = ", ".join([f"{k}: {v}" for k, v in tool.parameters.items()])
        prompt += f"- {tool.name}({params})：{tool.description}\n"
    return prompt


# 初始化对话历史
def initialize_chat():
    tools = get_tool_registry()
    system_prompt = generate_system_prompt(tools)
    history = [{"role": "system", "content": system_prompt}]
    return history


# 与本地 Ollama 模型交互
def call_llm(history, user_input):
    history.append({"role": "user", "content": user_input})
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "deepseek-coder:6.7b",
        "messages": history,
        "stream": False
    })
    content = response.json()["message"]["content"]
    history.append({"role": "assistant", "content": content})
    return content


# 处理工具调用
def handle_tool_call(output, history):
    if output.strip().startswith("TOOL_CALL:"):
        try:
            tool_call = json.loads(output.replace("TOOL_CALL:", "").strip())
            result = client.call_tool(tool_call["name"], tool_call["parameters"])
            history.append({"role": "user", "content": f"TOOL_RESULT: {result}"})
            response = requests.post("http://localhost:11434/api/chat", json={
                "model": "deepseek-coder:6.7b",
                "messages": history,
                "stream": False
            })
            final_answer = response.json()["message"]["content"]
            history.append({"role": "assistant", "content": final_answer})
            return final_answer
        except Exception as e:
            return f"[ERROR: tool call failed: {e}]"
    return output


# 主聊天循环
def chat():
    history = initialize_chat()
    while True:
        user_input = input("You: ")
        output = call_llm(history, user_input)
        final_output = handle_tool_call(output, history)
        print("Bot:", final_output)


# def listen_for_tool_changes():
#     while True:
#         notification = client.receive_notification()
#         if notification.method == "notifications/tools/list_changed":
#             tools = get_tool_registry()
#             system_prompt = generate_system_prompt(tools)
#             # 更新系统提示词
#             history[0]["content"] = system_prompt

if __name__ == "__main__":
    chat()
