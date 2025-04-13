import json
from prompt import REACT_PROMPT
from tools import get_closing_price,tools
import re
import logging
import requests
from fastapi import HTTPException

# https://time.geekbang.org/column/article/857271?utm_campaign=geektime_search&utm_content=geektime_search&utm_medium=geektime_search&utm_source=geektime_search&utm_term=geektime_search
# https://github.com/xingyunyang01/Geek02/blob/main/FunctionCalling/functioncalling.py

# 设置日志的基本配置
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG，显示所有级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志输出格式
)

# Ollama 本地 API 地址
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:7b"

def request(messages):
    try:
        # 将 OpenAI 格式的 messages 转换为 Ollama 需要的 prompt 
        prompt = "\n".join([f"{msg['role']} : {msg['content']}" for msg in messages]) 
        # 构建 Ollama 请求参数 
        data = { 
                "model" :  MODEL_NAME,  # 确保本地已下载模型（ollama pull deepseek） 
                "prompt" : prompt , 
                "stream" : False , # 是否流式输出 
                "options" : {"temperature" : 0.7}}
        
        # 发送请求到 Ollama
        response = requests.post(
            OLLAMA_API_URL, 
            json=data,
            headers = {"Content-Type" : "application/json"},
            timeout = 30
            )
        response.raise_for_status()
        
        # 解析结果
        result = response.json()
        return {"response": result["response"]}
    
    except requests.RequestException as e:
        raise HTTPException(500, f"Ollama API 错误: {str(e)}")


if __name__ == "__main__":
    instructions = "你是一个股票助手，可以回答股票相关的问题"

    query = "青岛啤酒和贵州茅台的收盘价哪个贵？"
    
    prompt = REACT_PROMPT.format(instructions=instructions, tools=tools, tool_names="get_closing_price", input=query)

    logging.info("prompt={}".format(prompt))

    messages = [{"role": "user", "content": prompt}]

    while True:
        response = request(messages)
        logging.info("----"+str(response))
        response_text = response.get("response", "")

        print("大模型的回复：{}".format(response_text))

        final_answer_match = re.search(r'Final Answer:\s*(.*)', response_text)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            print("最终答案:", final_answer)
            break

        messages.append(response_text)

        action_match = re.search(r'Action:\s*(\w+)', response_text)
        action_input_match = re.search(r'Action Input:\s*({.*?}|".*?")', response_text, re.DOTALL)

        if action_match and action_input_match:
            tool_name = action_match.group(1)
            params = json.loads(action_input_match.group(1))

            if tool_name == "get_closing_price":
                observation = get_closing_price(params['name'])
                print("人类的回复：Observation:", observation)
                messages.append({"role": "user", "content": f"Observation: {observation}"})