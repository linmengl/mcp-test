import requests
from fastapi import HTTPException

# Ollama 本地 API 地址
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:7b"

def request(prompt):
    try:
        # 构造 Ollama 请求
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,  # 关闭流式返回
            "options": {
                "num_predict": 2000,  # 控制输出 token 数
                "temperature": 0.7
            }
        }
        
        # 发送请求到 Ollama
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        # 解析结果
        result = response.json()
        return {"response": result["response"]}
    
    except requests.RequestException as e:
        raise HTTPException(500, f"Ollama API 错误: {str(e)}")