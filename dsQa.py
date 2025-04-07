import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Ollama 本地 API 地址
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:7b"

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.6

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 构造 Ollama 请求
        payload = {
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "stream": False,  # 关闭流式返回
            "options": {
                "num_predict": request.max_tokens,  # 控制输出 token 数
                "temperature": request.temperature
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