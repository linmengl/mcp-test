import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 初始化 FastAPI
app = FastAPI()

# 加载 DeepSeek-R1 模型
MODEL_NAME = "deepseek-ai/deepseek-r1-7b"  # 或 "deepseek-ai/deepseek-moe-16b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# 定义请求数据模型
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 100  # 限制最大生成长度

# AI 聊天 API
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")