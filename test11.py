from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 初始化 FastAPI
app = FastAPI()

# 定义请求数据模型
class ChatRequest(BaseModel):
    prompt: str
    model: str = "111"

# API
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return {"response": request.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
# 前端 Web 界面
@app.get("/", response_class=HTMLResponse)
async def get_html():
    return """
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI 问答系统</title>
        <script>
            async function sendMessage() {
                const prompt = document.getElementById("prompt").value;
                const responseBox = document.getElementById("response");
                responseBox.innerHTML = "思考中...";
                
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({prompt: prompt, model: "gpt-4-turbo"})
                });
                const data = await response.json();
                responseBox.innerHTML = "AI：" + (data.response || "请求失败");
            }
        </script>
    </head>
    <body style="text-align: center; font-family: Arial;">
        <h1>AI 问答系统</h1>
        <input type="text" id="prompt" placeholder="请输入你的问题" style="width: 300px; padding: 10px;">
        <button onclick="sendMessage()" style="padding: 10px 20px;">发送</button>
        <p id="response" style="margin-top: 20px;"></p>
    </body>
    </html>
    """

# 运行服务器（启动命令在下面）