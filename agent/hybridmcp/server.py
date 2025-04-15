from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ToolCall(BaseModel):
    name: str
    parameters: dict

@app.post("/tool_call")
async def tool_call(tool: ToolCall):
    if tool.name == "get_weather":
        city = tool.parameters.get("city", "北京")
        result = f"{city}今天多云，气温22度"
        return {"result": result}
    elif tool.name == "add":
        a = tool.parameters.get("a", 0)
        b = tool.parameters.get("b", 0)
        return {"result": f"{a} + {b} = {a + b}"}
    else:
        return {"result": f"[Unknown tool: {tool.name}]"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)