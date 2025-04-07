import json
import re
import logging
from typing import Dict, List, Optional
import requests
from fastapi import HTTPException
from prompt import REACT_PROMPT
from tools import get_closing_price, tools

# 常量定义（避免硬编码）
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:7b"
FINAL_ANSWER_PATTERN = r'Final Answer:\s*(.*)'
ACTION_PATTERN = r'Action:\s*(\w+)'
ACTION_INPUT_PATTERN = r'Action Input:\s*({.*?}|".*?")'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OllamaClient:
    """封装 Ollama API 请求的客户端"""
    def __init__(self, api_url: str, model: str, timeout: int = 30):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt: str, temperature: float = 0.7) -> Dict:
        """发送生成请求到 Ollama"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        try:
            logger.debug(f"Sending request to Ollama: {data}")
            response = requests.post(
                self.api_url,
                json=data,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            raise HTTPException(500, f"Ollama API 错误: {str(e)}")

def build_prompt(query: str, instructions: str) -> str:
    """构建符合 ReAct 模式的提示词"""
    return REACT_PROMPT.format(
        instructions=instructions,
        tools=json.dumps(tools, ensure_ascii=False),
        tool_names="get_closing_price",
        input=query
    )

def parse_response(response_text: str) -> Optional[Dict]:
    """解析模型响应，提取 Final Answer 或 Action"""
    final_answer_match = re.search(FINAL_ANSWER_PATTERN, response_text, re.DOTALL)
    if final_answer_match:
        return {"type": "final_answer", "content": final_answer_match.group(1).strip()}

    action_match = re.search(ACTION_PATTERN, response_text)
    action_input_match = re.search(ACTION_INPUT_PATTERN, response_text, re.DOTALL)
    if action_match and action_input_match:
        try:
            action_input = json.loads(action_input_match.group(1).replace('\n', ''))
            return {
                "type": "action",
                "name": action_match.group(1),
                "input": action_input
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse action input: {action_input_match.group(1)}")
    return None

def execute_tool(action_name: str, action_input: Dict) -> str:
    """执行工具函数并返回观测结果"""
    if action_name == "get_closing_price":
        return get_closing_price(action_input.get('name'))
    # 扩展其他工具时在此添加
    raise ValueError(f"未知工具: {action_name}")

def main_loop(initial_messages: List[Dict]) -> str:
    """主循环处理多轮交互"""
    client = OllamaClient(OLLAMA_API_URL, MODEL_NAME)
    messages = initial_messages.copy()

    while True:
        try:
            response = client.generate(prompt=messages[-1]["content"])
            response_text = response.get("response", "")
            logger.info(f"模型响应: {response_text}")

            parsed = parse_response(response_text)
            if not parsed:
                logger.warning("无法解析的响应格式")
                continue

            if parsed["type"] == "final_answer":
                logger.info(f"最终答案: {parsed['content']}")
                return parsed["content"]

            if parsed["type"] == "action":
                observation = execute_tool(parsed["name"], parsed["input"])
                logger.info(f"执行工具 {parsed['name']}, 输入: {parsed['input']}, 结果: {observation}")
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

        except HTTPException as e:
            logger.error(f"API 请求失败: {e.detail}")
            break
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}")
            break

    return "服务暂时不可用"

if __name__ == "__main__":
    instructions = "你是一个股票助手，可以回答股票相关的问题"
    query = "青岛啤酒和贵州茅台的收盘价哪个贵？"

    # 构建初始提示词
    prompt = build_prompt(query, instructions)
    logger.info(f"初始提示词: {prompt}")

    # 启动主循环
    final_answer = main_loop([{"role": "user", "content": prompt}])
    print(f"最终答案: {final_answer}")