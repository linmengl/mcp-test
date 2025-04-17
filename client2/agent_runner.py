import json
from typing import Optional
from tool_call_manager import ToolCallManager
from prompt_manager import PromptManager
import util


class AgentRunner:
    def __init__(self, session, model_name="deepseek-r1:7b"):
        """
        session: MCP 的 ClientSession 实例
        model_name: 模型名称（用于 Ollama 调用）
        """
        self.session = session
        self.model_name = model_name
        self.prompt_mgr = PromptManager()

    async def run(self, user_query: str) -> str:
        """
        用户提问一次，Agent 自动处理所有多轮推理和工具调用，返回最终答案。
        """
        self.prompt_mgr.reset()
        self.prompt_mgr.add_system(await self._build_system_prompt())
        self.prompt_mgr.add_user(user_query)

        while True:
            # 调用大模型生成回复
            messages = self.prompt_mgr.get_messages()
            response_json = await util.ollama_chat(messages)
            response_content = response_json["message"]["content"]
            print(f"\n🤖 Assistant Output:\n{response_content}\n")

            # 判断是否为最终回复
            if "Final Answer:" in response_content:
                self.prompt_mgr.add_assistant_message(response_content)
                return response_content.split("Final Answer:")[1].strip()

            # 尝试解析工具调用
            tool_calls = ToolCallManager.parse_tool_calls(response_content)
            self.prompt_mgr.add_assistant_message(response_content)

            if not tool_calls:
                # 工具调用解析失败，直接返回
                return response_content

            # 执行工具调用并加入上下文
            for tool_call in tool_calls:
                try:
                    tool_name = tool_call["tool"]
                    args = tool_call["arguments"]
                    result = await self.session.call_tool(tool_name, args)
                    self.prompt_mgr.add_tool_result(result, tool_name)
                except Exception as e:
                    error = {"error": str(e)}
                    self.prompt_mgr.add_tool_result(error, tool_call["tool"])

    async def _build_system_prompt(self) -> str:
        """
        构建 system prompt，列出所有工具。
        """
        tool_response = await self.session.list_tools()
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        } for t in tool_response.tools]
        return util.build_system_prompt(tools)