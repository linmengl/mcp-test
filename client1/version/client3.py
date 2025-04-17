import asyncio
import json
import re
import sys
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import util
from prompt_manager import PromptManager


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        response = await self.session.list_tools()
        print("\nConnected to server with tools:", [tool for tool in response.tools])

    # -------------------------------------------------------------------

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        tools_description = util.format_tools_prompt(available_tools)

        system_message = (
            f"""You are a helpful assistant with access to the following tools:
        {tools_description}

        ---

        📌 任务规则：

        🎯 请根据用户问题选择最合适的工具。  
        若无需调用工具，直接回答，格式如下：  
        `Final Answer: [你的回复]`

        🛑 当需要使用工具时，**必须仅输出以下格式的 JSON（严格按照格式，无其他内容）：**
        ```json
        {{
          "tool": "tool-name",
          "arguments": {{
            "argument-name": "value"
          }}
        }}"""
        )

        print("Sending system_message:", system_message)
        print("\n\n")

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]

        while True:
            # 1. 调用大模型
            response_json = await util.ollama_chat(messages)
            response_content = response_json["message"]["content"]

            # 2. 检查终止条件
            if "Final Answer:" in response_content:
                return response_content.split("Final Answer:")[1].strip()

            # 3. 尝试提取工具调用
            if tool_request := util.extract_json(response_content):
                try:
                    # 4. 调用MCP工具
                    tool_result = await self.session.call_tool(
                        tool_request["tool"],
                        tool_request["arguments"]
                    )

                    # 5. 保存上下文
                    messages.extend([
                        {"role": "assistant", "content": response_content},
                        {
                            "role": "tool",
                            "content": json.dumps(tool_result),
                            "tool_call_id": str(hash(tool_request["tool"]))
                        }
                    ])
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": str(e)}),
                        "tool_call_id": "..."
                    })
            else:
                return response_content  # 无法识别工具调用时直接返回

        # response_json = await util.ollama_chat(messages)
        # response_content = response_json["message"]["content"]
        # print("deepseek输出content", response_content)
        #
        # if util.should_terminate(response_content):
        #     return response_content
        #
        # json_pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
        # tool_request_json = re.search(json_pattern, response_content, re.DOTALL)
        # print("提取到的JSON:", tool_request_json.group())
        # json_block = json.loads(tool_request_json.group())
        # if tool_request_json:
        #     try:
        #         tool_name = json_block.get("tool")
        #         args = json_block.get("arguments", {})
        #
        #         # 调用工具并获取结果
        #         tool_result = await self.session.call_tool(tool_name, args)
        #
        #         # 将结果反馈给模型生成最终回复
        #         messages.append({
        #             "role": "tool",
        #             "content": json.dumps(tool_result),
        #             "tool_call_id": str(hash(tool_name))  # 唯一标识
        #         })
        #         return await util.ollama_chat(messages)
        #
        #     except Exception as e:
        #         return f"工具调用失败: {str(e)}"
        # else:
        #     return response_content  # 直接返回模型回复

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

# uv run client.py ../weather/weather.py
