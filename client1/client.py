import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import util1 as util
from prompt_manager1 import PromptManager
from tool_call_manager1 import ToolCallManager


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        # 创建一个异步退出栈
        self.exit_stack = AsyncExitStack()

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
        # 用 AsyncExitStack 来管理 stdio_client(...) 这个异步上下文，确保用完后会自动关闭或释放资源
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

        prompt_mgr = PromptManager(system_message)
        prompt_mgr.add_user_message(query)

        while True:
            # 1. 调用大模型
            # prompt_mgr.dialog = compress_dialog(prompt_mgr.dialog, max_tokens=4096)
            response_json = await util.ollama_chat(prompt_mgr.get_messages())
            response_content = response_json["message"]["content"]

            # 2. 检查终止条件
            if "Final Answer:" in response_content:
                return response_content.split("Final Answer:")[1].strip()

            # 假设你已经拿到 response_content
            tool_calls = ToolCallManager.parse_tool_calls(response_content)

            print("tool_calls:", tool_calls)

            if not tool_calls:
                # 没有工具调用，直接返回
                return response_content

            # 有工具调用，先记录 assistant 输出
            prompt_mgr.add_assistant_message(response_content)

            for tool_call in tool_calls:
                try:
                    tool_name = tool_call["tool"]
                    args = tool_call["arguments"]
                    tool_result = await self.session.call_tool(tool_name, args)
                    print("got tool result:", tool_result)
                    if tool_result.isError:
                        print("工具调用失败", tool_result)
                        return response_content
                    actual_dict = json.loads(tool_result.content[0].text)
                    prompt_mgr.add_tool_result(actual_dict)
                except Exception as e:
                    prompt_mgr.add_tool_result({"error": str(e)})

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
