import asyncio
import sys
from contextlib import AsyncExitStack
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # self.anthropic = Anthropic()
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model_name = "deepseek-r1:7b"

    # methods will go here

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def ollama_chat(self, messages):
        payload = {
            "model": self.model_name,
            "prompt": messages,
            "stream": False  # 关闭流式返回
        }

        headers = {
            "Content-Type": "application/json"
        }

        print("Sending payload:", payload)

        async with httpx.AsyncClient() as client:
            response = await client.post(self.ollama_url, headers=headers, json=payload)

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                print("Ollama API 请求失败！")
                print("响应状态码:", e.response.status_code)
                print("响应内容:", e.response.text)
                raise
            print(response)

            return response.json()

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # messages = [
        #     {
        #         "role": "user",
        #         "content": query
        #     }
        # ]

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        formatted_tools = self.format_tools_prompt(available_tools)

        # Initial Claude API call
        # response = self.anthropic.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     max_tokens=1000,
        #     messages=messages,
        #     tools=available_tools
        # )

        messages = [
            {
                "role": "system",
                "content": formatted_tools
            },
            {
                "role": "user",
                "content": query
            }
        ]

        print("Sending messages:", messages)

        response_json = await self.ollama_chat(messages)
        print("Received response:", response_json)
        content_blocks = response_json.get("message", {}).get("content", "")

        # Process response and handle tool calls
        final_text = [content_blocks]
        # 假设 Ollama 返回的 content 是字符串

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    def format_tools_prompt(self, tools):
        prompt = "你是一个智能助手，可以根据用户的请求，必要时调用工具来完成任务：\n"
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            input_schema = tool.get("input_schema", {})
            prompt += f"\n工具名: {name}\n说明: {desc}\n参数结构: {input_schema}\n"
        return prompt

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

# class Tool:
#     """Represents a tool with its properties and formatting."""
#     def __init__(self, name:str, description:str, input_schema: dict[str, Any]
#                  ) -> None:
#         self.name:str= name
#         self.description:str= description
#         self.input_schema:dict[str,Any] = input_schema
#         # 把工具的名字 / 工具的用途（description）和工具所需要的参数（args_desc）转化为文本
#     def format_for_llm(self) -> str:
#         """Format tool information for LLM.
#         Returns:
#             A formatted string describing the tool.
#         """
#         args_desc = []
#         if "properties" in self.input_schema:
#             for param_name, param_info in self.input_schema["properties"].items():
#                 arg_desc = (
#                     f"-{param_name}:{param_info.get('description','No description')}")
#                 if param_name in self.input_schema.get("required", []):
#                     arg_desc += " (required)"
#                 args_desc.append(arg_desc)
#                 return f"""
#         Tool: {self.name}
#         Description: {self.description}
#         Arguments: {chr(10).join(args_desc)}"""
