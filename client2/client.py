import argparse
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_runner import AgentRunner


async def main():
    parser = argparse.ArgumentParser(description="MCP Agent Runner")
    parser.add_argument("server", help="Path to server script (.py or .js)")
    parser.add_argument("--debug", action="store_true", help="Enable step-by-step debug mode")
    parser.add_argument("--mode", choices=["auto", "tool-first", "llm-first"], default="auto",
                        help="Agent behavior mode")
    args = parser.parse_args()

    # 启动 MCP Server
    server_params = StdioServerParameters(
        command="python",
        args=[args.server],
        env=None
    )

    print("🚀 Connecting to MCP server...")
    async with stdio_client(server_params) as (stdio, write), ClientSession(stdio, write) as session:
        await session.initialize()
        print("✅ Connected to MCP server.")

        agent = AgentRunner(session, behavior_mode=args.mode)

        print("\n🤖 MCP Agent Ready!")
        print(f"Behavior mode: {args.mode}")
        print("Type your query. Use 'quit' to exit.\n")

        while True:
            query = input("🔍 Query: ").strip()
            if query == "": break
            if query.lower() == 'quit':
                break

            if args.debug:
                print("\n⚙️ Step-by-step debug enabled.")
                input("Press Enter to start model reasoning...")

            if args.mode == "tool-first":
                print("🔧 Forcing tool-first mode. Wrapping user query for tool preference.")
                query = f"请优先判断是否可调用工具来完成此请求：{query}"
            elif args.mode == "llm-first":
                print("🧠 Forcing LLM-first mode. Wrapping user query for LLM preference.")
                query = f"请尽可能直接回答此请求，除非确实需要使用工具：{query}"

            response = await agent.run(query)

            print("\n--------------------------------------\n")
            print("\n💬 Final Answer:")
            print(response)
            print("\n--------------------------------------\n")
            if args.debug:
                input("Press Enter to continue...")


if __name__ == "__main__":
    asyncio.run(main())

# uv run client.py ../weather/weather.py
