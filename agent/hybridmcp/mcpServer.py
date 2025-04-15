from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("Demo Server")

# 注册一个加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """返回两个数字的和"""
    return a + b

# 启动 MCP 服务器
if __name__ == "__main__":
    mcp.run()