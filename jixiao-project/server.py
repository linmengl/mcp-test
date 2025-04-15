from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("jixiao-project")

# 注解向 MCP Server 注册工具
@mcp.tool()
def get_score_by_name(name: str) -> str:
    """根据员工的姓名获取该员工的绩效得分"""
    if name == "张三":
        return "name:张三 绩效评分：85.9"
    elif name == "李四":
        return "name:李四 绩效评分：89.9"
    else:
        return "未搜到该员工"

@mcp.resource("file://info.md")
def get_file() -> str:
    """读取info.md的内容，从而获取员工的信息，例如性别等"""
    with open("/Users/menglin/ai-model/ai-project/jixiao-project/info.md", "r", encoding="utf-8") as f:
        return f.read()