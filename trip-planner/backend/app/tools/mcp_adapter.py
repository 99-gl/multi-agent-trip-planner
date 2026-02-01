from langchain.tools import BaseTool
from typing import Any, List
from .mcp_tool import MCPTool


class MCPWrappedTool(BaseTool):
    """
    一个 LangChain Tool，对应 MCP Server 中的一个 tool
    """
    mcp: MCPTool
    tool_name: str

    def _run(self, **kwargs) -> str:
        return self.mcp.call(
            tool_name=self.tool_name,
            arguments=kwargs
        )

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError("Async not supported")


def mcp_to_langchain_tools(mcp: MCPTool) -> List[BaseTool]:
    """
    将 MCP Server 暴露的 tools 转换为 LangChain Tools
    """
    tools: List[BaseTool] = []

    for spec in mcp.list_tools():
        tools.append(
            MCPWrappedTool(
                name=spec["name"],
                description=spec.get("description", ""),
                mcp=mcp,
                tool_name=spec["name"],
            )
        )

    return tools
