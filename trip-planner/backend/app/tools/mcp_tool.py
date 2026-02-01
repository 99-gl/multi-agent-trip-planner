import asyncio
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPTool:
    """MCP工具管理器"""
    
    def __init__(self):
        self.mcp_tools: Optional[list[BaseTool]] = None
        self.mcp_client: Optional[MultiServerMCPClient] = None
        
    async def init_mcp_tools(self, server_config: Dict[str, Any]) -> list[BaseTool]:
        """
        初始化MCP服务器并获取工具列表
        
        Args:
            server_config: MCP服务器配置
            
        Returns:
            BaseTool列表
        """
        if self.mcp_tools is not None:
            return self.mcp_tools
        
        self.mcp_client = MultiServerMCPClient(server_config)
        self.mcp_tools = await self.mcp_client.get_tools()
        
        print(f"✅ MCP工具初始化成功")
        print(f"   工具数量: {len(self.mcp_tools)}")
        
        if len(self.mcp_tools):
            print("   可用工具:")
            for tool in self.mcp_tools[:5]:
                print(f"     - {tool.name}")
            if len(self.mcp_tools) > 5:
                print(f"     ... 还有 {len(self.mcp_tools) - 5} 个工具")
        
        return self.mcp_tools
    
    async def run(self, input_dict: Dict[str, Any]) -> Any:
        """
        运行MCP工具
        
        Args:
            input_dict: 包含action、tool_name和arguments的字典
            
        Returns:
            工具执行结果
        """
        if self.mcp_tools is None:
            raise ValueError("MCP工具未初始化，请先调用init_mcp_tools()")
            
        action = input_dict.get("action")
        tool_name = input_dict.get("tool_name")
        arguments = input_dict.get("arguments", {})
        
        if action != "call_tool":
            raise ValueError(f"不支持的action: {action}")
        
        # 查找对应的工具
        tool = next(
            (t for t in self.mcp_tools if t.name == tool_name),
            None
        )

        if tool is None:
            raise ValueError(f"未找到工具: {tool_name}")
            
        # 调用工具
        result = await tool.ainvoke(arguments)
        return result
