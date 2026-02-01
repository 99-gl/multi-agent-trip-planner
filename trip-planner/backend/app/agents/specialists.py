from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

from ..tools.mcp_tool import MCPTool
from ..config import get_settings
from ..workflow.state import AgentState

# 全局MCP工具实例
_amap_mcp_tool = None

async def get_amap_tools() -> list[BaseTool]:
    """
    使用 LangChain MCPClient 获取高德地图 MCP 工具列表
    异步获取高德地图MCP工具实例(单例模式)
    
    Returns:
        BaseTool列表
    """
    global _amap_mcp_tool
    
    if _amap_mcp_tool is not None and _amap_mcp_tool.mcp_tools is not None:
        return _amap_mcp_tool.mcp_tools
    
    settings = get_settings()
    
    if not settings.amap_api_key:
        raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
    
    # 创建MCP工具实例
    _amap_mcp_tool = MCPTool()
    
    # 配置MCP服务器
    server_config = {
        "amap": {
            "command": "uvx",
            "args": ["amap-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
            "transport": "stdio",
        }
    }
    
    # 初始化工具
    tools = await _amap_mcp_tool.init_mcp_tools(server_config)
    return tools

async def build_tool_llm(system_prompt: str):
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0,
    )

    # 获取高德地图MCP工具列表
    tools = await get_amap_tools()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # ⭐ 关键：LLM 直接 bind tools
    return prompt | llm.bind_tools(tools)


async def attraction_node(state: AgentState) -> AgentState:
    chain = await build_tool_llm(
        system_prompt=(
            "你是一个旅游助手，负责使用地图工具搜索城市中的热门景点。"
            "请使用合适的地图搜索工具获取信息。"
        )
    )

    city = state["request"].city
    query = f"搜索 {city} 的热门旅游景点"

    result = await chain.ainvoke({
        "input": query
    })

    return {
        "attraction_results": [result.content]
    }


async def weather_node(state: AgentState) -> AgentState:
    chain = await build_tool_llm(
        system_prompt=(
            "你是一个天气查询助手，负责查询指定城市的天气情况。"
            "请调用地图或天气相关工具获取天气信息。"
        )
    )

    city = state["request"].city
    query = f"查询 {city} 的天气情况"

    result = await chain.ainvoke({
        "input": query
    })

    return {
        "weather_results": [result.content]
    }


async def hotel_node(state: AgentState) -> AgentState:
    chain = await build_tool_llm(
        system_prompt=(
            "你是一个酒店推荐助手，负责搜索城市中适合游客入住的酒店。"
            "请使用地图搜索工具完成任务。"
        )
    )

    city = state["request"].city
    query = f"搜索 {city} 评价较好的酒店"

    result = await chain.ainvoke({
        "input": query
    })

    return {
        "hotel_results": [result.content]
    }
