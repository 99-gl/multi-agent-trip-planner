from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState
from ..tools.amap_tool import get_amap_mcp_tool
from ..tools.mcp_adapter import mcp_to_langchain_tools


# ===== 公共：构建 LLM + MCP Agent =====

def build_mcp_agent(system_prompt: str) -> AgentExecutor:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    mcp = get_amap_mcp_tool()
    tools = mcp_to_langchain_tools(mcp)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )


# ===== 1. 景点 Agent =====

def attraction_node(state: AgentState) -> AgentState:
    executor = build_mcp_agent(
        system_prompt=(
            "你是一个旅游助手，负责使用地图工具搜索城市中的热门景点。"
            "请使用合适的地图搜索工具获取信息。"
        )
    )

    city = state["request"].city
    query = f"搜索 {city} 的热门旅游景点"

    result = executor.invoke({
        "input": query
    })

    return {
        "attraction_results": [result["output"]]
    }


# ===== 2. 天气 Agent =====

def weather_node(state: AgentState) -> AgentState:
    executor = build_mcp_agent(
        system_prompt=(
            "你是一个天气查询助手，负责查询指定城市的天气情况。"
            "请调用地图或天气相关工具获取天气信息。"
        )
    )

    city = state["request"].city
    query = f"查询 {city} 的天气情况"

    result = executor.invoke({
        "input": query
    })

    return {
        "weather_results": [result["output"]]
    }


# ===== 3. 酒店 Agent =====

def hotel_node(state: AgentState) -> AgentState:
    executor = build_mcp_agent(
        system_prompt=(
            "你是一个酒店推荐助手，负责搜索城市中适合游客入住的酒店。"
            "请使用地图搜索工具完成任务。"
        )
    )

    city = state["request"].city
    query = f"搜索 {city} 评价较好的酒店"

    result = executor.invoke({
        "input": query
    })

    return {
        "hotel_results": [result["output"]]
    }
