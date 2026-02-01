from langgraph.graph import StateGraph, START, END
from .state import AgentState
from ..agents.specialists import attraction_node, weather_node, hotel_node
from ..agents.planner_agent import planner_node

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. 添加节点
    workflow.add_node("search_attractions", attraction_node)
    workflow.add_node("search_weather", weather_node)
    workflow.add_node("search_hotels", hotel_node)
    workflow.add_node("generate_plan", planner_node)

    # 2. 定义边
    workflow.add_edge(START, "search_attractions")
    workflow.add_edge("search_attractions", "search_weather")
    workflow.add_edge("search_weather", "search_hotels")
    workflow.add_edge("search_hotels", "generate_plan")
    workflow.add_edge("generate_plan", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    print("Graph built successfully.")