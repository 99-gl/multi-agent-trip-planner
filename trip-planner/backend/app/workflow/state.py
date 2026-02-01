from typing import TypedDict, Annotated, List, Optional
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from ..models.schemas import TripRequest, TripPlan

class AgentState(TypedDict):
    # 1. 对话 / 思考上下文（如果你有 LLM reasoning）
    messages: Annotated[List[BaseMessage], add_messages]

    # 2. 原始用户请求（API 注入，Graph 不修改）
    request: TripRequest

    # 3. 各 Node 的中间产出（Graph 累积）
    attraction_results: Annotated[List[str], add]
    weather_results: Annotated[List[str], add]
    hotel_results: Annotated[List[str], add]

    # 4. Planner Node 的最终结果
    final_plan: Optional[TripPlan]
