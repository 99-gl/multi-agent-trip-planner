import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState
from ..models.schemas import TripPlan


def planner_node(state: AgentState) -> AgentState:
    """
    汇总各个 specialist 的结果，生成最终 TripPlan
    """

    request = state["request"]

    attractions = "\n".join(state.get("attraction_results", []))
    weather = "\n".join(state.get("weather_results", []))
    hotels = "\n".join(state.get("hotel_results", []))

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个专业的旅行规划助手，负责根据给定信息生成结构化、可执行的旅行计划。"
            "请严格按照 JSON 格式输出，不要包含任何额外说明文本。"
        ),
        (
            "human",
            _build_planner_query(
                request=request,
                attractions=attractions,
                weather=weather,
                hotels=hotels,
            )
        )
    ])

    response = llm.invoke(prompt.format())

    plan = _parse_trip_plan(response.content, request)

    return {
        "final_plan": plan
    }


def _build_planner_query(
        request,
        attractions: str,
        weather: str,
        hotels: str,
    ) -> str:
    query = f"""
        请根据以下信息生成 {request.city} 的 {request.travel_days} 天旅行计划。

        【基本信息】
        - 城市: {request.city}
        - 日期: {request.start_date} 至 {request.end_date}
        - 天数: {request.travel_days} 天
        - 交通方式: {request.transportation}
        - 住宿偏好: {request.accommodation}
        - 用户偏好: {", ".join(request.preferences) if request.preferences else "无"}

        【景点信息】
        {attractions}

        【天气信息】
        {weather}

        【酒店信息】
        {hotels}

        【要求】
        1. 每天安排 2–3 个景点
        2. 每天包含早 / 中 / 晚餐建议
        3. 每天推荐一家具体酒店（必须来自酒店信息）
        4. 合理考虑交通距离与时间
        5. 返回 **完整 JSON**
        6. 景点需包含真实合理的经纬度
        """

    if request.free_text_input:
        query += f"\n【额外要求】\n{request.free_text_input}"

    return query

def _parse_trip_plan(response: str, request) -> TripPlan:
    """
    从 LLM 输出中提取 JSON 并构造 TripPlan
    """
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]

        data = json.loads(json_str)
        return TripPlan(**data)

    except Exception as e:
        raise RuntimeError(f"TripPlan JSON 解析失败: {e}")

