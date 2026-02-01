import json
from langchain_openai import ChatOpenAI

from ..workflow.state import AgentState
from ..models.schemas import TripPlan

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""

def planner_node(state: AgentState) -> AgentState:
    """
    汇总各个 specialist 的结果，生成最终 TripPlan
    """

    request = state["request"]

    attractions = "\n".join(state.get("attraction_results", []))
    weather = "\n".join(state.get("weather_results", []))
    hotels = "\n".join(state.get("hotel_results", []))

    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        temperature=0.2,
    )

    human_content = _build_planner_query(
            request=request,
            attractions=attractions,
            weather=weather,
            hotels=hotels,
        )

    messages = [
        {"role": "system", "content": PLANNER_AGENT_PROMPT},
        {"role": "user", "content": human_content}
    ]

    response = llm.invoke(messages)

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

