"""旅行规划API路由"""

from fastapi import APIRouter, HTTPException
from ...models.schemas import (
    TripRequest,
    TripPlanResponse,
    ErrorResponse
)
from ...workflow import get_trip_planner_workflow

router = APIRouter(prefix="/trip", tags=["旅行规划"])


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    try:
        print(f"\n{'='*60}")
        print(f"📥 收到旅行规划请求:")
        print(f"   城市: {request.city}")
        print(f"   日期: {request.start_date} - {request.end_date}")
        print(f"   天数: {request.travel_days}")
        print(f"{'='*60}\n")

        # 1. 获取 LangGraph 工作流
        print("🔄 获取旅行规划工作流...")
        workflow = get_trip_planner_workflow()

        # 2. 构造初始 State
        initial_state = {
            "messages": [],
            "request": request,

            "attraction_results": [],
            "weather_results": [],
            "hotel_results": [],

            "final_plan": None,
        }

        # 3. 执行工作流
        print("🚀 开始执行旅行规划工作流...")
        final_state = workflow.invoke(initial_state)

        print("✅ 工作流执行完成")

        # 4. 从最终 State 中取结果
        trip_plan = final_state.get("final_plan")

        return TripPlanResponse(
            success=True,
            message="旅行计划生成成功",
            data=trip_plan
        )

    except Exception as e:
        print(f"❌ 生成旅行计划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
        )



@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        # 简单检查工作流是否可用
        workflow = get_trip_planner_workflow()
        if not workflow:
            raise ValueError("旅行规划工作流不可用")
        
        return {
            "status": "healthy",
            "service": "trip-planner"
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )

