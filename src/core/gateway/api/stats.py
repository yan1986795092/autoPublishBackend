from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/gateway", tags=["stats"])


class StatsResponse(BaseModel):
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    avg_latency_ms: float


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    request: Request,
    from_time: Optional[str] = Query(None, description="ISO format datetime"),
    to_time: Optional[str] = Query(None, description="ISO format datetime"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    cost_tracker = request.state.cost_tracker
    from_dt = datetime.fromisoformat(from_time) if from_time else None
    to_dt = datetime.fromisoformat(to_time) if to_time else None

    stats = await cost_tracker.query_stats(from_dt, to_dt, provider)
    return StatsResponse(**stats)
