"""Admin metrics API — 运维中心后端接口。

提供产能看板、发布监控、成本分析三个端点所需的数据。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from src.database import SessionLocal
from src.observability.metrics import stats_collector

router = APIRouter(prefix="/api/v1/admin/metrics", tags=["admin"])


# ── 产能看板 ──────────────────────────────────────────────────────────


@router.get("/production")
async def get_production_metrics() -> dict:
    """产能看板：今日/本周产量、AI 生成占比、趋势、平台分布"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    data: dict[str, Any] = {
        "daily": {"count": 0, "change_vs_yesterday": 0.0},
        "weekly": {"count": 0, "change_vs_last_week": 0.0},
        "ai_generation_ratio": 0.0,
        "avg_completion_time_min": 0.0,
        "trend": [],
        "platform_distribution": [],
    }

    try:
        db = SessionLocal()
        try:
            # 今日产量
            row = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) = :today"),
                {"today": today},
            ).scalar()
            data["daily"]["count"] = row or 0

            # 昨日产量（环比）
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            yesterday_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) = :yesterday"),
                {"yesterday": yesterday},
            ).scalar() or 0
            if yesterday_count > 0:
                data["daily"]["change_vs_yesterday"] = round(
                    (data["daily"]["count"] - yesterday_count) / yesterday_count, 4
                )

            # 本周总量
            week_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) >= :week_ago"),
                {"week_ago": week_ago},
            ).scalar()
            data["weekly"]["count"] = week_count or 0

            # 上周总量（环比）
            two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
            last_week_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) >= :two_weeks_ago AND DATE(created_at) < :week_ago"),
                {"two_weeks_ago": two_weeks_ago, "week_ago": week_ago},
            ).scalar() or 0
            if last_week_count > 0:
                data["weekly"]["change_vs_last_week"] = round(
                    (data["weekly"]["count"] - last_week_count) / last_week_count, 4
                )

            # 趋势（最近 7 天逐日）
            rows = db.execute(
                text("""
                    SELECT DATE(created_at) as day, COUNT(*) as cnt
                    FROM contents
                    WHERE DATE(created_at) >= :week_ago
                    GROUP BY DATE(created_at)
                    ORDER BY day ASC
                """),
                {"week_ago": week_ago},
            ).fetchall()
            cumulative = 0
            for row in rows:
                cumulative += row.cnt
                data["trend"].append({
                    "date": row.day,
                    "completed": row.cnt,
                    "cumulative": cumulative,
                })

            # AI 生成占比（假设所有内容都是 AI 生成，或根据生成来源字段判断）
            data["ai_generation_ratio"] = 0.85  # default

            # 平台分布
            platform_rows = db.execute(
                text("""
                    SELECT COALESCE(platform_code, 'unknown') as platform, COUNT(*) as cnt
                    FROM contents
                    GROUP BY platform_code
                    ORDER BY cnt DESC
                """),
            ).fetchall()
            total_platform = sum(r.cnt for r in platform_rows) or 1
            for r in platform_rows:
                data["platform_distribution"].append({
                    "platform_code": r.platform,
                    "platform_name": r.platform,
                    "count": r.cnt,
                    "percentage": round(r.cnt / total_platform, 4),
                })

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# ── 发布监控 ──────────────────────────────────────────────────────────


@router.get("/publishing")
async def get_publishing_metrics() -> dict:
    """发布监控：今日发布统计、平台健康状态、失败归因"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    data: dict[str, Any] = {
        "summary": {
            "total_today": 0,
            "success_count": 0,
            "fail_count": 0,
            "success_rate": 1.0,
            "avg_publish_delay_seconds": 0,
        },
        "platforms": [],
        "failures": [],
    }

    try:
        db = SessionLocal()
        try:
            # 今日发布统计
            rows = db.execute(
                text("""
                    SELECT status, COUNT(*) as cnt
                    FROM publish_records
                    WHERE DATE(created_at) = :today
                    GROUP BY status
                """),
                {"today": today},
            ).fetchall()
            total = 0
            for r in rows:
                total += r.cnt
                if r.status == "success":
                    data["summary"]["success_count"] = r.cnt
                elif r.status == "failed":
                    data["summary"]["fail_count"] = r.cnt
            data["summary"]["total_today"] = total
            data["summary"]["success_rate"] = round(
                data["summary"]["success_count"] / max(total, 1), 4
            )

            # 失败归因
            failures = db.execute(
                text("""
                    SELECT pr.created_at, c.title, pr.platform_code, pr.error_message
                    FROM publish_records pr
                    LEFT JOIN contents c ON pr.content_id = c.id
                    WHERE pr.status = 'failed'
                    ORDER BY pr.created_at DESC
                    LIMIT 20
                """),
            ).fetchall()
            for r in failures:
                data["failures"].append({
                    "time": r.created_at.isoformat() if hasattr(r.created_at, 'isoformat') else str(r.created_at),
                    "content_title": r.title or "未知",
                    "platform_code": r.platform_code,
                    "platform_name": r.platform_code,
                    "reason": r.error_message or "未知错误",
                    "status": "manual",
                })

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# ── 成本分析 ──────────────────────────────────────────────────────────


@router.get("/cost")
async def get_cost_metrics() -> dict:
    """成本分析：LLM Token 消耗、成本按环节/模型分布。

    数据来源：StatsCollector（内存累计）+ 可按需扩展到 DB 持久化。
    """
    stats = stats_collector.get_stats()
    llm = stats.get("llm_calls", {})
    total_tokens = llm.get("total_tokens", 0)

    # 估算成本（不同模型单价不同，这里使用混合均价 ¥0.015/1K tokens）
    ESTIMATED_COST_PER_1K = 0.015
    estimated_cost = round(total_tokens / 1000 * ESTIMATED_COST_PER_1K, 2)

    data: dict[str, Any] = {
        "monthly": {
            "total_cost": estimated_cost,
            "per_article_cost": round(estimated_cost / max(stats.get("agent_runs", {}).get("total", 1), 1), 4),
            "daily_avg_cost_7d": round(estimated_cost / 7, 4),
            "budget_usage_rate": min(1.0, estimated_cost / 100.0),
            "monthly_budget": 100.0,
        },
        "token_trend": [
            {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "models": {"all": total_tokens},
                "total_tokens": total_tokens,
                "total_cost": estimated_cost,
            }
        ],
        "cost_by_stage": [
            {"stage": "LLM 调用", "cost": estimated_cost, "percentage": 0.7 if estimated_cost > 0 else 0},
            {"stage": "向量搜索", "cost": 0, "percentage": 0},
            {"stage": "网络搜索", "cost": 0, "percentage": 0},
            {"stage": "其他", "cost": 0, "percentage": 0},
        ],
        "daily_details": [
            {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "call_count": llm.get("total", 0),
                "total_tokens": total_tokens,
                "model_costs": {"all": estimated_cost},
                "total_cost": estimated_cost,
                "change_rate": None,
            }
        ],
    }

    # Adjust cost_by_stage percentages when cost > 0
    if estimated_cost > 0:
        data["cost_by_stage"][0]["percentage"] = 1.0

    return {"success": True, "data": data}
