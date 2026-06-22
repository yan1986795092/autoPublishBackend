from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text

from src.api.auth import get_current_user
from src.database import SessionLocal
from src.models import User

router = APIRouter(prefix="/api/v1/admin/metrics", tags=["admin"])

_DEFAULT_MONTHLY_BUDGET = 100.0
_ESTIMATED_COST_PER_1K = 0.015


@router.get("/production")
async def get_production_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")

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
            row = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :user_id AND DATE(created_at) = :today"),
                {"user_id": current_user.id, "today": today},
            ).scalar()
            data["daily"]["count"] = row or 0

            yesterday_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :user_id AND DATE(created_at) = :yesterday"),
                {"user_id": current_user.id, "yesterday": yesterday},
            ).scalar() or 0
            if yesterday_count > 0:
                data["daily"]["change_vs_yesterday"] = round(
                    (data["daily"]["count"] - yesterday_count) / yesterday_count,
                    4,
                )

            week_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :user_id AND DATE(created_at) >= :week_ago"),
                {"user_id": current_user.id, "week_ago": week_ago},
            ).scalar()
            data["weekly"]["count"] = week_count or 0

            last_week_count = db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM contents
                    WHERE user_id = :user_id
                      AND DATE(created_at) >= :two_weeks_ago
                      AND DATE(created_at) < :week_ago
                    """
                ),
                {
                    "user_id": current_user.id,
                    "two_weeks_ago": two_weeks_ago,
                    "week_ago": week_ago,
                },
            ).scalar() or 0
            if last_week_count > 0:
                data["weekly"]["change_vs_last_week"] = round(
                    (data["weekly"]["count"] - last_week_count) / last_week_count,
                    4,
                )

            rows = db.execute(
                text(
                    """
                    SELECT DATE(created_at) as day, COUNT(*) as cnt
                    FROM contents
                    WHERE user_id = :user_id
                      AND DATE(created_at) >= :week_ago
                    GROUP BY DATE(created_at)
                    ORDER BY day ASC
                    """
                ),
                {"user_id": current_user.id, "week_ago": week_ago},
            ).fetchall()
            cumulative = 0
            for row in rows:
                cumulative += row.cnt
                data["trend"].append(
                    {
                        "date": row.day,
                        "completed": row.cnt,
                        "cumulative": cumulative,
                    }
                )

            ai_total = db.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM contents
                    WHERE user_id = :user_id
                      AND metadata_json IS NOT NULL
                      AND metadata_json != ''
                    """
                ),
                {"user_id": current_user.id},
            ).scalar() or 0
            total = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :user_id"),
                {"user_id": current_user.id},
            ).scalar() or 0
            data["ai_generation_ratio"] = round(ai_total / max(total, 1), 4)

            avg_time = db.execute(
                text(
                    """
                    SELECT AVG(s.time_elapsed)
                    FROM agent_subtasks s
                    JOIN agent_tasks t ON t.id = s.task_id
                    WHERE t.user_id = :user_id
                      AND s.time_elapsed > 0
                      AND s.time_elapsed IS NOT NULL
                    """
                ),
                {"user_id": current_user.id},
            ).scalar()
            data["avg_completion_time_min"] = round((avg_time or 0) / 60, 2)

            platform_rows = db.execute(
                text(
                    """
                    SELECT
                        COALESCE(c.platform_code, 'unknown') as platform,
                        COALESCE(p.name, c.platform_code) as platform_name,
                        COUNT(*) as cnt
                    FROM contents c
                    LEFT JOIN platforms p ON c.platform_code = p.code
                    WHERE c.user_id = :user_id
                    GROUP BY c.platform_code, p.name
                    ORDER BY cnt DESC
                    """
                ),
                {"user_id": current_user.id},
            ).fetchall()
            total_platform = sum(row.cnt for row in platform_rows) or 1
            for row in platform_rows:
                data["platform_distribution"].append(
                    {
                        "platform_code": row.platform,
                        "platform_name": row.platform_name,
                        "count": row.cnt,
                        "percentage": round(row.cnt / total_platform, 4),
                    }
                )
        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


@router.get("/publishing")
async def get_publishing_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

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
            rows = db.execute(
                text(
                    """
                    SELECT pr.status, COUNT(*) as cnt
                    FROM publish_records pr
                    JOIN contents c ON c.id = pr.content_id
                    WHERE c.user_id = :user_id
                      AND DATE(pr.created_at) = :today
                    GROUP BY pr.status
                    """
                ),
                {"user_id": current_user.id, "today": today},
            ).fetchall()
            total = 0
            for row in rows:
                total += row.cnt
                if row.status == "success":
                    data["summary"]["success_count"] = row.cnt
                elif row.status == "failed":
                    data["summary"]["fail_count"] = row.cnt
            data["summary"]["total_today"] = total
            data["summary"]["success_rate"] = round(
                data["summary"]["success_count"] / max(total, 1),
                4,
            )

            avg_delay = db.execute(
                text(
                    """
                    SELECT AVG(EXTRACT(EPOCH FROM (pr.published_at - pr.created_at)))
                    FROM publish_records pr
                    JOIN contents c ON c.id = pr.content_id
                    WHERE c.user_id = :user_id
                      AND pr.status = 'success'
                      AND pr.published_at IS NOT NULL
                      AND pr.created_at IS NOT NULL
                    """
                ),
                {"user_id": current_user.id},
            ).scalar()
            data["summary"]["avg_publish_delay_seconds"] = round(avg_delay or 0)

            platform_rows = db.execute(
                text(
                    """
                    SELECT
                        pr.platform_code,
                        COALESCE(p.name, pr.platform_code) as platform_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN pr.status = 'success' THEN 1 ELSE 0 END) as success_count
                    FROM publish_records pr
                    JOIN contents c ON c.id = pr.content_id
                    LEFT JOIN platforms p ON pr.platform_code = p.code
                    WHERE c.user_id = :user_id
                      AND DATE(pr.created_at) >= :week_ago
                    GROUP BY pr.platform_code, p.name
                    """
                ),
                {"user_id": current_user.id, "week_ago": week_ago},
            ).fetchall()

            for row in platform_rows:
                week_total = row.total or 0
                week_success = row.success_count or 0
                week_rate = round(week_success / max(week_total, 1), 4)
                data["platforms"].append(
                    {
                        "platform_code": row.platform_code,
                        "platform_name": row.platform_name,
                        "status": "healthy" if week_rate >= 0.95 else ("degraded" if week_rate >= 0.7 else "down"),
                        "today_success": 0,
                        "today_total": 0,
                        "last_failure_time": None,
                        "last_failure_reason": None,
                        "week_success_rate": week_rate,
                        "week_trend": week_rate - 1.0,
                    }
                )

            today_rows = db.execute(
                text(
                    """
                    SELECT pr.platform_code,
                           COUNT(*) as total,
                           SUM(CASE WHEN pr.status = 'success' THEN 1 ELSE 0 END) as success_count
                    FROM publish_records pr
                    JOIN contents c ON c.id = pr.content_id
                    WHERE c.user_id = :user_id
                      AND DATE(pr.created_at) = :today
                    GROUP BY pr.platform_code
                    """
                ),
                {"user_id": current_user.id, "today": today},
            ).fetchall()
            for row in today_rows:
                for platform in data["platforms"]:
                    if platform["platform_code"] == row.platform_code:
                        platform["today_success"] = row.success_count or 0
                        platform["today_total"] = row.total or 0
                        break

            last_failures = db.execute(
                text(
                    """
                    SELECT pr.platform_code, pr.created_at, pr.error_message
                    FROM publish_records pr
                    JOIN contents c ON c.id = pr.content_id
                    WHERE c.user_id = :user_id
                      AND pr.status = 'failed'
                      AND pr.id IN (
                          SELECT MAX(pr2.id)
                          FROM publish_records pr2
                          JOIN contents c2 ON c2.id = pr2.content_id
                          WHERE c2.user_id = :user_id
                            AND pr2.status = 'failed'
                          GROUP BY pr2.platform_code
                      )
                    """
                ),
                {"user_id": current_user.id},
            ).fetchall()
            for row in last_failures:
                for platform in data["platforms"]:
                    if platform["platform_code"] == row.platform_code:
                        platform["last_failure_time"] = (
                            row.created_at.isoformat() if hasattr(row.created_at, "isoformat") else str(row.created_at)
                        )
                        platform["last_failure_reason"] = row.error_message
                        break

            one_hour_ago_dt = datetime.now(timezone.utc) - timedelta(hours=1)
            failures = db.execute(
                text(
                    """
                    SELECT pr.created_at, c.title, pr.platform_code, pr.error_message
                    FROM publish_records pr
                    JOIN contents c ON c.id = pr.content_id
                    WHERE c.user_id = :user_id
                      AND pr.status = 'failed'
                    ORDER BY pr.created_at DESC
                    LIMIT 20
                    """
                ),
                {"user_id": current_user.id},
            ).fetchall()
            platform_name_map = {item["platform_code"]: item["platform_name"] for item in data["platforms"]}
            for row in failures:
                created_dt = row.created_at
                try:
                    if created_dt is None:
                        is_recent = False
                    elif created_dt.tzinfo is None:
                        is_recent = created_dt.replace(tzinfo=timezone.utc) >= one_hour_ago_dt
                    else:
                        is_recent = created_dt >= one_hour_ago_dt
                except Exception:
                    is_recent = False
                data["failures"].append(
                    {
                        "time": created_dt.isoformat() if hasattr(created_dt, "isoformat") else str(created_dt or ""),
                        "content_title": row.title or "未知",
                        "platform_code": row.platform_code,
                        "platform_name": platform_name_map.get(row.platform_code, row.platform_code),
                        "reason": row.error_message or "未知错误",
                        "status": "retrying" if is_recent else "manual",
                    }
                )
        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


@router.get("/cost")
async def get_cost_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    month_start = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")

    data: dict[str, Any] = {
        "monthly": {
            "total_cost": 0,
            "per_article_cost": 0,
            "daily_avg_cost_7d": 0,
            "budget_usage_rate": 0,
            "monthly_budget": _DEFAULT_MONTHLY_BUDGET,
        },
        "token_trend": [],
        "cost_by_stage": [],
        "daily_details": [],
    }

    try:
        db = SessionLocal()
        try:
            monthly = db.execute(
                text(
                    """
                    SELECT
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(total_cost), 0) as total_cost,
                        COUNT(*) as task_count
                    FROM agent_tasks
                    WHERE user_id = :user_id
                      AND DATE(created_at) >= :month_start
                      AND status IN ('completed', 'partial')
                    """
                ),
                {"user_id": current_user.id, "month_start": month_start},
            ).one()

            total_cost = monthly.total_cost or 0
            total_tokens = monthly.total_tokens or 0

            if total_cost == 0 and total_tokens > 0:
                total_cost = round(total_tokens / 1000 * _ESTIMATED_COST_PER_1K, 2)

            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            week_cost = db.execute(
                text(
                    """
                    SELECT COALESCE(SUM(total_cost), 0)
                    FROM agent_tasks
                    WHERE user_id = :user_id
                      AND DATE(created_at) >= :week_ago
                      AND status IN ('completed', 'partial')
                    """
                ),
                {"user_id": current_user.id, "week_ago": week_ago},
            ).scalar() or 0

            total_contents = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :user_id"),
                {"user_id": current_user.id},
            ).scalar() or 0

            data["monthly"] = {
                "total_cost": round(total_cost, 4),
                "per_article_cost": round(total_cost / max(total_contents, 1), 4),
                "daily_avg_cost_7d": round(week_cost / 7, 4),
                "budget_usage_rate": round(min(1.0, total_cost / max(_DEFAULT_MONTHLY_BUDGET, 1)), 4),
                "monthly_budget": _DEFAULT_MONTHLY_BUDGET,
            }

            trend_rows = db.execute(
                text(
                    """
                    SELECT
                        DATE(created_at) as date,
                        SUM(total_tokens) as tokens,
                        SUM(total_cost) as cost
                    FROM agent_tasks
                    WHERE user_id = :user_id
                      AND status IN ('completed', 'partial')
                    GROUP BY DATE(created_at)
                    ORDER BY date ASC
                    LIMIT 60
                    """
                ),
                {"user_id": current_user.id},
            ).fetchall()
            data["token_trend"] = [
                {
                    "date": row.date,
                    "models": {"all": row.tokens or 0},
                    "total_tokens": row.tokens or 0,
                    "total_cost": round(row.cost or 0, 4),
                }
                for row in trend_rows
            ]

            stage_rows = db.execute(
                text(
                    """
                    SELECT
                        s.agent_type,
                        SUM(s.tokens_used) as total_tokens,
                        SUM(s.time_elapsed) as total_time
                    FROM agent_subtasks s
                    JOIN agent_tasks t ON t.id = s.task_id
                    WHERE t.user_id = :user_id
                    GROUP BY s.agent_type
                    ORDER BY total_tokens DESC
                    """
                ),
                {"user_id": current_user.id},
            ).fetchall()
            total_stage_tokens = sum(row.total_tokens or 0 for row in stage_rows) or 1
            stage_names = {
                "WritingAgent": "内容创作",
                "SEOAgent": "SEO 优化",
                "ReviewAgent": "质量审核",
                "ManagerAgent": "任务编排",
                "RewriteAgent": "内容改写",
                "TranslationAgent": "翻译",
                "FactCheckAgent": "事实核查",
                "ResearchAgent": "调研",
                "HotTopicAgent": "热点分析",
            }
            for row in stage_rows:
                if not row.agent_type or (row.total_tokens or 0) == 0:
                    continue
                ratio = (row.total_tokens or 0) / total_stage_tokens
                data["cost_by_stage"].append(
                    {
                        "stage": stage_names.get(row.agent_type, row.agent_type),
                        "cost": round(total_cost * ratio, 4),
                        "percentage": round(ratio, 4),
                    }
                )

            detail_rows = db.execute(
                text(
                    """
                    SELECT
                        DATE(created_at) as date,
                        COUNT(*) as call_count,
                        SUM(total_tokens) as tokens,
                        SUM(total_cost) as cost
                    FROM agent_tasks
                    WHERE user_id = :user_id
                      AND status IN ('completed', 'partial')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                    LIMIT 31
                    """
                ),
                {"user_id": current_user.id},
            ).fetchall()

            prev_cost: float | None = None
            for row in reversed(detail_rows):
                day_cost = round(row.cost or 0, 4)
                change = None
                if prev_cost is not None and prev_cost > 0:
                    change = round((day_cost - prev_cost) / prev_cost, 4)
                data["daily_details"].append(
                    {
                        "date": row.date,
                        "call_count": row.call_count or 0,
                        "total_tokens": row.tokens or 0,
                        "model_costs": {"all": day_cost},
                        "total_cost": day_cost,
                        "change_rate": change,
                    }
                )
                prev_cost = day_cost
        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}
