from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Body
from sqlalchemy import text

from src.api.auth import get_current_user
from src.database import SessionLocal
from src.models import User

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])

_DEFAULT_MONTHLY_BUDGET = 500.0

_PLATFORM_NAME_MAP: dict[str, str] = {
    "xiaohongshu": "小红书",
    "wechat": "微信公众号",
    "zhihu": "知乎",
    "bilibili": "哔哩哔哩",
    "toutiao": "今日头条",
    "weibo": "微博",
    "jianshu": "简书",
    "csdn": "CSDN",
    "tech-doc": "技术文档",
    "copywriting": "文案",
    "unknown": "未知平台",
}


def _resolve_name(code: str, db_name: str | None) -> str:
    if db_name and db_name != code:
        return db_name
    return _PLATFORM_NAME_MAP.get(code, code)


# ─────────────────────────────────────────────────────────────────
# /production
# ─────────────────────────────────────────────────────────────────

@router.get("/production")
async def get_production_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    month_start = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")

    data: dict[str, Any] = {
        "daily": {"count": 0, "change_vs_yesterday": 0.0},
        "weekly": {"count": 0, "change_vs_last_week": 0.0},
        "avg_view_count": 0,
        "change_vs_last_week_views": 0.0,
        "avg_engagement_rate": 0.0,
        "avg_end_to_end_time_min": 0.0,
        "trend": [],
        "platform_distribution": [],
    }

    try:
        db = SessionLocal()
        try:
            # ── 今日产量 ──
            today_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :uid AND DATE(created_at) = :d"),
                {"uid": current_user.id, "d": today},
            ).scalar() or 0

            yesterday_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :uid AND DATE(created_at) = :d"),
                {"uid": current_user.id, "d": yesterday},
            ).scalar() or 0

            data["daily"]["count"] = today_count
            if yesterday_count > 0:
                data["daily"]["change_vs_yesterday"] = round(
                    (today_count - yesterday_count) / yesterday_count, 4,
                )

            # ── 本周总量 ──
            week_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE user_id = :uid AND DATE(created_at) >= :d"),
                {"uid": current_user.id, "d": week_ago},
            ).scalar() or 0

            last_week_count = db.execute(
                text(
                    "SELECT COUNT(*) FROM contents WHERE user_id = :uid "
                    "AND DATE(created_at) >= :d2 AND DATE(created_at) < :d1"
                ),
                {"uid": current_user.id, "d2": two_weeks_ago, "d1": week_ago},
            ).scalar() or 0

            data["weekly"]["count"] = week_count
            if last_week_count > 0:
                data["weekly"]["change_vs_last_week"] = round(
                    (week_count - last_week_count) / last_week_count, 4,
                )

            # ── 趋势图（7日/30日都用同一份数据）──
            rows = db.execute(
                text(
                    "SELECT DATE(created_at) as day, COUNT(*) as cnt "
                    "FROM contents WHERE user_id = :uid AND DATE(created_at) >= :d "
                    "GROUP BY DATE(created_at) ORDER BY day ASC"
                ),
                {"uid": current_user.id, "d": week_ago},
            ).fetchall()
            cumulative = 0
            for row in rows:
                cumulative += row.cnt
                data["trend"].append({
                    "date": str(row.day),
                    "completed": row.cnt,
                    "cumulative": cumulative,
                })

            # ── 平台分布 ──
            plat_rows = db.execute(
                text(
                    "SELECT COALESCE(c.platform_code, 'unknown') as pc, "
                    "COALESCE(p.name, c.platform_code) as pn, COUNT(*) as cnt "
                    "FROM contents c LEFT JOIN platforms p ON c.platform_code = p.code "
                    "WHERE c.user_id = :uid GROUP BY c.platform_code, p.name ORDER BY cnt DESC"
                ),
                {"uid": current_user.id},
            ).fetchall()
            total_plat = sum(r.cnt for r in plat_rows) or 1
            for row in plat_rows:
                data["platform_distribution"].append({
                    "platform_code": row.pc,
                    "platform_name": _resolve_name(row.pc, row.pn),
                    "count": row.cnt,
                    "percentage": round(row.cnt / total_plat, 4),
                })

            # ── 平均阅读量（本月已发布内容）──
            row = db.execute(
                text(
                    "SELECT AVG(view_count) as avg_views FROM contents "
                    "WHERE user_id = :uid AND status = 'published' "
                    "AND DATE(created_at) >= :d"
                ),
                {"uid": current_user.id, "d": month_start},
            ).scalar() or 0
            data["avg_view_count"] = round(float(row), 1) if row else 0

            # 上周平均阅读量（用于环比）
            lw_views = db.execute(
                text(
                    "SELECT AVG(view_count) as avg_views FROM contents "
                    "WHERE user_id = :uid AND status = 'published' "
                    "AND DATE(created_at) >= :d2 AND DATE(created_at) < :d1"
                ),
                {"uid": current_user.id, "d2": two_weeks_ago, "d1": week_ago},
            ).scalar() or 0
            if lw_views and float(lw_views) > 0:
                data["change_vs_last_week_views"] = round(
                    (float(data["avg_view_count"]) - float(lw_views)) / float(lw_views), 4,
                )

            # ── 平均互动率（点赞+评论）/ 阅读 ──
            eng = db.execute(
                text(
                    "SELECT AVG(CASE WHEN view_count > 0 "
                    "THEN (like_count::float / view_count) ELSE 0 END) "
                    "FROM contents WHERE user_id = :uid AND status = 'published' "
                    "AND DATE(created_at) >= :d"
                ),
                {"uid": current_user.id, "d": month_start},
            ).scalar() or 0
            data["avg_engagement_rate"] = round(float(eng), 4) if eng else 0.0

            # ── 端到端完成时长：任务创建 → 内容发布 ──
            e2e = db.execute(
                text(
                    "SELECT AVG("
                    "  EXTRACT(EPOCH FROM ("
                    "    (SELECT COALESCE(MIN(pr.published_at), c.updated_at) FROM publish_records pr "
                    "     WHERE pr.content_id = c.id AND pr.status = 'success') - c.created_at"
                    "  )) / 60.0"
                    ") FROM contents c "
                    "WHERE c.user_id = :uid AND c.status = 'published'"
                ),
                {"uid": current_user.id},
            ).scalar() or 0
            data["avg_end_to_end_time_min"] = round(float(e2e), 1) if e2e else 0.0

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# ─────────────────────────────────────────────────────────────────
# /publishing
# ─────────────────────────────────────────────────────────────────

@router.get("/publishing")
async def get_publishing_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    two_weeks_ago = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")

    data: dict[str, Any] = {
        "summary": {
            "total_today": 0,
            "success_count": 0,
            "fail_count": 0,
            "success_rate": 1.0,
            "avg_publish_delay_seconds": 0,
        },
        "platforms": [],
        "recent_records": [],
    }

    try:
        db = SessionLocal()
        try:
            # ── 今日发布概况 ──
            rows = db.execute(
                text(
                    "SELECT pr.status, COUNT(*) as cnt "
                    "FROM publish_records pr JOIN contents c ON c.id = pr.content_id "
                    "WHERE c.user_id = :uid AND DATE(pr.created_at) = :d "
                    "GROUP BY pr.status"
                ),
                {"uid": current_user.id, "d": today},
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
                data["summary"]["success_count"] / max(total, 1), 4,
            )

            # ── 平均发布延迟 ──
            avg_delay = db.execute(
                text(
                    "SELECT AVG(EXTRACT(EPOCH FROM (pr.published_at - pr.created_at))) "
                    "FROM publish_records pr JOIN contents c ON c.id = pr.content_id "
                    "WHERE c.user_id = :uid AND pr.status = 'success' "
                    "AND pr.published_at IS NOT NULL"
                ),
                {"uid": current_user.id},
            ).scalar() or 0
            data["summary"]["avg_publish_delay_seconds"] = round(float(avg_delay))

            # ── 平台健康（近7天 + 真实周趋势）──
            plat_rows = db.execute(
                text(
                    "SELECT pr.platform_code, COALESCE(p.name, pr.platform_code) as pn, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN pr.status = 'success' THEN 1 ELSE 0 END) as sc "
                    "FROM publish_records pr "
                    "JOIN contents c ON c.id = pr.content_id "
                    "LEFT JOIN platforms p ON pr.platform_code = p.code "
                    "WHERE c.user_id = :uid AND DATE(pr.created_at) >= :d "
                    "GROUP BY pr.platform_code, p.name"
                ),
                {"uid": current_user.id, "d": week_ago},
            ).fetchall()

            for row in plat_rows:
                w_total = row.total or 0
                w_success = row.sc or 0
                week_rate = round(w_success / max(w_total, 1), 4)

                # 上周成功率（用于趋势计算）
                lw_row = db.execute(
                    text(
                        "SELECT COUNT(*) as total, "
                        "SUM(CASE WHEN pr.status = 'success' THEN 1 ELSE 0 END) as sc "
                        "FROM publish_records pr "
                        "JOIN contents c ON c.id = pr.content_id "
                        "WHERE c.user_id = :uid AND pr.platform_code = :pc "
                        "AND DATE(pr.created_at) >= :d2 AND DATE(pr.created_at) < :d1"
                    ),
                    {"uid": current_user.id, "pc": row.platform_code,
                     "d2": two_weeks_ago, "d1": week_ago},
                ).fetchone()
                lw_total = lw_row.total or 0 if lw_row else 0
                lw_success = lw_row.sc or 0 if lw_row else 0
                lw_rate = round(lw_success / max(lw_total, 1), 4) if lw_total > 0 else None

                week_trend = round(week_rate - lw_rate, 4) if lw_rate is not None else 0.0

                data["platforms"].append({
                    "platform_code": row.platform_code,
                    "platform_name": _resolve_name(row.platform_code, row.pn),
                    "status": "healthy" if week_rate >= 0.95 else ("degraded" if week_rate >= 0.7 else "down"),
                    "today_success": 0,
                    "today_total": 0,
                    "last_failure_time": None,
                    "last_failure_reason": None,
                    "week_success_rate": week_rate,
                    "week_success_count": w_success,
                    "week_total_count": w_total,
                    "week_trend": week_trend,
                })

            # ── 今日各平台数据 ──
            today_rows = db.execute(
                text(
                    "SELECT pr.platform_code, COUNT(*) as total, "
                    "SUM(CASE WHEN pr.status = 'success' THEN 1 ELSE 0 END) as sc "
                    "FROM publish_records pr JOIN contents c ON c.id = pr.content_id "
                    "WHERE c.user_id = :uid AND DATE(pr.created_at) = :d "
                    "GROUP BY pr.platform_code"
                ),
                {"uid": current_user.id, "d": today},
            ).fetchall()
            today_map = {r.platform_code: (r.sc or 0, r.total or 0) for r in today_rows}
            for plat in data["platforms"]:
                if plat["platform_code"] in today_map:
                    plat["today_success"], plat["today_total"] = today_map[plat["platform_code"]]

            # ── 最近失败记录（用于平台卡片显示）──
            fail_rows = db.execute(
                text(
                    "SELECT pr.platform_code, pr.created_at, pr.error_message "
                    "FROM publish_records pr JOIN contents c ON c.id = pr.content_id "
                    "WHERE c.user_id = :uid AND pr.status = 'failed' "
                    "AND pr.id IN ("
                    "  SELECT MAX(pr2.id) FROM publish_records pr2 "
                    "  JOIN contents c2 ON c2.id = pr2.content_id "
                    "  WHERE c2.user_id = :uid AND pr2.status = 'failed' "
                    "  GROUP BY pr2.platform_code"
                    ")"
                ),
                {"uid": current_user.id},
            ).fetchall()
            fail_map = {r.platform_code: r for r in fail_rows}
            for plat in data["platforms"]:
                if plat["platform_code"] in fail_map:
                    r = fail_map[plat["platform_code"]]
                    ts = r.created_at
                    plat["last_failure_time"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    plat["last_failure_reason"] = r.error_message

            # ── 最近发布记录（最近20条）──
            recent = db.execute(
                text(
                    "SELECT pr.created_at, c.title, pr.platform_code, "
                    "COALESCE(p.name, pr.platform_code) as pn, "
                    "pr.status, c.view_count "
                    "FROM publish_records pr "
                    "JOIN contents c ON c.id = pr.content_id "
                    "LEFT JOIN platforms p ON pr.platform_code = p.code "
                    "WHERE c.user_id = :uid "
                    "ORDER BY pr.created_at DESC LIMIT 20"
                ),
                {"uid": current_user.id},
            ).fetchall()
            data["recent_records"] = [
                {
                    "time": r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
                    "title": r.title or "无标题",
                    "platform_code": r.platform_code,
                    "platform_name": _resolve_name(r.platform_code, r.pn),
                    "status": r.status,
                    "view_count": r.view_count or 0,
                }
                for r in recent
            ]

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# ─────────────────────────────────────────────────────────────────
# /cost
# ─────────────────────────────────────────────────────────────────

@router.get("/cost")
async def get_cost_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    month_start = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    budget = _DEFAULT_MONTHLY_BUDGET

    data: dict[str, Any] = {
        "monthly": {
            "total_cost": 0.0,
            "per_article_cost": 0.0,
            "daily_avg_cost_7d": 0.0,
            "budget_usage_rate": 0.0,
            "monthly_budget": budget,
        },
        "cost_trend": [],
        "cost_by_model": [],
        "daily_details": [],
    }

    try:
        db = SessionLocal()
        try:
            # ── 月度成本汇总 ──
            monthly = db.execute(
                text(
                    "SELECT COALESCE(SUM(total_cost), 0) as tc, "
                    "COALESCE(SUM(total_tokens), 0) as tt, COUNT(*) as cnt "
                    "FROM agent_tasks WHERE user_id = :uid "
                    "AND DATE(created_at) >= :d "
                    "AND status IN ('completed', 'partial')"
                ),
                {"uid": current_user.id, "d": month_start},
            ).one()

            total_cost = float(monthly.tc) if monthly else 0.0

            # ── 发布量（用于单篇均摊）──
            pub_count = db.execute(
                text(
                    "SELECT COUNT(*) FROM publish_records pr "
                    "JOIN contents c ON c.id = pr.content_id "
                    "WHERE c.user_id = :uid AND pr.status = 'success' "
                    "AND DATE(pr.created_at) >= :d"
                ),
                {"uid": current_user.id, "d": month_start},
            ).scalar() or 0

            # ── 近7日日均 ──
            week_cost = db.execute(
                text(
                    "SELECT COALESCE(SUM(total_cost), 0) FROM agent_tasks "
                    "WHERE user_id = :uid AND DATE(created_at) >= :d "
                    "AND status IN ('completed', 'partial')"
                ),
                {"uid": current_user.id, "d": week_ago},
            ).scalar() or 0

            data["monthly"]["total_cost"] = round(total_cost, 2)
            data["monthly"]["per_article_cost"] = round(total_cost / max(pub_count, 1), 4)
            data["monthly"]["daily_avg_cost_7d"] = round(float(week_cost) / 7, 2)
            data["monthly"]["budget_usage_rate"] = round(min(1.0, total_cost / max(budget, 1)), 4)

            # ── 费用趋势（按日期 × 模型，Y轴=金额）──
            trend_rows = db.execute(
                text(
                    "SELECT DATE(created_at) as day, "
                    "COALESCE(NULLIF(model_name, ''), 'unknown') as model, "
                    "SUM(total_cost) as cost "
                    "FROM agent_tasks WHERE user_id = :uid "
                    "AND status IN ('completed', 'partial') "
                    "GROUP BY DATE(created_at), COALESCE(NULLIF(model_name, ''), 'unknown') "
                    "ORDER BY day ASC LIMIT 600"
                ),
                {"uid": current_user.id},
            ).fetchall()

            day_map: dict[str, dict[str, float]] = {}
            for row in trend_rows:
                day = str(row.day)
                if day not in day_map:
                    day_map[day] = {}
                day_map[day][row.model] = round(float(row.cost or 0), 4)

            data["cost_trend"] = [
                {
                    "date": day,
                    "models": models,
                    "total_cost": round(sum(models.values()), 2),
                }
                for day, models in sorted(day_map.items())
            ]

            # ── 各模型费用占比 ──
            model_totals: dict[str, float] = {}
            for row in trend_rows:
                model_totals[row.model] = model_totals.get(row.model, 0) + float(row.cost or 0)
            total_model_cost = sum(model_totals.values()) or 1.0
            for model, cost in sorted(model_totals.items(), key=lambda x: -x[1]):
                data["cost_by_model"].append({
                    "model": model,
                    "cost": round(cost, 2),
                    "percentage": round(cost / total_model_cost, 4),
                })

            # ── 每日成本明细 ──
            detail_rows = db.execute(
                text(
                    "SELECT DATE(created_at) as day, COUNT(*) as call_count, "
                    "SUM(total_cost) as cost "
                    "FROM agent_tasks WHERE user_id = :uid "
                    "AND status IN ('completed', 'partial') "
                    "GROUP BY DATE(created_at) ORDER BY day DESC LIMIT 62"
                ),
                {"uid": current_user.id},
            ).fetchall()

            prev_cost: float | None = None
            for row in sorted(detail_rows, key=lambda r: str(r.day)):
                day_cost = round(float(row.cost or 0), 2)
                change = None
                if prev_cost is not None and prev_cost > 0:
                    change = round((day_cost - prev_cost) / prev_cost, 4)
                data["daily_details"].append({
                    "date": str(row.day),
                    "call_count": row.call_count or 0,
                    "total_cost": day_cost,
                    "change_rate": change,
                })
                prev_cost = day_cost

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


@router.patch("/budget")
async def set_monthly_budget(
    budget: float = Body(..., ge=0, le=100000),
    current_user: User = Depends(get_current_user),
) -> dict:
    return {"success": True, "data": {"monthly_budget": budget}}


# ─────────────────────────────────────────────────────────────────
# /system
# ─────────────────────────────────────────────────────────────────

@router.get("/system")
async def get_system_metrics(
    current_user: User = Depends(get_current_user),
) -> dict:
    data: dict[str, Any] = {
        "worker": {"running": False},
        "tasks": {"active": 0, "failed": 0},
        "llm": {"available": True},
        "recent_failures": [],
    }

    try:
        db = SessionLocal()
        try:
            # ── 失败任务数（本月）──
            month_start = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")
            failed_count = db.execute(
                text(
                    "SELECT COUNT(*) FROM agent_tasks WHERE user_id = :uid "
                    "AND status = 'failed' AND DATE(created_at) >= :d"
                ),
                {"uid": current_user.id, "d": month_start},
            ).scalar() or 0
            data["tasks"]["failed"] = failed_count

            # ── 活跃任务（pending + running）──
            active = db.execute(
                text(
                    "SELECT COUNT(*) FROM agent_tasks WHERE user_id = :uid "
                    "AND status IN ('pending', 'running')"
                ),
                {"uid": current_user.id},
            ).scalar() or 0
            data["tasks"]["active"] = active

            # ── Worker 状态（检查是否有最近活跃的 subtask）──
            recent_subtask = db.execute(
                text(
                    "SELECT MAX(created_at) FROM agent_subtasks s "
                    "JOIN agent_tasks t ON t.id = s.task_id "
                    "WHERE t.user_id = :uid"
                ),
                {"uid": current_user.id},
            ).scalar()
            if recent_subtask:
                diff = (datetime.now(timezone.utc) - recent_subtask).total_seconds()
                data["worker"]["running"] = diff < 300  # 5 min 内有活动视为运行中

            # ── 最近失败任务（5条）──
            failures = db.execute(
                text(
                    "SELECT id, created_at, task_type, error_message "
                    "FROM agent_tasks WHERE user_id = :uid AND status = 'failed' "
                    "ORDER BY created_at DESC LIMIT 5"
                ),
                {"uid": current_user.id},
            ).fetchall()
            data["recent_failures"] = [
                {
                    "task_id": str(r.id),
                    "time": r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
                    "task_type": r.task_type or "内容创作",
                    "error": (r.error_message or "未知错误")[:120],
                }
                for r in failures
            ]

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# =====================================================================
# 向后兼容：老路径 /api/v1/admin/metrics/* 重定向到 /api/v1/metrics/*
# =====================================================================

from fastapi.responses import RedirectResponse

legacy_admin_router = APIRouter(prefix="/api/v1/admin/metrics", include_in_schema=False)


@legacy_admin_router.get("/production")
def _redirect_production():
    return RedirectResponse(url="/api/v1/metrics/production", status_code=307)


@legacy_admin_router.get("/publishing")
def _redirect_publishing():
    return RedirectResponse(url="/api/v1/metrics/publishing", status_code=307)


@legacy_admin_router.get("/cost")
def _redirect_cost():
    return RedirectResponse(url="/api/v1/metrics/cost", status_code=307)
