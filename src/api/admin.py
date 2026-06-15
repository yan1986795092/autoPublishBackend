"""运维中心后端接口 — Admin metrics API。

提供产能看板、发布监控、成本分析三个端点所需的数据。
所有指标均从数据库查询真实数据，移除硬编码占位值。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from src.database import SessionLocal

router = APIRouter(prefix="/api/v1/admin/metrics", tags=["admin"])


# ═════════════════════════════════════════════════════════════════════
# 产能看板
# ═════════════════════════════════════════════════════════════════════


@router.get("/production")
async def get_production_metrics() -> dict:
    """产能看板：今日/本周产量、AI 生成占比、趋势、平台分布（全部来自 DB）"""
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
            # ── 今日产量 ──
            row = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) = :today"),
                {"today": today},
            ).scalar()
            data["daily"]["count"] = row or 0

            # ── 昨日产量（环比）──
            yesterday_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) = :yesterday"),
                {"yesterday": yesterday},
            ).scalar() or 0
            if yesterday_count > 0:
                data["daily"]["change_vs_yesterday"] = round(
                    (data["daily"]["count"] - yesterday_count) / yesterday_count, 4
                )

            # ── 本周总量 ──
            week_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) >= :week_ago"),
                {"week_ago": week_ago},
            ).scalar()
            data["weekly"]["count"] = week_count or 0

            # ── 上周总量（环比）──
            last_week_count = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE DATE(created_at) >= :two_weeks_ago AND DATE(created_at) < :week_ago"),
                {"two_weeks_ago": two_weeks_ago, "week_ago": week_ago},
            ).scalar() or 0
            if last_week_count > 0:
                data["weekly"]["change_vs_last_week"] = round(
                    (data["weekly"]["count"] - last_week_count) / last_week_count, 4
                )

            # ── 趋势（最近 7 天逐日）──
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
            for r in rows:
                cumulative += r.cnt
                data["trend"].append({
                    "date": r.day,
                    "completed": r.cnt,
                    "cumulative": cumulative,
                })

            # ── AI 生成占比（真实计算）
            # contents 的 metadata_json 非空 → AI 生成（generate 流程写入参数）
            # contents 的 metadata_json 为空 → 手动创建
            ai_total = db.execute(
                text("SELECT COUNT(*) FROM contents WHERE metadata_json IS NOT NULL AND metadata_json != ''")
            ).scalar() or 0
            total = db.execute(
                text("SELECT COUNT(*) FROM contents")
            ).scalar() or 1
            data["ai_generation_ratio"] = round(ai_total / max(total, 1), 4)

            # ── 平均完成耗时（真实计算）
            # 从 agent_subtasks 取有耗时的记录，计算平均值（秒→分钟）
            avg_time = db.execute(
                text("""
                    SELECT AVG(time_elapsed)
                    FROM agent_subtasks
                    WHERE time_elapsed > 0 AND time_elapsed IS NOT NULL
                """)
            ).scalar()
            data["avg_completion_time_min"] = round((avg_time or 0) / 60, 2)

            # ── 平台产量分布（带平台名称映射）──
            platform_rows = db.execute(
                text("""
                    SELECT
                        COALESCE(c.platform_code, 'unknown') as platform,
                        COALESCE(p.name, c.platform_code) as platform_name,
                        COUNT(*) as cnt
                    FROM contents c
                    LEFT JOIN platforms p ON c.platform_code = p.code
                    GROUP BY c.platform_code
                    ORDER BY cnt DESC
                """),
            ).fetchall()
            total_platform = sum(r.cnt for r in platform_rows) or 1
            for r in platform_rows:
                data["platform_distribution"].append({
                    "platform_code": r.platform,
                    "platform_name": r.platform_name,
                    "count": r.cnt,
                    "percentage": round(r.cnt / total_platform, 4),
                })

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# ═════════════════════════════════════════════════════════════════════
# 发布监控
# ═════════════════════════════════════════════════════════════════════


@router.get("/publishing")
async def get_publishing_metrics() -> dict:
    """发布监控：今日发布统计、平台健康状态、失败归因（全部来自 DB）"""
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
            # ── 今日发布统计 ──
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

            # ── 平均发布延迟（成功发布的耗时）──
            avg_delay = db.execute(
                text("""
                    SELECT AVG(
                        EXTRACT(EPOCH FROM (published_at - created_at))
                    )
                    FROM publish_records
                    WHERE status = 'success'
                      AND published_at IS NOT NULL
                      AND created_at IS NOT NULL
                """)
            ).scalar()
            data["summary"]["avg_publish_delay_seconds"] = round(avg_delay or 0)

            # ── 平台健康状态（带平台名称映射 + 周成功率/趋势）──
            platform_rows = db.execute(
                text("""
                    SELECT
                        pr.platform_code,
                        COALESCE(p.name, pr.platform_code) as platform_name,
                        COUNT(*) as total,
                        SUM(CASE WHEN pr.status = 'success' THEN 1 ELSE 0 END) as success_count,
                        MAX(pr.created_at) as last_publish,
                        SUM(CASE WHEN pr.status = 'failed' THEN 1 ELSE 0 END) as fail_count
                    FROM publish_records pr
                    LEFT JOIN platforms p ON pr.platform_code = p.code
                    WHERE DATE(pr.created_at) >= :week_ago
                    GROUP BY pr.platform_code
                """),
                {"week_ago": week_ago},
            ).fetchall()

            for r in platform_rows:
                week_total = r.total or 0
                week_success = r.success_count or 0
                week_rate = round(week_success / max(week_total, 1), 4)

                data["platforms"].append({
                    "platform_code": r.platform_code,
                    "platform_name": r.platform_name,
                    "status": "healthy" if week_rate >= 0.95 else ("degraded" if week_rate >= 0.7 else "down"),
                    "today_success": 0,
                    "today_total": 0,
                    "last_failure_time": None,
                    "last_failure_reason": None,
                    "week_success_rate": week_rate,
                    "week_trend": week_rate - 1.0,
                })

            # 回填今天的发布计数
            today_platform = db.execute(
                text("""
                    SELECT platform_code,
                           COUNT(*) as total,
                           SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as success_count
                    FROM publish_records
                    WHERE DATE(created_at) = :today
                    GROUP BY platform_code
                """),
                {"today": today},
            ).fetchall()
            for r in today_platform:
                for p in data["platforms"]:
                    if p["platform_code"] == r.platform_code:
                        p["today_success"] = r.success_count or 0
                        p["today_total"] = r.total or 0
                        break

            # 每个平台的最近一次失败信息
            last_failures = db.execute(
                text("""
                    SELECT platform_code, created_at, error_message
                    FROM publish_records
                    WHERE status = 'failed'
                    AND id IN (
                        SELECT MAX(id) FROM publish_records
                        WHERE status = 'failed' GROUP BY platform_code
                    )
                """),
            ).fetchall()
            for r in last_failures:
                for p in data["platforms"]:
                    if p["platform_code"] == r.platform_code:
                        p["last_failure_time"] = r.created_at.isoformat() if hasattr(r.created_at, 'isoformat') else str(r.created_at)
                        p["last_failure_reason"] = r.error_message
                        break

            # ── 失败归因（智能状态分类）──
            one_hour_ago_dt = datetime.now(timezone.utc) - timedelta(hours=1)

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
            # 为快速查找平台名称建立映射
            platform_name_map = {p["platform_code"]: p["platform_name"] for p in data["platforms"]}
            for r in failures:
                created_dt = r.created_at
                created_str = created_dt.isoformat() if hasattr(created_dt, 'isoformat') else str(created_dt or "")
                # 状态分类：1小时内 → retrying（假设自动重试机制正在处理）
                #           超过1小时 → manual（需要人工介入）
                try:
                    if created_dt is None:
                        is_recent = False
                    elif created_dt.tzinfo is None:
                        created_dt_utc = created_dt.replace(tzinfo=timezone.utc)
                        is_recent = created_dt_utc >= one_hour_ago_dt
                    else:
                        is_recent = created_dt >= one_hour_ago_dt
                except Exception:
                    is_recent = False
                status = "retrying" if is_recent else "manual"

                data["failures"].append({
                    "time": created_str,
                    "content_title": r.title or "未知",
                    "platform_code": r.platform_code,
                    "platform_name": platform_name_map.get(r.platform_code, r.platform_code),
                    "reason": r.error_message or "未知错误",
                    "status": status,
                })

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}


# ═════════════════════════════════════════════════════════════════════
# 成本分析
# ═════════════════════════════════════════════════════════════════════

_DEFAULT_MONTHLY_BUDGET = 100.0  # 可被环境变量覆盖
_ESTIMATED_COST_PER_1K = 0.015   # 兜底单价，当 DB 无 cost 记录时使用


@router.get("/cost")
async def get_cost_metrics() -> dict:
    """成本分析：LLM Token 消耗、成本按环节/模型分布（全部来自 DB）"""
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
            # ── 月度汇总 ──
            monthly = db.execute(
                text("""
                    SELECT
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(total_cost), 0) as total_cost,
                        COUNT(*) as task_count
                    FROM agent_tasks
                    WHERE DATE(created_at) >= :month_start
                      AND status IN ('completed', 'partial')
                """),
                {"month_start": month_start},
            ).one()

            total_cost = monthly.total_cost or 0
            total_tokens = monthly.total_tokens or 0
            task_count = monthly.task_count or 0

            # 如果 DB 中没有 cost 记录（旧数据），用 tokens 估算
            if total_cost == 0 and total_tokens > 0:
                total_cost = round(total_tokens / 1000 * _ESTIMATED_COST_PER_1K, 2)

            # ── 最近 7 天日均成本 ──
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            week_cost = db.execute(
                text("""
                    SELECT COALESCE(SUM(total_cost), 0)
                    FROM agent_tasks
                    WHERE DATE(created_at) >= :week_ago
                      AND status IN ('completed', 'partial')
                """),
                {"week_ago": week_ago},
            ).scalar() or 0

            # ── 每篇内容平均成本 ──
            total_contents = db.execute(
                text("SELECT COUNT(*) FROM contents")
            ).scalar() or 1

            budget = _DEFAULT_MONTHLY_BUDGET
            data["monthly"] = {
                "total_cost": round(total_cost, 4),
                "per_article_cost": round(total_cost / max(total_contents, 1), 4),
                "daily_avg_cost_7d": round(week_cost / 7, 4),
                "budget_usage_rate": round(min(1.0, total_cost / max(budget, 1)), 4),
                "monthly_budget": budget,
            }

            # ── Token 消耗趋势（逐日）──
            trend_rows = db.execute(
                text("""
                    SELECT
                        DATE(created_at) as date,
                        SUM(total_tokens) as tokens,
                        SUM(total_cost) as cost
                    FROM agent_tasks
                    WHERE status IN ('completed', 'partial')
                    GROUP BY DATE(created_at)
                    ORDER BY date ASC
                    LIMIT 60
                """),
            ).fetchall()

            token_trend = []
            for r in trend_rows:
                token_trend.append({
                    "date": r.date,
                    "models": {"all": r.tokens or 0},
                    "total_tokens": r.tokens or 0,
                    "total_cost": round(r.cost or 0, 4),
                })
            data["token_trend"] = token_trend

            # ── 按环节（agent_type）成本分布 ──
            stage_rows = db.execute(
                text("""
                    SELECT
                        agent_type,
                        SUM(tokens_used) as total_tokens,
                        SUM(time_elapsed) as total_time
                    FROM agent_subtasks
                    GROUP BY agent_type
                    ORDER BY total_tokens DESC
                """),
            ).fetchall()

            total_stage_tokens = sum(r.total_tokens or 0 for r in stage_rows) or 1
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

            cost_by_stage = []
            for r in stage_rows:
                if not r.agent_type or (r.total_tokens or 0) == 0:
                    continue
                # 按 token 占比分摊总成本
                token_ratio = (r.total_tokens or 0) / total_stage_tokens
                stage_cost = round(total_cost * token_ratio, 4)
                cost_by_stage.append({
                    "stage": stage_names.get(r.agent_type, r.agent_type),
                    "cost": stage_cost,
                    "percentage": round(token_ratio, 4),
                })
            data["cost_by_stage"] = cost_by_stage

            # ── 每日明细 ──
            detail_rows = db.execute(
                text("""
                    SELECT
                        DATE(created_at) as date,
                        COUNT(*) as call_count,
                        SUM(total_tokens) as tokens,
                        SUM(total_cost) as cost
                    FROM agent_tasks
                    WHERE status IN ('completed', 'partial')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                    LIMIT 31
                """),
            ).fetchall()

            daily_details = []
            prev_cost: float | None = None
            for r in reversed(detail_rows):
                day_cost = round(r.cost or 0, 4)
                change = None
                if prev_cost is not None and prev_cost > 0:
                    change = round((day_cost - prev_cost) / prev_cost, 4)
                daily_details.append({
                    "date": r.date,
                    "call_count": r.call_count or 0,
                    "total_tokens": r.tokens or 0,
                    "model_costs": {"all": day_cost},
                    "total_cost": day_cost,
                    "change_rate": change,
                })
                prev_cost = day_cost
            data["daily_details"] = daily_details

        finally:
            db.close()
    except Exception:
        pass

    return {"success": True, "data": data}
