"""Plan 冻结 + retry→patch→replan 升级协议 (Layer 4).

Graph Harness 三承诺:
- (1) plan version immutable: 一旦 plan_tasks 完成, 后续不允许改 task 拓扑
      (只能改 task 的 status / output)
- (2) retry→patch→replan: handler fail 先 retry, retry 用尽再 patch,
      patch 失败才允许 replan(重新走 plan_tasks)
- (3) 每个层级有上限: max_retries, max_patches, max_replans

实现方式:
- 把 plan + version 存进 state["plan_version"]
- 任何 handler 都不允许改 plan["tasks"]
- upgrade_to_replan() 是唯一能改 plan 拓扑的入口, 但触发前必须把 retries/patches 用尽
- escalate_failure() 决定下一级升级: retry → patch → replan → halt

Reference:
- Graph Harness (arxiv 2604.11378): "From Agent Loops to Structured Graphs"
- LangGraph: RetryPolicy + TimeoutPolicy + error_handler 模式
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum

from .contracts import PlannerOutput, DEFAULT_RELIABILITY_CONFIG
from ..logging_shared import get_logger

logger = get_logger("agent.frozen_plan")


class UpgradeLevel(str, Enum):
    """失败升级协议的三级。

    RETRY: 同一 task 用不同 prompt / seed 重跑
    PATCH: 替换 task 的 agent_type 为同类的备选 agent(拓扑不变)
    REPLAN: 重新调用 planner, 整个 plan 替换
    """

    RETRY = "retry"
    PATCH = "patch"
    REPLAN = "replan"


@dataclass
class FrozenPlan:
    """冻结的执行计划。

    用 version_hash 标记 plan 拓扑; 如果执行中 plan 被改了(违规),
    state guard 可以检测到 hash mismatch 并触发 halt。
    """

    plan: PlannerOutput
    version_hash: str
    created_at: float
    max_retries: int = DEFAULT_RELIABILITY_CONFIG["max_retries"]
    max_patches: int = DEFAULT_RELIABILITY_CONFIG["max_patches"]
    max_replans: int = DEFAULT_RELIABILITY_CONFIG["max_replans"]
    retry_count: int = 0
    patch_count: int = 0
    replan_count: int = 0
    failure_history: list[dict] = field(default_factory=list)

    @classmethod
    def freeze(
        cls,
        plan: PlannerOutput,
        *,
        max_retries: int | None = None,
        max_patches: int | None = None,
        max_replans: int | None = None,
    ) -> "FrozenPlan":
        """从 PlannerOutput 创建一个冻结的 plan。"""
        # 计算 plan 拓扑的稳定 hash (基于 task id 和 dependencies, 不含 input)
        topo = [
            {"id": t.id, "agent_type": t.agent_type, "deps": sorted(t.dependencies)}
            for t in plan.tasks
        ]
        plan_json = json.dumps(topo, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()[:16]

        # FIX: 用 `is None` 判断, 不要用 `or`, 否则 0 会被当成 None
        return cls(
            plan=plan,
            version_hash=h,
            created_at=time.time(),
            max_retries=(
                max_retries
                if max_retries is not None
                else DEFAULT_RELIABILITY_CONFIG["max_retries"]
            ),
            max_patches=(
                max_patches
                if max_patches is not None
                else DEFAULT_RELIABILITY_CONFIG["max_patches"]
            ),
            max_replans=(
                max_replans
                if max_replans is not None
                else DEFAULT_RELIABILITY_CONFIG["max_replans"]
            ),
        )

    def can_upgrade(self, level: UpgradeLevel) -> bool:
        """检查该级升级是否还有预算。"""
        if level == UpgradeLevel.RETRY:
            return self.retry_count < self.max_retries
        if level == UpgradeLevel.PATCH:
            return self.patch_count < self.max_patches
        if level == UpgradeLevel.REPLAN:
            return self.replan_count < self.max_replans
        return False

    def consume(self, level: UpgradeLevel) -> None:
        """消费一次升级预算。"""
        if level == UpgradeLevel.RETRY:
            self.retry_count += 1
        elif level == UpgradeLevel.PATCH:
            self.patch_count += 1
        elif level == UpgradeLevel.REPLAN:
            self.replan_count += 1

    def escalate_failure(self, task_id: str, reason: str) -> UpgradeLevel | None:
        """决定下一级升级: retry → patch → replan → halt。

        关键属性: 只有上层穷尽才进下一层, 防止"无限 retry" 或 "频繁 replan"
        (Graph Harness 称为 "failure loop pathology")。

        Args:
            task_id: 失败的 task id(记录到 history)
            reason: 失败原因(记录到 history)

        Returns:
            UpgradeLevel 表示可升级到该级, None 表示耗尽, 任务彻底失败。
        """
        self.failure_history.append({
            "task_id": task_id,
            "reason": reason[:200],
            "at": time.time(),
        })

        if self.can_upgrade(UpgradeLevel.RETRY):
            self.consume(UpgradeLevel.RETRY)
            logger.warning(
                "task failure → upgrade to RETRY",
                task_id=task_id,
                retry_count=self.retry_count,
                max=self.max_retries,
                reason=reason[:100],
            )
            return UpgradeLevel.RETRY
        if self.can_upgrade(UpgradeLevel.PATCH):
            self.consume(UpgradeLevel.PATCH)
            logger.warning(
                "task failure → upgrade to PATCH",
                task_id=task_id,
                patch_count=self.patch_count,
                max=self.max_patches,
                reason=reason[:100],
            )
            return UpgradeLevel.PATCH
        if self.can_upgrade(UpgradeLevel.REPLAN):
            self.consume(UpgradeLevel.REPLAN)
            logger.warning(
                "task failure → upgrade to REPLAN",
                task_id=task_id,
                replan_count=self.replan_count,
                max=self.max_replans,
                reason=reason[:100],
            )
            return UpgradeLevel.REPLAN
        logger.error(
            "task failure → HALT (all upgrades exhausted)",
            task_id=task_id,
            retries=self.retry_count,
            patches=self.patch_count,
            replans=self.replan_count,
        )
        return None

    def is_exhausted(self) -> bool:
        """所有升级预算都用尽了。"""
        return (
            self.retry_count >= self.max_retries
            and self.patch_count >= self.max_patches
            and self.replan_count >= self.max_replans
        )

    def to_dict(self) -> dict:
        """序列化(给 state 用)。"""
        return {
            "plan": self.plan.model_dump(),
            "version_hash": self.version_hash,
            "created_at": self.created_at,
            "max_retries": self.max_retries,
            "max_patches": self.max_patches,
            "max_replans": self.max_replans,
            "retry_count": self.retry_count,
            "patch_count": self.patch_count,
            "replan_count": self.replan_count,
            "failure_history": self.failure_history,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "FrozenPlan | None":
        """反序列化(从 state 恢复)。

        FIX #9: 返回类型改为 Optional, 避免调用方 None.version_hash 崩。
        """
        if not data:
            return None
        try:
            plan = PlannerOutput.model_validate(data["plan"])
        except Exception as e:
            logger.warning("frozen_plan.from_dict: plan 反序列化失败", error=str(e)[:200])
            return None
        try:
            return cls(
                plan=plan,
                version_hash=data["version_hash"],
                created_at=data["created_at"],
                max_retries=data.get("max_retries", DEFAULT_RELIABILITY_CONFIG["max_retries"]),
                max_patches=data.get("max_patches", DEFAULT_RELIABILITY_CONFIG["max_patches"]),
                max_replans=data.get("max_replans", DEFAULT_RELIABILITY_CONFIG["max_replans"]),
                retry_count=data.get("retry_count", 0),
                patch_count=data.get("patch_count", 0),
                replan_count=data.get("replan_count", 0),
                failure_history=data.get("failure_history", []),
            )
        except Exception as e:
            logger.warning("frozen_plan.from_dict: 构造失败", error=str(e)[:200])
            return None