"""Agent I/O 校验器 — 按 schema 解析输入 + 校验输出

核心逻辑:
  1. 按 AgentCard.input_schema 的 source/consume 声明装配输入
  2. 校验必填字段完整性
  3. 输出后校验 output_fields 是否满足 output_schema
  4. 不通过 → 报错，不兜底
"""

from __future__ import annotations
from typing import Any


# ── 按 schema 装配输入 ──

def build_agent_input(
    agent_type: str,
    task_input: dict,
    card: "AgentCard",  # noqa: F821
    task_queue: list[dict],
    task_results: dict[str, dict],
) -> dict:
    """根据 AgentCard.input_schema 装配 t_input。

    只注入 schema 声明的字段，不会多一个也不会少一个。
    不存在的字段 → 不在返回值里 → handler 读不到。
    """
    schema = card.input_schema
    if not schema:
        return {}

    props = schema.get("properties", {})
    resolved: dict[str, Any] = {}

    for field_name, rules in props.items():
        source = rules.get("source", "")

        if source == "consume":
            value = _consume_upstream(field_name, task_results)
            if value is not None:
                resolved[field_name] = value

        elif source.startswith("param:"):
            key = source.replace("param:", "")
            if key in task_input:
                resolved[field_name] = task_input[key]

        elif source == "meta":
            if field_name in task_input:
                resolved[field_name] = task_input[field_name]

        # source 为空（无注解字段）→ 直接从 task_input 读取
        elif field_name in task_input:
            resolved[field_name] = task_input[field_name]

    return resolved


def _consume_upstream(
    field_name: str,
    task_results: dict[str, dict],
) -> Any | None:
    """从已完成任务中找最近产出 field_name 的任务。

    策略：逆序遍历 task_results（最新完成的任务优先），
    因为任务按执行顺序插入，最新完成的就是离当前最近的。

    对于线性 pipeline（如 WebSearch → WritingAgent → ReviewAgent → FixAgent），
    这个策略永远返回正确的"最近"产出者。

    对于更复杂的 DAG，可以升级为 BFS 依赖链解析，当前场景暂不需要。
    """
    for tid in reversed(list(task_results.keys())):
        result = task_results[tid]
        output = result.get("output", {})
        output_fields = output.get("output_fields", {}) if isinstance(output, dict) else {}
        if field_name in output_fields:
            return output_fields[field_name]

    return None


# ── 输入校验 ──

def validate_input(agent_type: str, resolved: dict, card: "AgentCard") -> list[str]:
    """校验输入完整性，返回所有问题列表。空列表 = 通过。

    校验规则:
      - required 列表中的字段必须存在且非空
      - type=string 必须是 str 类型
      - type=array 必须是 list 类型
    """
    schema = card.input_schema
    if not schema:
        return []

    errors: list[str] = []
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    for field_name in required:
        if field_name not in resolved or resolved[field_name] is None:
            errors.append(f"[{agent_type}] 缺少必填输入字段 '{field_name}'")
            continue

        value = resolved[field_name]
        rules = props.get(field_name, {})

        # 空字符串/空列表 = 无效
        if isinstance(value, (str, list, dict)) and len(value) == 0:
            errors.append(f"[{agent_type}] 必填字段 '{field_name}' 为空")

        # 类型校验
        expected_type = rules.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"[{agent_type}] 字段 '{field_name}' 应为 string, 实际 {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, (list, tuple)):
            errors.append(f"[{agent_type}] 字段 '{field_name}' 应为 array, 实际 {type(value).__name__}")
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(f"[{agent_type}] 字段 '{field_name}' 应为 integer, 实际 {type(value).__name__}")

    return errors


# ── 输出校验 ──

def validate_output(agent_type: str, result: dict, card: "AgentCard") -> list[str]:
    """校验输出是否满足 output_schema。

    从 result["output"]["output_fields"] 中读取产出物字段。
    """
    schema = card.output_schema
    if not schema:
        return []

    errors: list[str] = []
    output = result.get("output", {})
    output_fields = output.get("output_fields", {}) if isinstance(output, dict) else {}
    required = set(schema.get("required", []))
    props = schema.get("properties", {})

    for field_name in required:
        if field_name not in output_fields or output_fields[field_name] is None:
            errors.append(f"[{agent_type}] 输出缺少必填字段 '{field_name}'")
            continue

        value = output_fields[field_name]
        rules = props.get(field_name, {})

        if isinstance(value, (str, list, dict)) and len(value) == 0:
            errors.append(f"[{agent_type}] 输出字段 '{field_name}' 为空")

        expected_type = rules.get("type")
        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"[{agent_type}] 输出字段 '{field_name}' 应为 string, 实际 {type(value).__name__}")

    return errors
