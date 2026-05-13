"""FactCheck — ReviewAgent 的事实核查工具

提供基于规则的事实核查能力：
1. 数据/统计断言标注（含有具体数字的陈述）
2. 日期/时间一致性检查
3. 常识性断言标记（提示 LLM 验证）

注意：真正的自动化事实核查需要外部知识库支持，
当前实现提供"标记可能需核查的断言"功能，
配合 ReviewAgent 的 LLM 调用做深度分析。
"""

from __future__ import annotations
import re
from typing import Optional


# 需要来源声明的敏感断言模式
SENSITIVE_CLAIMS = [
    r"(?:根据|据统计|数据显示|研究表明|调查显示|报告指出)[^。]{10,}",
    r"(?:排名|评级|评分)[^。]*?(?:第一|第二|第三|顶级|领先)",
    r"(?:增长|下降|提升|降低)[^。]*?\d+[%％]",
    r"(?:市场(?:规模|份额|占有率)|用户(?:量|数|增长))[^。]*?\d+",
    r"\d+[万亿亿万千]\s*(?:用户|人|美元|元|台|辆)",
    r"(?:最早|首个|首次|独家|唯一)[^。]{5,}",
]


def extract_claims(text: str) -> list[dict]:
    """从文本中提取需要事实核查的断言"""
    if not text:
        return []

    claims = []
    seen: set[str] = set()

    for i, pattern in enumerate(SENSITIVE_CLAIMS):
        for match in re.finditer(pattern, text):
            claim_text = match.group().strip()
            if len(claim_text) < 15 or claim_text in seen:
                continue
            seen.add(claim_text)

            claim_type = "unknown"
            if re.search(r"\d{4}\s*年", claim_text):
                claim_type = "date_or_time"
            elif re.search(r"\d+[%％]", claim_text):
                claim_type = "statistic"
            elif re.search(r"(?:最早|首个|首次|独家|唯一)", claim_text):
                claim_type = "superlative"

            claims.append({
                "claim": claim_text[:200],
                "pattern_index": i,
                "claim_type": claim_type,
                "needs_citation": claim_type in ("statistic", "superlative"),
                "suggestion": _suggest_verification(claim_type, claim_text),
            })

    return claims


def _suggest_verification(claim_type: str, claim_text: str) -> str:
    """根据断言类型提供核实建议"""
    suggestions = {
        "statistic": "建议补充数据来源引用或注明数据时间范围",
        "date_or_time": "建议核实时间/数字的准确性",
        "superlative": "慎用绝对化表述('首个/独家/唯一')，建议补充限定条件或数据来源",
        "unknown": "建议标注信息来源",
    }
    return suggestions.get(claim_type, suggestions["unknown"])


def check_date_consistency(text: str) -> list[dict]:
    """检查文本中的日期一致性"""
    issues = []
    years = re.findall(r"(\d{4})\s*年", text)
    if len(years) >= 2:
        unique_years = set(years)
        if len(unique_years) > 1:
            issues.append({
                "type": "year_inconsistency",
                "years": list(unique_years),
                "severity": "info",
                "suggestion": f"文中出现了多个年份：{', '.join(sorted(unique_years))}，请确认时间线正确",
            })
    return issues


def verify_factual_claims(text: str, known_facts: Optional[dict] = None) -> list[dict]:
    """对照已知事实检查文本中的断言"""
    issues = []
    if not known_facts:
        return issues

    for fact_key, fact_value in known_facts.items():
        if isinstance(fact_value, (int, float)):
            pattern = rf"{fact_key}[^。]*?(\d+(?:\.\d+)?)"
            match = re.search(pattern, text)
            if match:
                stated_value = float(match.group(1))
                if abs(stated_value - fact_value) / max(fact_value, 1) > 0.1:
                    issues.append({
                        "type": "fact_mismatch",
                        "field": fact_key,
                        "stated_value": stated_value,
                        "expected_value": fact_value,
                        "severity": "error",
                        "suggestion": f"'{fact_key}'的数值存在偏差（文中{stated_value}，实际{fact_value}）",
                    })
    return issues


def factcheck_report(text: str, known_facts: Optional[dict] = None) -> dict:
    """生成完整的事实核查报告"""
    claims = extract_claims(text)
    date_issues = check_date_consistency(text)
    fact_issues = verify_factual_claims(text, known_facts)

    severity_count = {"error": 0, "warning": 0, "info": 0}
    for issue in claims + date_issues + fact_issues:
        sev = issue.get("severity", "info") if "severity" in issue else \
              "warning" if issue.get("needs_citation") else "info"
        severity_count[sev] = severity_count.get(sev, 0) + 1

    total_issues = len(claims) + len(date_issues) + len(fact_issues)
    return {
        "total_claims_checked": total_issues,
        "claims_without_citation": sum(1 for c in claims if c.get("needs_citation")),
        "date_issues": len(date_issues),
        "fact_mismatches": len(fact_issues),
        "severity_summary": severity_count,
        "claims": claims[:10],
        "date_issues": date_issues,
        "fact_issues": fact_issues,
        "passed": severity_count.get("error", 0) == 0 and severity_count.get("warning", 0) == 0,
    }
