"""Export audit and routing results for the Grader Quality Console UI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .schema import GradedItem


_DISPLAY_NAMES = {
    "cpc_bond": "CPC Bond",
    "cpc_management": "CPC Management",
    "diagnostic_reasoning": "Diagnostic Reasoning",
    "management_reasoning": "Management Reasoning",
    "r_idea": "R-IDEAs",
}

_FLAG_LABELS = {
    "high_spread": "wide spread",
    "high_iqr": "high IQR",
    "insufficient_graders": "too few graders",
    "bimodal": "bimodal scores",
    "no_scores": "no scores",
}

_PRIORITY_RANK = {"Urgent": 0, "Review": 1, "Routine": 2}


def _display_name(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _round(value: Any, digits: int = 4) -> float | None:
    number = _finite(value)
    return round(number, digits) if number is not None else None


def _task_scale(items: Sequence[GradedItem], normalized: bool) -> str:
    if normalized:
        return "0–1 normalized"
    scales = sorted({round(item.scale_max, 3) for item in items if item.scale_max})
    if not scales:
        return "—"
    return f"0–{scales[-1]:g}"


def _case_priority(entry: Mapping[str, Any]) -> str:
    flags = set(entry.get("flags") or [])
    spread = _finite(entry.get("spread_fraction")) or 0.0
    if "bimodal" in flags or spread >= 0.5:
        return "Urgent"
    if spread >= 0.25 or len(flags) >= 2:
        return "Review"
    return "Routine"


def _excerpt(text: str, limit: int) -> str:
    text = (text or "").replace("_x000D_", " ").replace("\r", " ")
    text = " ".join(text.split()).lstrip("#").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _grader_signal(row: Mapping[str, Any]) -> str:
    flags = set(row.get("flags") or [])
    if row.get("status") == "insufficient_data":
        return "Insufficient"
    if "drifting" in flags or "low_agreement" in flags:
        return "Review"
    if flags:
        return "Watch"
    return "Stable"


def _build_tasks(
    audit: Mapping[str, Any],
    by_benchmark: Mapping[str, Sequence[GradedItem]],
) -> list[dict[str, Any]]:
    agreement = audit.get("agreement", {}).get("tasks", {})
    disagreement = audit.get("disagreement", {}).get("summary", {})
    tasks = []
    for name in sorted(agreement):
        task = agreement[name]
        review = disagreement.get(name, {})
        items = by_benchmark.get(name, [])
        tasks.append(
            {
                "name": name,
                "label": _display_name(name),
                "items": task.get("n_items"),
                "scoredItems": task.get("n_scored_items"),
                "scores": task.get("n_scores"),
                "graders": task.get("n_graders"),
                "qwk": _round(task.get("mean_pairwise_qwk")),
                "icc": _round(task.get("mean_pairwise_icc")),
                "alpha": _round(task.get("krippendorff_alpha_interval")),
                "exact": _round(task.get("pairwise_exact")),
                "spread": _round(task.get("mean_spread_fraction")),
                "needsHuman": review.get("n_needs_human"),
                "reviewRate": _round(review.get("needs_human_rate")),
                "flagCounts": review.get("flag_counts", {}),
                "scoreScale": _task_scale(items, bool(task.get("normalized"))),
                "normalized": bool(task.get("normalized")),
            }
        )
    return tasks


def _build_graders(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in audit.get("graders", {}).get("graders", []):
        pair_qwk = _finite(row.get("pair_qwk"))
        agreement_rate = None if pair_qwk is None else round(pair_qwk * 100, 1)
        bias_z = _finite(row.get("robust_z_bias")) if row.get("bias_estimable") else None
        rows.append(
            {
                "id": f"{row.get('benchmark')}:{row.get('grader')}",
                "grader": row.get("grader"),
                "task": row.get("benchmark"),
                "label": _display_name(str(row.get("benchmark"))),
                "initials": row.get("grader"),
                "specialty": _display_name(str(row.get("benchmark"))),
                "scores": row.get("n_scores"),
                "coverage": _round(row.get("coverage")),
                "agreementRate": agreement_rate,
                "pairQwk": _round(pair_qwk),
                "pairCount": row.get("pair_count"),
                "biasZ": _round(bias_z, 2),
                "biasEstimable": bool(row.get("bias_estimable")),
                "biasEffect": _round(row.get("bias_effect_fraction")),
                "componentSize": row.get("component_size"),
                "outlierRate": _round(row.get("outlier_rate")),
                "driftRho": _round(row.get("drift_rho")),
                "flags": list(row.get("flags") or []),
                "status": row.get("status"),
                "signal": _grader_signal(row),
            }
        )
    return sorted(rows, key=lambda row: (str(row["task"]), str(row["grader"])))


def _build_cases(
    audit: Mapping[str, Any],
    items_by_id: Mapping[str, GradedItem],
    *,
    max_cases: int,
    max_response_chars: int,
) -> list[dict[str, Any]]:
    cases = []
    for entry in audit.get("disagreement", {}).get("items", []):
        if not entry.get("needs_human"):
            continue
        item = items_by_id.get(str(entry.get("item_id")))
        scores = []
        if item is not None:
            scores = [
                {"name": grader, "score": score}
                for grader, score in sorted(item.scores_by_grader.items())
            ]
        flags = list(entry.get("flags") or [])
        cases.append(
            {
                "id": str(entry.get("item_id")),
                "task": entry.get("benchmark"),
                "taskLabel": _display_name(str(entry.get("benchmark"))),
                "priority": _case_priority(entry),
                "disagreementLevel": round((_finite(entry.get("spread_fraction")) or 0.0) * 100),
                "spreadFraction": _round(entry.get("spread_fraction")),
                "flags": flags,
                "flagLabels": [_FLAG_LABELS.get(flag, flag) for flag in flags],
                "nScores": entry.get("n_scores"),
                "scoreScale": item.scale_max if item is not None else entry.get("scale_max"),
                "scores": scores,
                "response": _excerpt(item.response if item is not None else "", max_response_chars),
                "reference": _excerpt(item.final_diagnosis if item is not None else "", 240),
                "study": item.study if item is not None else "",
                "model": item.model if item is not None else "",
                "status": "Needs review",
                "assignedTo": "Unassigned",
                "updatedAt": "latest audit",
                "llmConfidence": None,
            }
        )
    cases.sort(
        key=lambda case: (
            _PRIORITY_RANK.get(case["priority"], 9),
            -(case["spreadFraction"] or 0.0),
            -len(case["flags"]),
        )
    )
    return cases[:max_cases] if max_cases > 0 else cases


def _build_pre_screen(routing: Mapping[str, Any] | None) -> dict[str, Any]:
    if not routing or not routing.get("summary"):
        return {
            "available": False,
            "model": None,
            "threshold": None,
            "autoRate": None,
            "autoPrecision": None,
            "hardCaseRecall": None,
            "reasonCounts": {},
            "byTask": {},
            "decisions": [],
        }
    summary = routing.get("summary", {})
    selection = routing.get("calibration", {}).get("selection", {})
    config = routing.get("config", {})
    decisions = []
    for decision in routing.get("decisions", []):
        if decision.get("route") != "auto_accept" and not decision.get("reasons"):
            continue
        decisions.append(
            {
                "id": str(decision.get("item_id")),
                "task": decision.get("benchmark"),
                "taskLabel": _display_name(str(decision.get("benchmark"))),
                "route": decision.get("route"),
                "reasons": list(decision.get("reasons") or []),
                "judgeScore": _round(decision.get("judge_score")),
                "judgeConfidence": _round(decision.get("judge_confidence")),
                "physicianSpread": _round(decision.get("physician_spread_fraction")),
                "withinTolerance": decision.get("within_tolerance"),
                "correct": decision.get("correct"),
            }
        )
    return {
        "available": True,
        "model": config.get("judge_model"),
        "threshold": _round(selection.get("threshold")),
        "autoRate": _round(summary.get("auto_rate")),
        "autoPrecision": _round(summary.get("auto_precision")),
        "hardCaseRecall": _round(summary.get("hard_case_recall")),
        "reasonCounts": summary.get("reason_counts", {}),
        "byTask": summary.get("by_task", {}),
        "decisions": decisions,
    }


def build_console_payload(
    audit: Mapping[str, Any],
    items: Sequence[GradedItem] | None = None,
    *,
    routing: Mapping[str, Any] | None = None,
    dataset_name: str = "GRAND-ROUNDS",
    max_cases: int = 100,
    max_response_chars: int = 600,
) -> dict[str, Any]:
    items = list(items or [])
    items_by_id = {item.item_id: item for item in items}
    by_benchmark: dict[str, list[GradedItem]] = {}
    for item in items:
        by_benchmark.setdefault(item.benchmark, []).append(item)

    dataset = audit.get("dataset", {})
    overall = audit.get("agreement", {}).get("overall", {})
    grader_summary = audit.get("graders", {}).get("summary", {})
    status_counts = grader_summary.get("status_counts", {})
    total_rows = sum(status_counts.values())
    cases = _build_cases(
        audit,
        items_by_id,
        max_cases=max_cases,
        max_response_chars=max_response_chars,
    )
    n_items = dataset.get("n_items") or 0
    cases_for_review = sum(
        1
        for entry in audit.get("disagreement", {}).get("items", [])
        if entry.get("needs_human")
    )
    pre_screen = _build_pre_screen(routing)

    return {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "items": n_items,
            "scoredItems": dataset.get("n_scored_items"),
            "scores": dataset.get("n_scores"),
            "tasks": dataset.get("n_tasks"),
            "graders": overall.get("n_graders") or grader_summary.get("n_unique_graders"),
            "models": dataset.get("n_models"),
            "studies": dataset.get("n_studies"),
            "meanScoresPerItem": _round(dataset.get("mean_scores_per_item"), 2),
        },
        "overview": {
            "meanQwk": _round(overall.get("mean_task_pairwise_qwk")),
            "minQwk": _round(overall.get("min_task_pairwise_qwk")),
            "casesForReview": cases_for_review,
            "reviewRate": _round(cases_for_review / n_items) if n_items else None,
            "stableRows": status_counts.get("ok", 0),
            "flaggedRows": status_counts.get("flag", 0),
            "insufficientRows": status_counts.get("insufficient_data", 0),
            "totalRows": total_rows,
            "uniqueGraders": grader_summary.get("n_unique_graders"),
            "autoCoverage": pre_screen["autoRate"],
            "autoPrecision": pre_screen["autoPrecision"],
            "judgeModel": pre_screen["model"],
        },
        "tasks": _build_tasks(audit, by_benchmark),
        "graders": _build_graders(audit),
        "cases": cases,
        "preScreen": pre_screen,
        "reviewConfig": audit.get("config", {}).get("disagreement", {}),
    }
