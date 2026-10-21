"""Triage policy: auto-accept easy items, send disagreements and hard cases to humans."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .agreement import NORMALIZED_BENCHMARKS, compare_scores
from .judge import Judge, Judgment
from .schema import GradedItem


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class RoutingConfig:
    mode: str = "audit"
    confidence_threshold: float = 0.70
    max_spread_fraction: float = 0.20
    min_graders: int = 2
    judge_tolerance: float | None = None
    judge_tolerance_fraction: float = 0.10
    judge_tolerance_min: float = 0.50
    use_physician_grades_in_triage: bool = False
    human_cost_per_item: float = 25.0
    judge_cost_per_item: float = 0.01
    target_precision: float = 0.90

    def __post_init__(self) -> None:
        if self.mode not in {"audit", "triage"}:
            raise ValueError("mode must be 'audit' or 'triage'")


@dataclass
class RoutingDecision:
    item_id: str
    benchmark: str
    mode: str
    route: str
    reasons: list[str] = field(default_factory=list)
    judge_score: float | None = None
    judge_confidence: float | None = None
    judge_error: str | None = None
    physician_consensus: float | None = None
    reference_score: float | None = None
    physician_spread: float | None = None
    physician_spread_fraction: float | None = None
    scale_max: float | None = None
    tolerance: float | None = None
    absolute_error: float | None = None
    within_tolerance: bool | None = None
    correct: bool | None = None
    needs_human_by_physicians: bool | None = None
    estimated_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _judgments_for(
    items: Sequence[GradedItem],
    judgments: Mapping[str, Judgment] | Judge,
) -> dict[str, Judgment]:
    if hasattr(judgments, "judge"):
        return {item.item_id: judgments.judge(item) for item in items}
    return dict(judgments)


def tolerance_for(item: GradedItem, config: RoutingConfig) -> float:
    if config.judge_tolerance is not None:
        return float(config.judge_tolerance)
    scale = item.scale_max
    step = item.scale_step
    return max(
        1e-9,
        config.judge_tolerance_fraction * scale,
        config.judge_tolerance_min * step,
    )


def route_item(
    item: GradedItem,
    judgment: Judgment,
    config: RoutingConfig | None = None,
) -> RoutingDecision:
    config = config or RoutingConfig()
    reference = item.reference_score
    consensus = item.mean_score
    spread = item.spread
    spread_fraction = item.spread_fraction
    scale_max = item.scale_max
    tolerance = tolerance_for(item, config)
    n_scores = len(item.scores)

    reasons: list[str] = []
    judge_score = _finite(judgment.score)

    if judgment.error or judge_score is None:
        reasons.append("judge_error")
    if not item.response.strip():
        reasons.append("empty_response")
    if judge_score is not None and (judge_score < 0 or judge_score > scale_max + 1e-9):
        reasons.append("judge_off_scale")

    physicians_disagree = n_scores < config.min_graders or (
        spread_fraction is not None and spread_fraction >= config.max_spread_fraction
    )

    audit_mode = config.mode == "audit" or (
        config.mode == "triage" and config.use_physician_grades_in_triage
    )
    if audit_mode:
        if n_scores < config.min_graders:
            reasons.append("insufficient_graders")
        elif spread_fraction is not None and spread_fraction >= config.max_spread_fraction:
            reasons.append("physician_disagreement")

    if judgment.confidence is None or judgment.confidence < config.confidence_threshold:
        reasons.append("low_judge_confidence")

    absolute_error = None
    within_tolerance = None
    correct = None
    if reference is not None and judge_score is not None:
        absolute_error = abs(judge_score - reference)
        within_tolerance = absolute_error <= tolerance
        correct = within_tolerance
        if audit_mode and not within_tolerance:
            reasons.append("judge_outlier")

    route = "human_review" if reasons else "auto_accept"
    cost = config.judge_cost_per_item + (
        config.human_cost_per_item if route == "human_review" else 0.0
    )
    return RoutingDecision(
        item_id=item.item_id,
        benchmark=item.benchmark,
        mode=config.mode,
        route=route,
        reasons=list(dict.fromkeys(reasons)),
        judge_score=judge_score,
        judge_confidence=_finite(judgment.confidence),
        judge_error=judgment.error,
        physician_consensus=consensus,
        reference_score=reference,
        physician_spread=spread,
        physician_spread_fraction=spread_fraction,
        scale_max=_finite(scale_max),
        tolerance=_finite(tolerance),
        absolute_error=_finite(absolute_error),
        within_tolerance=within_tolerance,
        correct=correct,
        needs_human_by_physicians=physicians_disagree if n_scores else None,
        estimated_cost=_finite(cost),
    )


def route_items(
    items: Sequence[GradedItem],
    judgments: Mapping[str, Judgment] | Judge,
    config: RoutingConfig | None = None,
) -> list[RoutingDecision]:
    config = config or RoutingConfig()
    judgment_map = _judgments_for(items, judgments)
    decisions = []
    for item in items:
        judgment = judgment_map.get(item.item_id)
        if judgment is None:
            judgment = Judgment(
                score=None,
                confidence=0.0,
                error="missing_judgment",
            )
        decisions.append(route_item(item, judgment, config))
    return decisions


def _reference_correctness(
    decision: RoutingDecision,
) -> bool | None:
    return decision.correct


def _judge_agreement(decisions: Sequence[RoutingDecision]) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    for benchmark in sorted({decision.benchmark for decision in decisions}):
        group = [
            decision
            for decision in decisions
            if decision.benchmark == benchmark
            and decision.judge_score is not None
            and decision.reference_score is not None
        ]
        normalized = benchmark in NORMALIZED_BENCHMARKS
        if not group:
            tasks[benchmark] = {
                "n": 0,
                "qwk": None,
                "icc_2_1": None,
                "mae": None,
                "rmse": None,
                "spearman": None,
                "exact": None,
                "normalized": normalized,
            }
            continue
        predicted = {}
        reference = {}
        for decision in group:
            scale = decision.scale_max or 1.0
            divisor = scale if normalized else 1.0
            predicted[decision.item_id] = float(decision.judge_score) / divisor
            reference[decision.item_id] = float(decision.reference_score) / divisor
        values = list(predicted.values()) + list(reference.values())
        if normalized:
            min_rating, max_rating, step = 0.0, 1.0, 0.05
        else:
            min_rating = 0.0
            max_rating = max(decision.scale_max or 1.0 for decision in group)
            step = 1.0 if all(abs(value - round(value)) < 1e-9 for value in values) else 0.5
        result = compare_scores(
            predicted,
            reference,
            min_rating=min_rating,
            max_rating=max_rating,
            step=step,
        )
        result["normalized"] = normalized
        tasks[benchmark] = result
    return tasks


def summarize_routing(
    decisions: Sequence[RoutingDecision],
    config: RoutingConfig | None = None,
) -> dict[str, Any]:
    config = config or RoutingConfig()
    n_items = len(decisions)
    auto = [decision for decision in decisions if decision.route == "auto_accept"]
    human = [decision for decision in decisions if decision.route == "human_review"]
    labeled = [decision for decision in decisions if decision.correct is not None]
    auto_labeled = [decision for decision in auto if decision.correct is not None]
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        for reason in decision.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    auto_precision = (
        _finite(np.mean([decision.correct for decision in auto_labeled]))
        if auto_labeled
        else None
    )
    physician_hard = [
        decision for decision in decisions if decision.needs_human_by_physicians
    ]
    hard_recall = (
        _finite(
            np.mean([decision.route == "human_review" for decision in physician_hard])
        )
        if physician_hard
        else None
    )

    manual_cost = n_items * config.human_cost_per_item
    actual_cost = sum(decision.estimated_cost or 0.0 for decision in decisions)
    savings = manual_cost - actual_cost if config.human_cost_per_item > 0 else None

    by_task: dict[str, dict[str, Any]] = {}
    for benchmark in sorted({decision.benchmark for decision in decisions}):
        group = [decision for decision in decisions if decision.benchmark == benchmark]
        group_auto = [decision for decision in group if decision.route == "auto_accept"]
        group_labeled = [decision for decision in group_auto if decision.correct is not None]
        by_task[benchmark] = {
            "n_items": len(group),
            "n_auto": len(group_auto),
            "n_human": len(group) - len(group_auto),
            "auto_rate": _finite(len(group_auto) / len(group)) if group else None,
            "auto_precision": (
                _finite(np.mean([decision.correct for decision in group_labeled]))
                if group_labeled
                else None
            ),
        }

    return {
        "mode": config.mode,
        "n_items": n_items,
        "n_auto": len(auto),
        "n_human": len(human),
        "auto_rate": _finite(len(auto) / n_items) if n_items else None,
        "human_rate": _finite(len(human) / n_items) if n_items else None,
        "auto_precision": auto_precision,
        "n_labeled": len(labeled),
        "n_auto_labeled": len(auto_labeled),
        "physician_hard_cases": len(physician_hard),
        "hard_case_recall": hard_recall,
        "reason_counts": reason_counts,
        "by_task": by_task,
        "judge_agreement": _judge_agreement(decisions),
        "cost": {
            "manual_cost": _finite(manual_cost),
            "actual_cost": _finite(actual_cost),
            "estimated_savings": _finite(savings),
            "savings_fraction": _finite(savings / manual_cost) if savings is not None and manual_cost > 0 else None,
            "human_cost_per_item": config.human_cost_per_item,
            "judge_cost_per_item": config.judge_cost_per_item,
        },
    }


def calibration_curve(
    items: Sequence[GradedItem],
    judgments: Mapping[str, Judgment] | Judge,
    *,
    config: RoutingConfig | None = None,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    config = config or RoutingConfig()
    judgment_map = _judgments_for(items, judgments)

    if thresholds is None:
        values = [
            judgment.confidence
            for judgment in judgment_map.values()
            if judgment.confidence is not None
        ]
        thresholds = sorted(set([0.0, *[max(0.0, min(1.0, value)) for value in values]]))
    else:
        thresholds = sorted(set(thresholds))

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        eligible = []
        for item in items:
            judgment = judgment_map.get(item.item_id)
            if judgment is None or judgment.score is None or judgment.error:
                continue
            confidence = judgment.confidence if judgment.confidence is not None else 0.0
            if confidence < threshold:
                continue
            reference = item.reference_score
            if reference is None:
                continue
            tolerance = tolerance_for(item, config)
            correct = abs(float(judgment.score) - reference) <= tolerance
            eligible.append((item, judgment, correct))
        auto_all = 0
        for item in items:
            judgment = judgment_map.get(item.item_id)
            if judgment is None or judgment.score is None or judgment.error:
                continue
            confidence = judgment.confidence if judgment.confidence is not None else 0.0
            if confidence >= threshold:
                auto_all += 1
        precision = (
            _finite(np.mean([entry[2] for entry in eligible])) if eligible else None
        )
        rows.append(
            {
                "threshold": float(threshold),
                "n_auto": auto_all,
                "n_auto_labeled": len(eligible),
                "coverage": _finite(auto_all / len(items)) if items else None,
                "precision": precision,
            }
        )
    return rows


def select_confidence_threshold(
    curve: Sequence[Mapping[str, Any]],
    *,
    target_precision: float = 0.90,
    min_auto_labeled: int = 1,
) -> dict[str, Any]:
    valid = [
        row
        for row in curve
        if row.get("precision") is not None
        and row["precision"] >= target_precision
        and row.get("n_auto_labeled", 0) >= min_auto_labeled
    ]
    if not valid:
        return {
            "threshold": 1.0,
            "precision": None,
            "coverage": 0.0,
            "n_auto": 0,
            "n_auto_labeled": 0,
            "meets_target": False,
            "target_precision": target_precision,
        }
    best = max(
        valid,
        key=lambda row: (row["coverage"], row["precision"], -row["threshold"]),
    )
    return {
        "threshold": float(best["threshold"]),
        "precision": best["precision"],
        "coverage": best["coverage"],
        "n_auto": best["n_auto"],
        "n_auto_labeled": best["n_auto_labeled"],
        "meets_target": True,
        "target_precision": target_precision,
    }
