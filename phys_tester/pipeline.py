"""End-to-end audit and routing workflows."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .agreement import AgreementReport, agreement_report
from .disagreement import DisagreementConfig, disagreement_report
from .judge import Judge, Judgment
from .quality import GraderDiagnosticsConfig, GraderReport, grader_report
from .routing import (
    RoutingConfig,
    calibration_curve,
    route_items,
    select_confidence_threshold,
    summarize_routing,
)
from .schema import GradedItem


def dataset_summary(items: Sequence[GradedItem]) -> dict[str, Any]:
    tasks = Counter(item.benchmark for item in items)
    models = Counter(item.model for item in items if item.model)
    studies = Counter(item.study for item in items if item.study)
    score_counts = [len(item.scores) for item in items]
    return {
        "n_items": len(items),
        "n_scored_items": sum(1 for count in score_counts if count > 0),
        "n_scores": sum(score_counts),
        "n_tasks": len(tasks),
        "tasks": dict(tasks.most_common()),
        "n_models": len(models),
        "top_models": dict(models.most_common(10)),
        "n_studies": len(studies),
        "studies": dict(studies.most_common()),
        "items_without_scores": sum(1 for count in score_counts if count == 0),
        "items_with_one_score": sum(1 for count in score_counts if count == 1),
        "items_with_two_or_more_scores": sum(1 for count in score_counts if count >= 2),
        "mean_scores_per_item": (
            sum(score_counts) / len(score_counts) if score_counts else None
        ),
    }


def run_audit(
    items: Sequence[GradedItem],
    *,
    agreement_min_common: int = 3,
    grader_config: GraderDiagnosticsConfig | None = None,
    disagreement_config: DisagreementConfig | None = None,
) -> dict[str, Any]:
    grader_config = grader_config or GraderDiagnosticsConfig()
    disagreement_config = disagreement_config or DisagreementConfig()
    agreement: AgreementReport = agreement_report(items, min_common=agreement_min_common)
    quality: GraderReport = grader_report(
        items,
        config=grader_config,
        pairs=agreement.pairs,
    )
    disagreement = disagreement_report(items, config=disagreement_config)
    return {
        "dataset": dataset_summary(items),
        "agreement": agreement.to_dict(),
        "graders": quality.to_dict(),
        "disagreement": disagreement,
        "config": {
            "agreement_min_common": agreement_min_common,
            "grader": grader_config.__dict__,
            "disagreement": disagreement_config.__dict__,
        },
    }


def run_routing(
    items: Sequence[GradedItem],
    judge: Judge | Mapping[str, Judgment],
    *,
    config: RoutingConfig | None = None,
    calibrate: bool = True,
) -> dict[str, Any]:
    config = config or RoutingConfig()
    if hasattr(judge, "judge"):
        judgments = {item.item_id: judge.judge(item) for item in items}
    else:
        judgments = dict(judge)

    decisions = route_items(items, judgments, config)
    summary = summarize_routing(decisions, config)

    calibration = None
    if calibrate:
        curve = calibration_curve(items, judgments, config=config)
        selection = select_confidence_threshold(
            curve, target_precision=config.target_precision
        )
        at_recommended = None
        if selection["meets_target"]:
            recommended_config = replace(
                config, confidence_threshold=float(selection["threshold"])
            )
            recommended_decisions = route_items(items, judgments, recommended_config)
            at_recommended = summarize_routing(recommended_decisions, recommended_config)
        calibration = {
            "curve": curve,
            "selection": selection,
            "at_recommended_threshold": at_recommended,
        }

    return {
        "dataset": dataset_summary(items),
        "config": config.__dict__,
        "summary": summary,
        "calibration": calibration,
        "decisions": [decision.to_dict() for decision in decisions],
    }
