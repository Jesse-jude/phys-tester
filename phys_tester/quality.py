"""Grader bias, volatility, drift, and agreement diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .agreement import NORMALIZED_BENCHMARKS, PairAgreement, pairwise_agreement
from .schema import GradedItem


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class GraderDiagnosticsConfig:
    min_scores: int = 8
    min_panel_graders: int = 3
    bias_z: float = 2.5
    bias_min_effect_fraction: float = 0.05
    bias_min_partners: int = 2
    outlier_z: float = 3.0
    outlier_rate: float = 0.15
    drift_rho: float = 0.40
    drift_p: float = 0.05
    drift_min_scores: int = 12
    drift_min_delta_fraction: float = 0.10
    low_qwk: float = 0.40
    min_pair_common: int = 3


@dataclass
class GraderDiagnostic:
    benchmark: str
    grader: str
    n_scores: int
    n_items_in_task: int
    coverage: float | None
    mean_score: float | None
    score_sd: float | None
    mean_residual: float | None
    median_residual: float | None
    mae_vs_consensus: float | None
    rmse_vs_consensus: float | None
    residual_sd: float | None
    robust_z_bias: float | None
    outlier_rate: float | None
    pair_qwk: float | None
    pair_count: int
    drift_rho: float | None
    drift_p: float | None
    drift_slope: float | None
    first_half_mean_residual: float | None
    second_half_mean_residual: float | None
    half_delta: float | None
    status: str
    flags: list[str] = field(default_factory=list)
    bias_effect_fraction: float | None = None
    n_bias_partners: int = 0
    consistent_bias: bool | None = None
    component_size: int = 0
    bias_estimable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraderReport:
    graders: list[GraderDiagnostic]
    task_pooled: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graders": [grader.to_dict() for grader in self.graders],
            "task_pooled": self.task_pooled,
            "summary": self.summary,
        }


def _group(items: Sequence[GradedItem]) -> dict[str, list[GradedItem]]:
    groups: dict[str, list[GradedItem]] = {}
    for item in items:
        groups.setdefault(item.benchmark, []).append(item)
    return groups


def _weighted_pair_qwk(
    pairs: Sequence[PairAgreement],
    grader: str,
) -> tuple[float | None, int]:
    weighted = 0.0
    weight = 0
    count = 0
    for pair in pairs:
        if grader not in (pair.grader_a, pair.grader_b):
            continue
        if pair.qwk is None:
            continue
        weighted += pair.qwk * pair.n_common
        weight += pair.n_common
        count += 1
    return (_finite(weighted / weight) if weight else None, count)


def _components(
    graders: Sequence[str],
    pairs: Sequence[PairAgreement],
    min_common: int,
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {grader: set() for grader in graders}
    for pair in pairs:
        if pair.n_common < min_common or pair.mean_difference is None:
            continue
        adjacency[pair.grader_a].add(pair.grader_b)
        adjacency[pair.grader_b].add(pair.grader_a)
    components: dict[str, set[str]] = {}
    for start in graders:
        if start in components:
            continue
        seen = {start}
        stack = [start]
        while stack:
            grader = stack.pop()
            for neighbor in adjacency.get(grader, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        for grader in seen:
            components[grader] = seen
    return components


def _partner_effects(
    grader: str,
    pairs: Sequence[PairAgreement],
    min_common: int,
) -> list[float]:
    effects = []
    for pair in pairs:
        if pair.n_common < min_common or pair.mean_difference is None:
            continue
        if pair.grader_a == grader:
            effects.append(pair.mean_difference)
        elif pair.grader_b == grader:
            effects.append(-pair.mean_difference)
    return effects


def _drift(
    observations: Sequence[tuple[int, float]],
    config: GraderDiagnosticsConfig,
) -> dict[str, float | None]:
    observations = sorted(observations, key=lambda entry: entry[0])
    if len(observations) < config.drift_min_scores:
        return {
            "rho": None,
            "p": None,
            "slope": None,
            "first_half": None,
            "second_half": None,
            "half_delta": None,
        }
    x = np.array([entry[0] for entry in observations], dtype=float)
    y = np.array([entry[1] for entry in observations], dtype=float)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "rho": None,
            "p": None,
            "slope": None,
            "first_half": None,
            "second_half": None,
            "half_delta": None,
        }
    from scipy import stats

    result = stats.spearmanr(x, y)
    slope = float(np.polyfit(x, y, 1)[0])
    midpoint = len(observations) // 2
    first = float(np.mean(y[:midpoint])) if midpoint else None
    second = float(np.mean(y[midpoint:])) if midpoint < len(y) else None
    return {
        "rho": _finite(result.statistic),
        "p": _finite(result.pvalue),
        "slope": _finite(slope),
        "first_half": _finite(first),
        "second_half": _finite(second),
        "half_delta": _finite(second - first) if first is not None and second is not None else None,
    }


def _diagnose_task(
    benchmark: str,
    items: Sequence[GradedItem],
    pairs: Sequence[PairAgreement],
    config: GraderDiagnosticsConfig,
) -> tuple[list[GraderDiagnostic], dict[str, Any]]:
    normalized = benchmark in NORMALIZED_BENCHMARKS
    graders = sorted({grader for item in items for grader in item.scores_by_grader})
    observations: dict[str, list[tuple[int, float]]] = {grader: [] for grader in graders}
    per_grader_residuals: dict[str, list[float]] = {grader: [] for grader in graders}
    pooled_residuals: list[float] = []

    for item in items:
        scores = (
            item.normalized_scores_by_grader
            if normalized
            else item.scores_by_grader
        )
        if len(scores) < 2:
            continue
        total = sum(scores.values())
        count = len(scores)
        for grader, score in scores.items():
            leave_one_out = (total - score) / (count - 1)
            residual = score - leave_one_out
            observations[grader].append((item.order, residual))
            per_grader_residuals[grader].append(residual)
            pooled_residuals.append(residual)

    pooled = np.array(pooled_residuals, dtype=float) if pooled_residuals else np.array([])
    if pooled.size:
        center = float(np.median(pooled))
        mad = float(np.median(np.abs(pooled - center)))
        pooled_scale = 1.4826 * mad
        if pooled_scale <= 1e-9:
            pooled_scale = float(np.std(pooled))
        if pooled_scale <= 1e-9:
            pooled_scale = 1.0
    else:
        center = 0.0
        mad = 0.0
        pooled_scale = 1.0

    task_scale = 1.0 if normalized else max((item.scale_max for item in items), default=1.0)
    task_pairs = [pair for pair in pairs if pair.benchmark == benchmark]
    components = _components(graders, task_pairs, config.min_pair_common)
    partials: dict[str, dict[str, Any]] = {}
    for grader in graders:
        values = [
            score
            for item in items
            for name, score in item.scores_by_grader.items()
            if name == grader
        ]
        residuals = per_grader_residuals[grader]
        pair_qwk, pair_count = _weighted_pair_qwk(task_pairs, grader)
        partials[grader] = {
            "values": values,
            "n_scores": len(values),
            "coverage": len(values) / len(items) if items else None,
            "residuals": residuals,
            "mean_residual": _finite(np.mean(residuals)) if residuals else None,
            "median_residual": _finite(np.median(residuals)) if residuals else None,
            "mae": _finite(np.mean(np.abs(residuals))) if residuals else None,
            "rmse": _finite(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
            "residual_sd": _finite(np.std(residuals, ddof=1)) if len(residuals) > 1 else None,
            "outlier_rate": (
                _finite(np.mean(np.abs(residuals) > config.outlier_z * pooled_scale))
                if residuals
                else None
            ),
            "pair_qwk": pair_qwk,
            "pair_count": pair_count,
            "drift": _drift(observations[grader], config),
        }

    eligible_means = [
        partial["mean_residual"]
        for partial in partials.values()
        if partial["n_scores"] >= config.min_scores
        and partial["mean_residual"] is not None
    ]
    panel_center = (
        _finite(np.median(eligible_means))
        if len(eligible_means) >= config.min_panel_graders
        else None
    )

    diagnostics: list[GraderDiagnostic] = []
    for grader in graders:
        partial = partials[grader]
        values = partial["values"]
        residuals = partial["residuals"]
        n_scores = partial["n_scores"]
        drift = partial["drift"]
        flags: list[str] = []
        status = "ok"

        robust_z = None
        if (
            panel_center is not None
            and partial["mean_residual"] is not None
            and n_scores >= config.min_scores
        ):
            standard_error = max(
                1e-9, pooled_scale / math.sqrt(n_scores)
            )
            robust_z = _finite(
                (partial["mean_residual"] - panel_center) / standard_error
            )

        partner_effects = _partner_effects(
            grader, task_pairs, config.min_pair_common
        )
        component_size = len(components.get(grader, {grader}))
        consistent_bias = None
        if len(partner_effects) >= config.bias_min_partners:
            consistent_bias = all(effect > 0 for effect in partner_effects) or all(
                effect < 0 for effect in partner_effects
            )
        bias_effect_fraction = (
            abs(partial["mean_residual"]) / task_scale
            if partial["mean_residual"] is not None and task_scale > 0
            else None
        )
        bias_estimable = bool(
            component_size >= config.min_panel_graders
            and len(partner_effects) >= config.bias_min_partners
            and consistent_bias is True
            and bias_effect_fraction is not None
            and bias_effect_fraction >= config.bias_min_effect_fraction
        )

        if n_scores < config.min_scores:
            status = "insufficient_data"
        else:
            if bias_estimable and robust_z is not None and robust_z >= config.bias_z:
                flags.append("lenient")
            elif bias_estimable and robust_z is not None and robust_z <= -config.bias_z:
                flags.append("severe")
            if partial["outlier_rate"] is not None and partial["outlier_rate"] >= config.outlier_rate:
                flags.append("volatile")
            effect_fraction = (
                abs(drift["half_delta"]) / task_scale
                if drift["half_delta"] is not None and task_scale > 0
                else 0.0
            )
            if (
                drift["rho"] is not None
                and drift["p"] is not None
                and abs(drift["rho"]) >= config.drift_rho
                and drift["p"] <= config.drift_p
                and effect_fraction >= config.drift_min_delta_fraction
            ):
                flags.append("drifting")
            if (
                partial["pair_qwk"] is not None
                and partial["pair_qwk"] < config.low_qwk
                and partial["pair_count"] >= 1
            ):
                flags.append("low_agreement")
            if flags:
                status = "flag"

        diagnostics.append(
            GraderDiagnostic(
                benchmark=benchmark,
                grader=grader,
                n_scores=n_scores,
                n_items_in_task=len(items),
                coverage=_finite(partial["coverage"]),
                mean_score=_finite(np.mean(values)) if values else None,
                score_sd=_finite(np.std(values, ddof=1)) if len(values) > 1 else None,
                mean_residual=partial["mean_residual"],
                median_residual=partial["median_residual"],
                mae_vs_consensus=partial["mae"],
                rmse_vs_consensus=partial["rmse"],
                residual_sd=partial["residual_sd"],
                robust_z_bias=robust_z,
                outlier_rate=partial["outlier_rate"],
                pair_qwk=partial["pair_qwk"],
                pair_count=partial["pair_count"],
                drift_rho=drift["rho"],
                drift_p=drift["p"],
                drift_slope=drift["slope"],
                first_half_mean_residual=drift["first_half"],
                second_half_mean_residual=drift["second_half"],
                half_delta=drift["half_delta"],
                status=status,
                flags=flags,
                bias_effect_fraction=bias_effect_fraction,
                n_bias_partners=len(partner_effects),
                consistent_bias=consistent_bias,
                component_size=component_size,
                bias_estimable=bias_estimable,
            )
        )

    pooled_summary = {
        "benchmark": benchmark,
        "n_items": len(items),
        "n_graders": len(graders),
        "n_residuals": len(pooled_residuals),
        "residual_center": _finite(center),
        "residual_mad": _finite(mad),
        "robust_residual_scale": _finite(pooled_scale),
        "panel_center": panel_center,
        "n_eligible_graders": len(eligible_means),
        "task_scale": _finite(task_scale),
        "normalized": normalized,
    }
    return diagnostics, pooled_summary


def grader_report(
    items: Sequence[GradedItem],
    *,
    config: GraderDiagnosticsConfig | None = None,
    pairs: Sequence[PairAgreement] | None = None,
) -> GraderReport:
    config = config or GraderDiagnosticsConfig()
    groups = _group(items)
    pairs = pairs if pairs is not None else pairwise_agreement(items, min_common=config.min_pair_common)
    graders: list[GraderDiagnostic] = []
    task_pooled: list[dict[str, Any]] = []
    for benchmark, group in sorted(groups.items()):
        task_diagnostics, pooled = _diagnose_task(benchmark, group, pairs, config)
        graders.extend(task_diagnostics)
        task_pooled.append(pooled)

    flag_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for grader in graders:
        status_counts[grader.status] = status_counts.get(grader.status, 0) + 1
        for flag in grader.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return GraderReport(
        graders=graders,
        task_pooled=task_pooled,
        summary={
            "n_grader_task_rows": len(graders),
            "n_unique_graders": len({grader.grader for grader in graders}),
            "status_counts": status_counts,
            "flag_counts": flag_counts,
            "flagged": [
                {
                    "benchmark": grader.benchmark,
                    "grader": grader.grader,
                    "flags": grader.flags,
                    "status": grader.status,
                }
                for grader in graders
                if grader.flags
            ],
        },
    )
