"""Item-level disagreement and hard-case detection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .schema import GradedItem


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class DisagreementConfig:
    min_graders: int = 2
    spread_fraction: float = 0.20
    iqr_fraction: float = 0.15
    bimodal_gap_fraction: float = 0.25
    bimodal_min_group: int = 2


@dataclass
class ItemDisagreement:
    item_id: str
    benchmark: str
    n_scores: int
    mean_score: float | None
    median_score: float | None
    stdev: float | None
    spread: float | None
    spread_fraction: float | None
    iqr: float | None
    iqr_fraction: float | None
    scale_max: float
    flags: list[str] = field(default_factory=list)
    needs_human: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bimodal(values: Sequence[float], config: DisagreementConfig, scale_max: float) -> bool:
    if len(values) < 2 * config.bimodal_min_group:
        return False
    ordered = sorted(values)
    best_gap = 0.0
    best_index = None
    for index in range(len(ordered) - 1):
        gap = ordered[index + 1] - ordered[index]
        if gap > best_gap:
            best_gap = gap
            best_index = index
    if best_index is None:
        return False
    lower = best_index + 1
    upper = len(ordered) - lower
    if lower < config.bimodal_min_group or upper < config.bimodal_min_group:
        return False
    return scale_max > 0 and best_gap / scale_max >= config.bimodal_gap_fraction


def item_disagreement(
    item: GradedItem,
    config: DisagreementConfig | None = None,
) -> ItemDisagreement:
    config = config or DisagreementConfig()
    values = item.scores
    scale_max = item.scale_max
    flags: list[str] = []
    if not values:
        flags.append("no_scores")
    elif len(values) < config.min_graders:
        flags.append("insufficient_graders")

    if values:
        array = np.array(values, dtype=float)
        spread = float(array.max() - array.min())
        spread_fraction = spread / scale_max if scale_max > 0 else None
        q25, q75 = np.percentile(array, [25, 75])
        iqr = float(q75 - q25)
        iqr_fraction = iqr / scale_max if scale_max > 0 else None
        mean_score = _finite(array.mean())
        median_score = _finite(np.median(array))
        stdev = _finite(array.std(ddof=1)) if len(array) > 1 else None
    else:
        spread = None
        spread_fraction = None
        iqr = None
        iqr_fraction = None
        mean_score = None
        median_score = None
        stdev = None

    if spread_fraction is not None and spread_fraction >= config.spread_fraction:
        flags.append("high_spread")
    if iqr_fraction is not None and iqr_fraction >= config.iqr_fraction:
        flags.append("high_iqr")
    if values and _bimodal(values, config, scale_max):
        flags.append("bimodal")

    hard_flags = {"high_spread", "high_iqr", "bimodal", "insufficient_graders", "no_scores"}
    return ItemDisagreement(
        item_id=item.item_id,
        benchmark=item.benchmark,
        n_scores=len(values),
        mean_score=mean_score,
        median_score=median_score,
        stdev=stdev,
        spread=spread,
        spread_fraction=spread_fraction,
        iqr=iqr,
        iqr_fraction=iqr_fraction,
        scale_max=scale_max,
        flags=flags,
        needs_human=any(flag in hard_flags for flag in flags),
    )


def disagreement_report(
    items: Sequence[GradedItem],
    *,
    config: DisagreementConfig | None = None,
) -> dict[str, Any]:
    config = config or DisagreementConfig()
    records = [item_disagreement(item, config) for item in items]
    tasks: dict[str, dict[str, Any]] = {}
    for record in records:
        summary = tasks.setdefault(
            record.benchmark,
            {
                "n_items": 0,
                "n_with_scores": 0,
                "n_needs_human": 0,
                "flag_counts": {},
                "spread_fractions": [],
            },
        )
        summary["n_items"] += 1
        if record.n_scores:
            summary["n_with_scores"] += 1
            if record.spread_fraction is not None:
                summary["spread_fractions"].append(record.spread_fraction)
        if record.needs_human:
            summary["n_needs_human"] += 1
        for flag in record.flags:
            summary["flag_counts"][flag] = summary["flag_counts"].get(flag, 0) + 1

    for summary in tasks.values():
        fractions = summary.pop("spread_fractions")
        summary["mean_spread_fraction"] = _finite(np.mean(fractions)) if fractions else None
        summary["needs_human_rate"] = (
            _finite(summary["n_needs_human"] / summary["n_items"]) if summary["n_items"] else None
        )

    return {
        "config": asdict(config),
        "summary": tasks,
        "items": [record.to_dict() for record in records],
    }
