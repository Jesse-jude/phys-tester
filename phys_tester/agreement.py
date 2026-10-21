"""Inter-rater agreement metrics for sparse physician grading panels."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import GradedItem

NORMALIZED_BENCHMARKS = {"management_reasoning"}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired(rater_a: Sequence[float | None], rater_b: Sequence[float | None]) -> list[tuple[float, float]]:
    pairs = []
    for a, b in zip(rater_a, rater_b):
        if a is None or b is None:
            continue
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((float(a), float(b)))
    return pairs


def _infer_step(values: Iterable[float]) -> float:
    ordered = sorted({float(value) for value in values})
    if len(ordered) < 2:
        return 1.0
    diffs = [round(b - a, 9) for a, b in zip(ordered, ordered[1:]) if b > a]
    if not diffs:
        return 1.0
    if all(abs(diff - round(diff)) < 1e-9 for diff in diffs):
        return 1.0
    return 0.5


def _labels(
    values: Sequence[float],
    *,
    min_rating: float | None,
    max_rating: float | None,
    step: float | None,
) -> np.ndarray:
    if min_rating is not None and max_rating is not None and max_rating > min_rating:
        step = step or _infer_step(values)
        count = int(round((max_rating - min_rating) / step))
        return np.round(np.linspace(min_rating, max_rating, count + 1), 9)
    return np.array(sorted(set(values)), dtype=float)


def quadratic_weighted_kappa(
    rater_a: Sequence[float | None],
    rater_b: Sequence[float | None],
    *,
    min_rating: float | None = None,
    max_rating: float | None = None,
    step: float | None = None,
) -> float | None:
    pairs = _paired(rater_a, rater_b)
    if len(pairs) < 2:
        return None
    a = np.array([pair[0] for pair in pairs])
    b = np.array([pair[1] for pair in pairs])
    labels = _labels(np.concatenate([a, b]), min_rating=min_rating, max_rating=max_rating, step=step)
    if len(labels) < 2:
        return None
    index_a = np.abs(labels[:, None] - a[None, :]).argmin(axis=0)
    index_b = np.abs(labels[:, None] - b[None, :]).argmin(axis=0)
    size = len(labels)
    observed = np.zeros((size, size), dtype=float)
    np.add.at(observed, (index_a, index_b), 1.0)
    total = observed.sum()
    if total <= 0:
        return None
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    weights = (np.arange(size)[:, None] - np.arange(size)[None, :]) ** 2 / (size - 1) ** 2
    denominator = float((weights * expected).sum())
    if denominator <= 0:
        return None
    numerator = float((weights * observed).sum())
    return _finite(1.0 - numerator / denominator)


def icc_2_1(matrix: Sequence[Sequence[float | None]]) -> float | None:
    data = np.array(
        [[np.nan if value is None else float(value) for value in row] for row in matrix],
        dtype=float,
    )
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return None
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    if n < 2 or k < 2:
        return None
    grand = float(data.mean())
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ssr = k * float(((row_means - grand) ** 2).sum())
    ssc = n * float(((col_means - grand) ** 2).sum())
    sst = float(((data - grand) ** 2).sum())
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1)) if n > 1 and k > 1 else 0.0
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    if abs(denominator) < 1e-12:
        return None
    return _finite((msr - mse) / denominator)


def icc_2_k(matrix: Sequence[Sequence[float | None]]) -> float | None:
    data = np.array(
        [[np.nan if value is None else float(value) for value in row] for row in matrix],
        dtype=float,
    )
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        return None
    data = data[~np.isnan(data).any(axis=1)]
    n, k = data.shape
    if n < 2 or k < 2:
        return None
    grand = float(data.mean())
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ssr = k * float(((row_means - grand) ** 2).sum())
    ssc = n * float(((col_means - grand) ** 2).sum())
    sst = float(((data - grand) ** 2).sum())
    sse = sst - ssr - ssc
    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1)) if n > 1 and k > 1 else 0.0
    denominator = msr + (msc - mse) / n
    if abs(denominator) < 1e-12:
        return None
    return _finite((msr - mse) / denominator)


def krippendorff_alpha(
    units: Sequence[Sequence[float | None]],
    *,
    level: str = "interval",
) -> float | None:
    if level not in {"interval", "nominal"}:
        raise ValueError("level must be 'interval' or 'nominal'")
    cleaned: list[list[float]] = []
    for unit in units:
        values = [float(value) for value in unit if value is not None and math.isfinite(float(value))]
        if len(values) >= 2:
            cleaned.append(values)
    if not cleaned:
        return None
    values = sorted({value for unit in cleaned for value in unit})
    if len(values) < 2:
        return None
    index = {value: position for position, value in enumerate(values)}
    size = len(values)
    observed = np.zeros((size, size), dtype=float)
    total = 0
    for unit in cleaned:
        counts = np.zeros(size, dtype=float)
        for value in unit:
            counts[index[value]] += 1
        total += len(unit)
        m = len(unit)
        for i in range(size):
            if counts[i] == 0:
                continue
            observed[i, i] += counts[i] * (counts[i] - 1) / (m - 1)
            for j in range(size):
                if i != j and counts[j] > 0:
                    observed[i, j] += counts[i] * counts[j] / (m - 1)
    if total < 2:
        return None
    marginals = observed.sum(axis=0)
    if level == "interval":
        delta = (np.array(values)[:, None] - np.array(values)[None, :]) ** 2
    else:
        delta = (np.arange(size)[:, None] != np.arange(size)[None, :]).astype(float)
    observed_difference = float((observed * delta).sum())
    expected_difference = float((np.outer(marginals, marginals) * delta).sum())
    if expected_difference <= 0:
        return None
    return _finite(1.0 - (total - 1) * observed_difference / expected_difference)


def _pair_metrics(
    item_pairs: Sequence[tuple[GradedItem, float, float]],
    *,
    min_common: int,
    normalized: bool = False,
) -> dict[str, float | int | None]:
    n_common = len(item_pairs)
    if n_common == 0:
        return {
            "n_common": 0,
            "qwk": None,
            "icc": None,
            "mae": None,
            "exact": None,
            "mean_difference": None,
        }
    values_a = [pair[1] for pair in item_pairs]
    values_b = [pair[2] for pair in item_pairs]
    if normalized:
        maximum = 1.0
        step = 0.05
    else:
        maximum = max(pair[0].scale_max for pair in item_pairs)
        step = 0.5 if any(pair[0].scale_step == 0.5 for pair in item_pairs) else 1.0
    qwk = quadratic_weighted_kappa(
        values_a, values_b, min_rating=0.0, max_rating=maximum, step=step
    ) if n_common >= min_common else None
    icc = icc_2_1(np.column_stack([values_a, values_b])) if n_common >= min_common else None
    differences = [abs(a - b) for a, b in zip(values_a, values_b)]
    signed = [a - b for a, b in zip(values_a, values_b)]
    return {
        "n_common": n_common,
        "qwk": qwk,
        "icc": icc,
        "mae": _finite(np.mean(differences)),
        "exact": _finite(np.mean([difference <= 1e-9 for difference in differences])),
        "mean_difference": _finite(np.mean(signed)),
    }


@dataclass
class PairAgreement:
    benchmark: str
    grader_a: str
    grader_b: str
    n_common: int
    qwk: float | None
    icc: float | None
    mae: float | None
    exact: float | None
    normalized: bool = False
    mean_difference: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskAgreement:
    benchmark: str
    n_items: int
    n_scored_items: int
    n_scores: int
    n_graders: int
    mean_scores_per_item: float | None
    complete_cases: int
    icc_2_1: float | None
    icc_2_k: float | None
    krippendorff_alpha_interval: float | None
    krippendorff_alpha_nominal: float | None
    mean_pairwise_qwk: float | None
    mean_pairwise_icc: float | None
    mean_pairwise_mae: float | None
    pairwise_exact: float | None
    mean_spread_fraction: float | None
    graders: list[str] = field(default_factory=list)
    pair_count: int = 0
    normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgreementReport:
    tasks: dict[str, TaskAgreement]
    pairs: list[PairAgreement]
    overall: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": {name: task.to_dict() for name, task in self.tasks.items()},
            "pairs": [pair.to_dict() for pair in self.pairs],
            "overall": self.overall,
        }


def _group_by_benchmark(items: Sequence[GradedItem]) -> dict[str, list[GradedItem]]:
    groups: dict[str, list[GradedItem]] = {}
    for item in items:
        groups.setdefault(item.benchmark, []).append(item)
    return groups


def pairwise_agreement(
    items: Sequence[GradedItem],
    *,
    min_common: int = 3,
) -> list[PairAgreement]:
    results: list[PairAgreement] = []
    for benchmark, group in _group_by_benchmark(items).items():
        normalized = benchmark in NORMALIZED_BENCHMARKS
        graders = sorted({grader for item in group for grader in item.scores_by_grader})
        for grader_a, grader_b in combinations(graders, 2):
            pairs = []
            for item in group:
                score_map = (
                    item.normalized_scores_by_grader
                    if normalized
                    else item.scores_by_grader
                )
                if grader_a in score_map and grader_b in score_map:
                    pairs.append((item, score_map[grader_a], score_map[grader_b]))
            metrics = _pair_metrics(
                pairs, min_common=min_common, normalized=normalized
            )
            results.append(
                PairAgreement(
                    benchmark=benchmark,
                    grader_a=grader_a,
                    grader_b=grader_b,
                    n_common=int(metrics["n_common"] or 0),
                    qwk=metrics["qwk"],
                    icc=metrics["icc"],
                    mae=metrics["mae"],
                    exact=metrics["exact"],
                    normalized=normalized,
                    mean_difference=metrics["mean_difference"],
                )
            )
    return results


def _task_agreement(
    benchmark: str,
    items: Sequence[GradedItem],
    pairs: Sequence[PairAgreement],
) -> TaskAgreement:
    normalized = benchmark in NORMALIZED_BENCHMARKS
    graders = sorted({grader for item in items for grader in item.scores_by_grader})
    scored_items = [item for item in items if item.scores]
    n_scores = sum(len(item.scores) for item in items)

    matrices = [
        item.normalized_scores_by_grader if normalized else item.scores_by_grader
        for item in scored_items
    ]
    matrix = [[score_map.get(grader) for grader in graders] for score_map in matrices]
    complete_cases = 0
    if graders:
        complete_cases = sum(
            1 for row in matrix if all(value is not None for value in row)
        )

    task_pairs = [pair for pair in pairs if pair.benchmark == benchmark and pair.n_common >= 3]

    def weighted_pair_mean(attribute: str) -> float | None:
        weighted_sum = 0.0
        weight = 0
        for pair in task_pairs:
            value = getattr(pair, attribute)
            if value is not None:
                weighted_sum += value * pair.n_common
                weight += pair.n_common
        return _finite(weighted_sum / weight) if weight else None

    pair_qwk = weighted_pair_mean("qwk")
    pair_icc = weighted_pair_mean("icc")

    item_pair_mae: list[float] = []
    item_pair_exact: list[float] = []
    for item in items:
        values = item.normalized_scores if normalized else item.scores
        for a, b in combinations(values, 2):
            item_pair_mae.append(abs(a - b))
            item_pair_exact.append(1.0 if abs(a - b) <= 1e-9 else 0.0)

    spread_fractions = [
        item.spread_fraction for item in items if item.spread_fraction is not None
    ]

    return TaskAgreement(
        benchmark=benchmark,
        n_items=len(items),
        n_scored_items=len(scored_items),
        n_scores=n_scores,
        n_graders=len(graders),
        mean_scores_per_item=_finite(np.mean([len(item.scores) for item in scored_items])) if scored_items else None,
        complete_cases=complete_cases,
        icc_2_1=icc_2_1(matrix),
        icc_2_k=icc_2_k(matrix),
        krippendorff_alpha_interval=krippendorff_alpha(matrix, level="interval"),
        krippendorff_alpha_nominal=krippendorff_alpha(matrix, level="nominal"),
        mean_pairwise_qwk=pair_qwk,
        mean_pairwise_icc=pair_icc,
        mean_pairwise_mae=_finite(np.mean(item_pair_mae)) if item_pair_mae else None,
        pairwise_exact=_finite(np.mean(item_pair_exact)) if item_pair_exact else None,
        mean_spread_fraction=_finite(np.mean(spread_fractions)) if spread_fractions else None,
        graders=graders,
        pair_count=len(task_pairs),
        normalized=normalized,
    )


def agreement_report(
    items: Sequence[GradedItem],
    *,
    min_common: int = 3,
) -> AgreementReport:
    groups = _group_by_benchmark(items)
    pairs = pairwise_agreement(items, min_common=min_common)
    tasks = {
        benchmark: _task_agreement(benchmark, group, pairs)
        for benchmark, group in sorted(groups.items())
    }
    all_graders = sorted({grader for item in items for grader in item.scores_by_grader})
    task_qwks = [
        task.mean_pairwise_qwk for task in tasks.values() if task.mean_pairwise_qwk is not None
    ]
    overall = {
        "n_items": len(items),
        "n_scored_items": sum(1 for item in items if item.scores),
        "n_scores": sum(len(item.scores) for item in items),
        "n_graders": len(all_graders),
        "n_tasks": len(tasks),
        "mean_task_pairwise_qwk": _finite(np.mean(task_qwks)) if task_qwks else None,
        "min_task_pairwise_qwk": _finite(min(task_qwks)) if task_qwks else None,
        "graders": all_graders,
    }
    return AgreementReport(tasks=tasks, pairs=pairs, overall=overall)


def compare_scores(
    predicted: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    min_rating: float | None = None,
    max_rating: float | None = None,
    step: float | None = None,
) -> dict[str, Any]:
    common = sorted(set(predicted) & set(reference))
    pred = [float(predicted[key]) for key in common]
    ref = [float(reference[key]) for key in common]
    if len(common) < 2:
        return {
            "n": len(common),
            "qwk": None,
            "icc_2_1": None,
            "mae": None,
            "rmse": None,
            "spearman": None,
            "pearson": None,
            "exact": None,
        }
    differences = np.array(pred) - np.array(ref)
    spearman = None
    pearson = None
    try:
        from scipy import stats

        if len(set(pred)) > 1 and len(set(ref)) > 1:
            spearman = _finite(stats.spearmanr(pred, ref).statistic)
            pearson = _finite(stats.pearsonr(pred, ref).statistic)
    except Exception:
        pass
    icc = icc_2_1(np.column_stack([pred, ref]))
    return {
        "n": len(common),
        "qwk": quadratic_weighted_kappa(
            pred, ref, min_rating=min_rating, max_rating=max_rating, step=step
        ),
        "icc_2_1": icc,
        "mae": _finite(np.mean(np.abs(differences))),
        "rmse": _finite(np.sqrt(np.mean(differences**2))),
        "spearman": spearman,
        "pearson": pearson,
        "exact": _finite(np.mean(np.abs(differences) <= 1e-9)),
    }
