from __future__ import annotations

import numpy as np

from phys_tester.agreement import (
    agreement_report,
    compare_scores,
    icc_2_1,
    krippendorff_alpha,
    pairwise_agreement,
    quadratic_weighted_kappa,
)

from tests.helpers import make_item


def test_qwk_perfect_and_reversed():
    a = [0, 1, 2, 3, 4]
    assert quadratic_weighted_kappa(a, a) == 1.0
    reversed_score = list(reversed(a))
    value = quadratic_weighted_kappa(a, reversed_score)
    assert value is not None and value < 0


def test_qwk_bond_scale_with_missing_category():
    a = [0, 2, 3, 4, 5, 5, 2]
    assert quadratic_weighted_kappa(a, a, min_rating=0, max_rating=5, step=1) == 1.0


def test_icc_perfect_and_degenerate():
    matrix = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    assert abs(icc_2_1(matrix) - 1.0) < 1e-9
    assert icc_2_1(np.array([[1], [2], [3]])) is None
    assert icc_2_1(np.array([[1, 2]])) is None


def test_krippendorff_alpha():
    units = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    assert abs(krippendorff_alpha(units) - 1.0) < 1e-9
    assert abs(krippendorff_alpha(units, level="nominal") - 1.0) < 1e-9
    assert krippendorff_alpha([[1], [2]]) is None


def test_agreement_report_on_demo(demo_items):
    report = agreement_report(demo_items)
    assert report.overall["n_items"] == 240
    assert report.overall["n_tasks"] == 6
    assert report.overall["mean_task_pairwise_qwk"] is not None
    assert report.tasks["cpc_bond"].n_scores > 0
    assert report.tasks["cpc_bond"].krippendorff_alpha_interval is not None
    pairs = pairwise_agreement(demo_items)
    assert pairs
    assert any(pair.qwk is not None for pair in pairs)


def test_management_pairs_are_scale_normalized():
    items = [
        make_item(
            {"P01": 6.0, "P02": 9.0},
            benchmark="management_reasoning",
            item_id="m1",
            max_score=12.0,
        ),
        make_item(
            {"P01": 4.0, "P02": 4.0},
            benchmark="management_reasoning",
            item_id="m2",
            max_score=8.0,
        ),
    ]
    pairs = pairwise_agreement(items, min_common=2)
    assert len(pairs) == 1
    assert pairs[0].normalized is True
    assert pairs[0].n_common == 2
    assert abs(pairs[0].mae - 0.125) < 1e-9
    assert pairs[0].exact == 0.5
    assert abs(pairs[0].mean_difference + 0.125) < 1e-9


def test_compare_scores():
    reference = {"a": 1, "b": 2, "c": 3, "d": 4}
    predicted = {"a": 1, "b": 2, "c": 2, "d": 4}
    metrics = compare_scores(predicted, reference, min_rating=0, max_rating=5, step=1)
    assert metrics["n"] == 4
    assert metrics["exact"] == 0.75
    assert metrics["mae"] == 0.25
    assert metrics["qwk"] is not None
