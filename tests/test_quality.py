from __future__ import annotations

from phys_tester.quality import GraderDiagnosticsConfig, grader_report
from phys_tester.schema import Grade, GradedItem


def test_grader_report_detects_injected_behavior(demo_items):
    report = grader_report(demo_items)
    flagged = report.summary["flagged"]
    assert flagged
    flagged_graders = {entry["grader"] for entry in flagged}
    assert "P03" in flagged_graders or "P06" in flagged_graders or "P08" in flagged_graders
    assert all(row.grader for row in report.graders)
    assert report.summary["status_counts"]


def test_grader_report_allows_threshold_relaxation(demo_items):
    strict = grader_report(
        demo_items,
        config=GraderDiagnosticsConfig(bias_z=99.0, outlier_rate=1.0, drift_rho=0.99, low_qwk=-1.0),
    )
    assert strict.summary["flagged"] == []
    assert all(row.status in {"ok", "insufficient_data"} for row in strict.graders)


def test_task_pair_qwk_excludes_other_benchmarks():
    items = []
    for index, (left, right) in enumerate([(0, 5), (5, 0), (0, 5), (5, 0)]):
        items.append(
            GradedItem(
                item_id=f"a{index}",
                benchmark="task_a",
                max_score=5,
                grades=[Grade("G1", left), Grade("G2", right)],
            )
        )
    for index, (left, right) in enumerate([(0, 0), (1, 1), (2, 2), (3, 3)]):
        items.append(
            GradedItem(
                item_id=f"b{index}",
                benchmark="task_b",
                max_score=3,
                grades=[Grade("G1", left), Grade("G2", right)],
            )
        )
    report = grader_report(
        items,
        config=GraderDiagnosticsConfig(
            bias_z=99.0,
            outlier_rate=1.0,
            drift_rho=0.99,
            low_qwk=-1.0,
            min_panel_graders=99,
        ),
    )
    diagnostics = {(row.benchmark, row.grader): row for row in report.graders}
    poor = diagnostics[("task_a", "G1")]
    perfect = diagnostics[("task_b", "G1")]
    assert poor.pair_count == 1
    assert poor.pair_qwk is not None and poor.pair_qwk < 0.2
    assert perfect.pair_count == 1
    assert perfect.pair_qwk is not None and perfect.pair_qwk > 0.9


def _panel_items(
    scores: dict[str, float], count: int, prefix: str = "item"
) -> list[GradedItem]:
    return [
        GradedItem(
            item_id=f"{prefix}-{index}",
            benchmark="task",
            max_score=5,
            grades=[Grade(grader, score) for grader, score in scores.items()],
        )
        for index in range(count)
    ]


def test_disconnected_two_grader_panel_is_not_flagged_for_bias():
    items = _panel_items({"A": 3, "B": 3, "C": 3}, 8, "core")
    items.extend(_panel_items({"D": 5, "E": 3}, 8, "pair"))
    report = grader_report(items)
    diagnostics = {row.grader: row for row in report.graders}
    for grader in ("D", "E"):
        assert diagnostics[grader].component_size == 2
        assert diagnostics[grader].bias_estimable is False
        assert "lenient" not in diagnostics[grader].flags
        assert "severe" not in diagnostics[grader].flags


def test_consistent_three_grader_bias_is_flagged():
    items = _panel_items({"A": 5, "B": 3, "C": 3}, 12)
    report = grader_report(items)
    diagnostics = {row.grader: row for row in report.graders}
    assert diagnostics["A"].component_size == 3
    assert diagnostics["A"].n_bias_partners == 2
    assert diagnostics["A"].consistent_bias is True
    assert diagnostics["A"].bias_estimable is True
    assert "lenient" in diagnostics["A"].flags
    assert diagnostics["B"].consistent_bias is False
    assert diagnostics["B"].bias_estimable is False
    assert "lenient" not in diagnostics["B"].flags
    assert "severe" not in diagnostics["B"].flags
