from __future__ import annotations

from phys_tester.console import build_console_payload
from phys_tester.pipeline import run_audit

from tests.helpers import make_item


def _items():
    return [
        make_item(
            {"P01": 0, "P02": 5},
            item_id=f"case-{index}",
            max_score=5,
            response="A long structured clinical response. _x000D_ " * 20,
        )
        for index in range(8)
    ]


def test_console_payload_exports_audit_data():
    items = _items()
    audit = run_audit(items)
    payload = build_console_payload(audit, items, max_cases=3, max_response_chars=40)
    assert payload["dataset"]["items"] == 8
    assert payload["overview"]["casesForReview"] == 8
    assert len(payload["tasks"]) == 1
    assert payload["tasks"][0]["name"] == "cpc_bond"
    assert payload["tasks"][0]["qwk"] is not None
    assert len(payload["graders"]) == 2
    assert payload["graders"][0]["agreementRate"] is not None
    assert len(payload["cases"]) == 3
    case = payload["cases"][0]
    assert case["id"].startswith("case-")
    assert case["priority"] == "Urgent"
    assert case["disagreementLevel"] == 100
    assert {score["name"] for score in case["scores"]} == {"P01", "P02"}
    assert case["response"].endswith("…")
    assert len(case["response"]) <= 40
    assert "_x000D_" not in case["response"]
    assert payload["preScreen"]["available"] is False


def test_console_payload_matches_ui_contract():
    payload = build_console_payload(run_audit(_items()), _items())
    assert payload["version"] == 1
    assert {"name", "items", "scoredItems", "scores", "tasks", "graders", "models", "studies", "meanScoresPerItem"} <= set(payload["dataset"])
    assert {"meanQwk", "minQwk", "casesForReview", "reviewRate", "stableRows", "flaggedRows", "insufficientRows", "totalRows", "uniqueGraders", "autoCoverage", "autoPrecision", "judgeModel"} <= set(payload["overview"])
    assert {"name", "label", "items", "scoredItems", "scores", "graders", "qwk", "icc", "alpha", "exact", "spread", "needsHuman", "reviewRate", "flagCounts", "scoreScale", "normalized"} <= set(payload["tasks"][0])
    assert {"id", "grader", "task", "label", "initials", "specialty", "scores", "coverage", "agreementRate", "pairQwk", "pairCount", "biasZ", "biasEstimable", "biasEffect", "componentSize", "outlierRate", "driftRho", "flags", "status", "signal"} <= set(payload["graders"][0])
    assert {"id", "task", "taskLabel", "priority", "disagreementLevel", "spreadFraction", "flags", "flagLabels", "nScores", "scoreScale", "scores", "response", "reference", "study", "model", "status", "assignedTo", "updatedAt", "llmConfidence"} <= set(payload["cases"][0])
    assert {"available", "model", "threshold", "autoRate", "autoPrecision", "hardCaseRecall", "reasonCounts", "byTask", "decisions"} <= set(payload["preScreen"])
    assert {"min_graders", "spread_fraction", "iqr_fraction", "bimodal_gap_fraction", "bimodal_min_group"} <= set(payload["reviewConfig"])


def test_console_payload_includes_routing():
    items = _items()
    audit = run_audit(items)
    routing = {
        "config": {"judge_model": "judge-x"},
        "summary": {
            "auto_rate": 0.5,
            "auto_precision": 1.0,
            "hard_case_recall": 1.0,
            "reason_counts": {"low_judge_confidence": 4},
            "by_task": {"cpc_bond": {"n_auto": 4}},
        },
        "calibration": {"selection": {"threshold": 0.8}},
        "decisions": [
            {
                "item_id": "case-0",
                "benchmark": "cpc_bond",
                "route": "human_review",
                "reasons": ["low_judge_confidence"],
                "judge_score": 2.0,
                "judge_confidence": 0.4,
                "physician_spread_fraction": 1.0,
                "within_tolerance": False,
                "correct": False,
            }
        ],
    }
    payload = build_console_payload(audit, items, routing=routing)
    pre = payload["preScreen"]
    assert pre["available"] is True
    assert pre["model"] == "judge-x"
    assert pre["threshold"] == 0.8
    assert pre["autoRate"] == 0.5
    assert pre["decisions"][0]["reasons"] == ["low_judge_confidence"]
