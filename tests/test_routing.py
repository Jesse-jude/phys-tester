from __future__ import annotations

from phys_tester.judge import Judgment
from phys_tester.routing import (
    RoutingConfig,
    calibration_curve,
    route_item,
    route_items,
    select_confidence_threshold,
    summarize_routing,
)

from tests.helpers import make_item


def test_audit_auto_accepts_agreement():
    item = make_item({"P01": 4, "P02": 4}, item_id="ok")
    decision = route_item(item, Judgment(score=4, confidence=0.9), RoutingConfig(mode="audit"))
    assert decision.route == "auto_accept"
    assert decision.reasons == []
    assert decision.correct is True


def test_audit_routes_physician_disagreement():
    item = make_item({"P01": 1, "P02": 5}, item_id="disagree")
    decision = route_item(item, Judgment(score=3, confidence=0.9), RoutingConfig(mode="audit"))
    assert decision.route == "human_review"
    assert "physician_disagreement" in decision.reasons


def test_audit_routes_judge_outlier():
    item = make_item({"P01": 4, "P02": 4}, item_id="outlier")
    decision = route_item(item, Judgment(score=1, confidence=0.95), RoutingConfig(mode="audit"))
    assert decision.route == "human_review"
    assert "judge_outlier" in decision.reasons


def test_low_confidence_routes_to_human():
    item = make_item({"P01": 4, "P02": 4}, item_id="lowconf")
    decision = route_item(item, Judgment(score=4, confidence=0.2), RoutingConfig(mode="audit"))
    assert decision.route == "human_review"
    assert "low_judge_confidence" in decision.reasons


def test_judge_error_routes_to_human():
    item = make_item({"P01": 4, "P02": 4}, item_id="error")
    decision = route_item(item, Judgment(score=None, confidence=0.0, error="boom"), RoutingConfig())
    assert decision.route == "human_review"
    assert "judge_error" in decision.reasons


def test_triage_ignores_physician_disagreement_by_default():
    item = make_item({"P01": 1, "P02": 5}, item_id="triage")
    decision = route_item(
        item,
        Judgment(score=5, confidence=0.95),
        RoutingConfig(mode="triage"),
    )
    assert decision.route == "auto_accept"


def test_summary_and_calibration():
    items = []
    judgments = {}
    for index in range(10):
        item = make_item({"P01": 4, "P02": 4}, item_id=f"good-{index}")
        items.append(item)
        judgments[item.item_id] = Judgment(score=4, confidence=0.95)
    for index in range(10):
        item = make_item({"P01": 1, "P02": 1}, item_id=f"bad-{index}")
        items.append(item)
        judgments[item.item_id] = Judgment(score=5, confidence=0.50)
    decisions = route_items(items, judgments, RoutingConfig(mode="triage", confidence_threshold=0.7))
    summary = summarize_routing(decisions)
    assert summary["n_auto"] == 10
    assert summary["auto_precision"] == 1.0
    assert summary["hard_case_recall"] is None

    curve = calibration_curve(items, judgments)
    selection = select_confidence_threshold(curve, target_precision=0.9)
    assert selection["meets_target"]
    assert selection["coverage"] == 0.5
    assert selection["precision"] == 1.0
