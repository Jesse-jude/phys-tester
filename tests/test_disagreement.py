from __future__ import annotations

from phys_tester.disagreement import DisagreementConfig, disagreement_report, item_disagreement

from tests.helpers import make_item


def test_high_spread_item_needs_human():
    item = make_item({"P01": 0, "P02": 5})
    result = item_disagreement(item)
    assert result.needs_human
    assert "high_spread" in result.flags
    assert result.spread == 5.0
    assert result.spread_fraction == 1.0


def test_agreeing_item_does_not_need_human():
    item = make_item({"P01": 4, "P02": 4})
    result = item_disagreement(item)
    assert not result.needs_human
    assert result.flags == []


def test_insufficient_graders_flagged():
    item = make_item({"P01": 4})
    result = item_disagreement(item)
    assert result.needs_human
    assert "insufficient_graders" in result.flags


def test_bimodal_detection():
    item = make_item({"P01": 1, "P02": 1, "P03": 5, "P04": 5})
    result = item_disagreement(item, DisagreementConfig(bimodal_gap_fraction=0.25))
    assert "bimodal" in result.flags


def test_disagreement_report_summary(demo_items):
    report = disagreement_report(demo_items)
    assert report["summary"]
    assert sum(task["n_items"] for task in report["summary"].values()) == 240
    assert any(task["n_needs_human"] >= 0 for task in report["summary"].values())
