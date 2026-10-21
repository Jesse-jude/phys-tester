from __future__ import annotations

import json

import numpy as np

from phys_tester.demo import write_demo
from phys_tester.io import load_items, read_records
from phys_tester.schema import GradedItem, parse_grades


def test_parse_grades_excludes_reconciled():
    grades = parse_grades(
        json.dumps(
            [
                {"grader": "P01", "score": 4},
                {"grader": "P02", "score": "3.5"},
                {"grader": "RECONCILED", "score": 4, "reconciled": True},
            ]
        )
    )
    assert [(grade.grader, grade.score) for grade in grades] == [
        ("P01", 4.0),
        ("P02", 3.5),
        ("RECONCILED", 4.0),
    ]
    assert [grade.grader for grade in grades if grade.is_independent] == ["P01", "P02"]


def test_graded_item_properties():
    item = GradedItem.from_record(
        {
            "id": "x",
            "benchmark": "cpc_bond",
            "case_id": "c",
            "response": "resp",
            "grade": json.dumps(
                [
                    {"grader": "P01", "score": 2},
                    {"grader": "P02", "score": 5},
                    {"grader": "FINAL", "score": 4, "reconciled": True},
                ]
            ),
            "max_score": 5,
            "extra_field": "kept",
        }
    )
    assert item.scores_by_grader == {"P01": 2.0, "P02": 5.0}
    assert item.mean_score == 3.5
    assert item.spread == 3.0
    assert item.reference_score == 4.0
    assert item.scale_max == 5.0
    assert item.metadata["extra_field"] == "kept"


def test_single_grade_mapping_form():
    grades = parse_grades({"grader": "P01", "score": 4})
    assert len(grades) == 1
    assert grades[0].grader == "P01"
    assert grades[0].score == 4.0


def test_grade_mapping_form():
    item = GradedItem.from_record(
        {
            "benchmark": "r_idea",
            "grade": {"P01": 7, "P02": "8"},
            "response": "x",
            "id": "mapping",
        }
    )
    assert item.scores_by_grader == {"P01": 7.0, "P02": 8.0}


def test_numpy_scores_and_string_question_number():
    item = GradedItem.from_record(
        {
            "id": "management-1",
            "benchmark": "management_reasoning",
            "case_id": "1",
            "question_number": "q1b",
            "max_score": 12,
            "response": "x",
            "grade": [
                {"grader": "P01", "score": np.int64(6)},
                {"grader": "P02", "score": np.float64(9.0)},
            ],
        }
    )
    assert item.scores_by_grader == {"P01": 6.0, "P02": 9.0}
    assert item.question_number == "q1b"
    assert item.scale_max == 12.0
    assert item.normalized_scores_by_grader == {"P01": 0.5, "P02": 0.75}


def test_rubric_supplies_missing_max_score():
    item = GradedItem.from_record(
        {
            "benchmark": "management_reasoning",
            "response": "x",
            "rubric": {"max_score": 9, "rubric_items": []},
            "grade": json.dumps([{"grader": "P01", "score": 4.5}]),
        }
    )
    assert item.scale_max == 9.0
    assert item.scale_step == 0.5


def test_load_demo_roundtrip(tmp_path):
    path = write_demo(tmp_path / "demo.json", n_items=24, seed=3)
    records = read_records(path)
    assert len(records) == 24
    items = load_items(path)
    assert len(items) == 24
    assert items[0].order == 0
    assert items[-1].order == 23
    subset = load_items(path, benchmark="cpc_bond", limit=2)
    assert len(subset) == 2
    assert all(item.benchmark == "cpc_bond" for item in subset)
