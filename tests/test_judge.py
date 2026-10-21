from __future__ import annotations

from phys_tester.judge import (
    MockJudge,
    build_judge_prompt,
    parse_judgment,
)

from tests.helpers import make_item


def test_parse_judgment_fenced_and_think():
    raw = '<think>hidden</think>\n```json\n{"score": 4, "confidence": 0.8, "justification": "clear"}\n```'
    judgment = parse_judgment(raw)
    assert judgment.score == 4
    assert judgment.confidence == 0.8
    assert judgment.justification == "clear"
    assert judgment.error is None


def test_parse_judgment_words_and_fallback():
    judgment = parse_judgment('{"score": "3.5", "confidence": "high"}')
    assert judgment.score == 3.5
    assert judgment.confidence == 0.9
    fallback = parse_judgment("The score is 2.")
    assert fallback.score == 2
    assert fallback.error is None


def test_parse_judgment_missing_score():
    judgment = parse_judgment('{"confidence": 0.7}')
    assert judgment.score is None
    assert judgment.error == "missing_score"
    assert parse_judgment("").error == "empty_response"


def test_prompt_includes_context_not_other_grades():
    item = make_item(
        {"P01": 4, "P02": 5},
        benchmark="cpc_bond",
        response="Neurosarcoidosis is the leading diagnosis.",
        metadata={"final_diagnosis": "Neurosarcoidosis"},
    )
    system, user = build_judge_prompt(item)
    assert "independent physician grader" in system
    assert "Neurosarcoidosis is the leading diagnosis." in user
    assert "Allowed scores" in user
    assert "P01" not in user and "P02" not in user


def test_mock_judge_reads_metadata():
    item = make_item(
        {"P01": 4, "P02": 4},
        metadata={"mock_judge_score": 3, "mock_judge_confidence": 0.4},
    )
    judgment = MockJudge().judge(item)
    assert judgment.score == 3
    assert judgment.confidence == 0.4
    assert judgment.metadata["mock"] is True


def test_mock_judge_mapping():
    item = make_item({"P01": 4, "P02": 4}, item_id="map")
    judgment = MockJudge(scores={"map": 2}, confidence=0.6).judge(item)
    assert judgment.score == 2
    assert judgment.confidence == 0.6
