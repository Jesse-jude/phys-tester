from __future__ import annotations

import json

from phys_tester.schema import GradedItem


def make_item(
    scores: dict[str, float],
    *,
    benchmark: str = "cpc_bond",
    item_id: str = "item-1",
    response: str = "A structured clinical response.",
    max_score: float | None = None,
    order: int = 0,
    reconciled: float | None = None,
    metadata: dict | None = None,
) -> GradedItem:
    grades = [{"grader": grader, "score": score} for grader, score in scores.items()]
    if reconciled is not None:
        grades.append({"grader": "RECONCILED", "score": reconciled, "reconciled": True})
    record = {
        "id": item_id,
        "benchmark": benchmark,
        "case_id": "case-1",
        "model": "model-a",
        "study": "test",
        "response": response,
        "grade": json.dumps(grades),
        "max_score": max_score,
    }
    if metadata:
        record.update(metadata)
    return GradedItem.from_record(record, order=order)
