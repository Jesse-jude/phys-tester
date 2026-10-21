"""Synthetic GRAND-ROUNDS-style data for offline demos and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


TASKS = [
    ("management_reasoning", 6.0, 0.5),
    ("cpc_bond", 5.0, 1.0),
    ("diagnostic_reasoning", 19.0, 1.0),
    ("r_idea", 10.0, 1.0),
    ("cpc_management", 2.0, 1.0),
    ("bi_triage", 5.0, 1.0),
]

BOND_SCORES = np.array([0.0, 2.0, 3.0, 4.0, 5.0])
GRADERS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]
MODELS = ["Clinician-A", "Clinician-B", "GPT-synthetic", "Model-X"]

_PHRASES = [
    "The leading diagnosis is supported by the exposure history and focal findings.",
    "Alternative explanations include inflammatory, infectious, and malignant causes.",
    "Recommended next steps include targeted laboratory studies and cross-sectional imaging.",
    "The response is concise, organized, and links each recommendation to the case data.",
    "A reasonable plan should prioritize the highest-risk treatable condition.",
    "The differential is broad but the ranking is not fully justified by the evidence.",
]


def _snap(value: float, task: str, step: float) -> float:
    if task in {"cpc_bond", "bi_triage"}:
        return float(min(BOND_SCORES, key=lambda score: (abs(score - value), score)))
    if step > 0:
        value = round(value / step) * step
    return float(value)


def _true_score(rng: np.random.Generator, task: str, scale: float) -> float:
    if task in {"cpc_bond", "bi_triage"}:
        probabilities = [0.04, 0.10, 0.16, 0.38, 0.32]
        return float(rng.choice(BOND_SCORES, p=probabilities))
    if task == "cpc_management":
        return float(rng.choice([0.0, 1.0, 2.0], p=[0.10, 0.35, 0.55]))
    if task == "management_reasoning":
        return float(rng.choice([0.0, 1.5, 3.0, 4.5, 6.0], p=[0.10, 0.20, 0.40, 0.20, 0.10]))
    if task == "diagnostic_reasoning":
        return float(rng.integers(5, 20))
    if task == "r_idea":
        return float(rng.integers(2, 11))
    return float(rng.uniform(0, scale))


def _grader_score(
    rng: np.random.Generator,
    grader: str,
    truth: float,
    task: str,
    scale: float,
    step: float,
    order: int,
    n_items: int,
) -> float:
    noise = rng.normal(0, 0.055 * scale)
    bias = 0.0
    if grader == "P03":
        bias = 0.11 * scale
    elif grader == "P06":
        bias = -0.09 * scale
    elif grader == "P08":
        bias = (-0.20 + 0.40 * (order / max(1, n_items - 1))) * scale
    elif grader == "P05":
        noise += rng.normal(0, 0.05 * scale)
    elif grader == "P07" and rng.random() < 0.22:
        bias = rng.choice([-1.0, 1.0]) * 0.35 * scale
    value = truth + bias + noise
    return float(np.clip(_snap(value, task, step), 0, scale))


def generate_demo(
    n_items: int = 240,
    *,
    seed: int = 7,
    reconciled_fraction: float = 0.20,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for index in range(n_items):
        task, scale, step = TASKS[index % len(TASKS)]
        truth = _true_score(rng, task, scale)
        n_graders = int(rng.integers(2, 5))
        selected = sorted(rng.choice(GRADERS, size=n_graders, replace=False).tolist())
        grades = []
        for grader in selected:
            score = _grader_score(rng, grader, truth, task, scale, step, index, n_items)
            grades.append({"grader": grader, "score": score})
        if rng.random() < reconciled_fraction:
            grades.append({"grader": "RECONCILED", "score": truth, "reconciled": True})

        observed = [grade["score"] for grade in grades if grade["grader"] != "RECONCILED"]
        consensus = float(np.mean(observed))
        spread_fraction = (max(observed) - min(observed)) / scale if observed else 0.0
        if spread_fraction <= 0.10:
            confidence = 0.92
        elif spread_fraction <= 0.20:
            confidence = 0.72
        else:
            confidence = 0.35
        confidence = float(np.clip(confidence + rng.normal(0, 0.04), 0.05, 0.99))

        judge_score = consensus + rng.normal(0, 0.025 * scale)
        if rng.random() < 0.08:
            judge_score += rng.choice([-1.0, 1.0]) * 0.22 * scale
        judge_score = float(np.clip(_snap(judge_score, task, step), 0, scale))

        phrase_a = str(rng.choice(_PHRASES))
        phrase_b = str(rng.choice(_PHRASES))
        response = f"{phrase_a} {phrase_b} Case-specific reasoning follows for item {index}."
        vignette = (
            f"Synthetic case {index}: a patient presents with a constellation of "
            "symptoms used only to exercise the grading pipeline."
        )
        record: dict[str, Any] = {
            "id": f"demo-{index:04d}",
            "benchmark": task,
            "case_id": f"case-{index // 3:04d}",
            "model": str(rng.choice(MODELS)),
            "study": "synthetic2026",
            "response": response,
            "grade": json.dumps(grades),
            "final_diagnosis": f"Synthetic diagnosis {index % 17}",
            "case_vignette": vignette,
            "question_text": "What is the most appropriate assessment and plan?",
            "question_number": 1,
            "max_score": scale,
            "is_case_released": True,
            "mock_judge_score": judge_score,
            "mock_judge_confidence": confidence,
        }
        if task == "management_reasoning":
            record["rubric"] = {
                "max_score": 6,
                "rubric_items": [
                    {"points": 2, "text": "Prioritizes the highest-risk diagnosis"},
                    {"points": 2, "text": "Selects an appropriate diagnostic test"},
                    {"points": 2, "text": "Includes a safe disposition or follow-up"},
                ],
            }
        if task in {"cpc_bond", "bi_triage", "cpc_management"}:
            record["is_case_released"] = False
        records.append(record)
    return records


def write_demo(path: str | Path, *, n_items: int = 240, seed: int = 7) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(generate_demo(n_items, seed=seed), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path
