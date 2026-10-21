"""Normalized records for physician-graded clinical responses."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping

BENCHMARK_MAX_SCORES: dict[str, float] = {
    "cpc_bond": 5.0,
    "bi_triage": 5.0,
    "cpc_management": 2.0,
    "diagnostic_reasoning": 19.0,
    "r_idea": 10.0,
}

BENCHMARK_SCORE_STEPS: dict[str, float] = {
    "management_reasoning": 0.5,
}

RECONCILED_GRADERS = {"RECONCILED", "FINAL", "ADJUDICATED"}

_GRADE_KEYS = ("grade", "grades", "physician_scores", "rater_scores")
_ID_KEYS = ("id", "record_id", "item_id")
_GRADER_KEYS = ("grader", "rater", "rater_id", "physician", "annotator")
_SCORE_KEYS = ("score", "total", "value", "sum", "bond_score", "r_idea_score")

_RECORD_ALIASES: dict[str, tuple[str, ...]] = {
    "benchmark": ("benchmark", "task", "dataset_name"),
    "case_id": ("case_id", "case", "case_number"),
    "model": ("model", "participant_group", "system"),
    "study": ("study", "citation", "source_study"),
    "response": ("response", "prediction", "answer", "output"),
    "final_diagnosis": ("final_diagnosis", "diagnosis", "gold_diagnosis"),
    "case_vignette": ("case_vignette", "vignette", "case_text"),
    "question_text": ("question_text", "question", "prompt"),
    "question_number": ("question_number", "question_no", "qnum"),
    "max_score": ("max_score", "maximum_score", "scale_max"),
    "rubric": ("rubric", "rubrics", "scoring_rubric"),
    "test_plan": ("test_plan", "reference_test_plan"),
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _as_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value)
    return "" if text.strip().lower() in {"nan", "none", "<na>"} else text


def _first(record: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in record and not _is_missing(record[key]):
            return record[key]
    return default


def _parse_jsonish(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def _parse_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        for key in _SCORE_KEYS:
            if key in value:
                return _parse_score(value[key])
    return None


def _is_reconciled(grader: str, entry: Mapping[str, Any]) -> bool:
    if grader.strip().upper() in RECONCILED_GRADERS:
        return True
    flag = entry.get("reconciled")
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() in {"true", "1", "yes", "y"}
    return False


@dataclass(frozen=True)
class Grade:
    grader: str
    score: float
    reconciled: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def is_independent(self) -> bool:
        return not self.reconciled


def _grade_from_entry(entry: Any) -> Grade | None:
    if isinstance(entry, Mapping):
        grader = _first(entry, _GRADER_KEYS)
        score = _first(entry, _SCORE_KEYS)
        if grader is None or _parse_score(score) is None:
            return None
        grader_name = str(grader)
        parsed = _parse_score(score)
        assert parsed is not None
        return Grade(
            grader=grader_name,
            score=parsed,
            reconciled=_is_reconciled(grader_name, entry),
            raw=dict(entry),
        )
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        parsed = _parse_score(entry[1])
        if parsed is None:
            return None
        grader_name = str(entry[0])
        return Grade(grader=grader_name, score=parsed)
    return None


def parse_grades(value: Any) -> list[Grade]:
    parsed = _parse_jsonish(value)
    if parsed is None:
        return []
    if isinstance(parsed, Mapping):
        if any(key in parsed for key in _GRADE_KEYS):
            return parse_grades(parsed[_first_key(parsed, _GRADE_KEYS)])
        if any(key in parsed for key in _GRADER_KEYS) and any(
            key in parsed for key in _SCORE_KEYS
        ):
            grade = _grade_from_entry(parsed)
            return [grade] if grade is not None else []
        grades = []
        for grader, score in parsed.items():
            grade = _grade_from_entry({"grader": grader, "score": score})
            if grade is not None:
                grades.append(grade)
        return grades
    if isinstance(parsed, (list, tuple)):
        grades = []
        for entry in parsed:
            grade = _grade_from_entry(entry)
            if grade is not None:
                grades.append(grade)
        return grades
    return []


def _first_key(record: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in record:
            return key
    return keys[0]


def _stable_id(record: Mapping[str, Any], benchmark: str) -> str:
    parts = [
        benchmark,
        str(_first(record, _RECORD_ALIASES["case_id"], "")),
        str(_first(record, _RECORD_ALIASES["model"], "")),
        str(_first(record, ("run_number", "run"), "")),
        str(_first(record, ("participant_id", "participant"), "")),
        str(_first(record, _RECORD_ALIASES["question_number"], "")),
        str(_first(record, _RECORD_ALIASES["response"], ""))[:120],
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{benchmark}:{digest}"


@dataclass
class GradedItem:
    item_id: str
    benchmark: str
    case_id: str = ""
    model: str = ""
    study: str = ""
    response: str = ""
    final_diagnosis: str = ""
    case_vignette: str = ""
    question_text: str = ""
    question_number: int | str | None = None
    max_score: float | None = None
    rubric: Any = None
    test_plan: str = ""
    grades: list[Grade] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    order: int = 0

    @classmethod
    def from_record(cls, record: Mapping[str, Any], order: int = 0) -> "GradedItem":
        benchmark = str(_first(record, _RECORD_ALIASES["benchmark"], "unknown"))
        explicit_id = _first(record, _ID_KEYS)
        item_id = (
            str(explicit_id)
            if explicit_id is not None
            else _stable_id(record, benchmark)
        )
        grades_raw = _first(record, _GRADE_KEYS)
        max_score = _parse_score(_first(record, _RECORD_ALIASES["max_score"]))
        question_number = _first(record, _RECORD_ALIASES["question_number"])
        if question_number is not None:
            try:
                question_number = int(question_number)
            except (TypeError, ValueError):
                question_number = str(question_number)
        used_keys = set(_GRADE_KEYS) | set(_ID_KEYS)
        for aliases in _RECORD_ALIASES.values():
            used_keys.update(aliases)
        metadata = {
            key: value for key, value in record.items() if key not in used_keys
        }
        return cls(
            item_id=item_id,
            benchmark=benchmark,
            case_id=_as_text(_first(record, _RECORD_ALIASES["case_id"])),
            model=_as_text(_first(record, _RECORD_ALIASES["model"])),
            study=_as_text(_first(record, _RECORD_ALIASES["study"])),
            response=_as_text(_first(record, _RECORD_ALIASES["response"])),
            final_diagnosis=_as_text(_first(record, _RECORD_ALIASES["final_diagnosis"])),
            case_vignette=_as_text(_first(record, _RECORD_ALIASES["case_vignette"])),
            question_text=_as_text(_first(record, _RECORD_ALIASES["question_text"])),
            question_number=question_number,
            max_score=max_score,
            rubric=_parse_jsonish(_first(record, _RECORD_ALIASES["rubric"])),
            test_plan=_as_text(_first(record, _RECORD_ALIASES["test_plan"])),
            grades=parse_grades(grades_raw),
            metadata=metadata,
            order=order,
        )

    @property
    def independent_grades(self) -> list[Grade]:
        return [grade for grade in self.grades if grade.is_independent]

    @property
    def reconciled_grades(self) -> list[Grade]:
        return [grade for grade in self.grades if grade.reconciled]

    @property
    def scores_by_grader(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for grade in self.independent_grades:
            scores.setdefault(grade.grader, grade.score)
        return scores

    @property
    def scores(self) -> list[float]:
        return list(self.scores_by_grader.values())

    @property
    def mean_score(self) -> float | None:
        values = self.scores
        if not values:
            return None
        return float(sum(values) / len(values))

    @property
    def spread(self) -> float | None:
        values = self.scores
        if len(values) < 2:
            return None
        return float(max(values) - min(values))

    @property
    def scale_max(self) -> float:
        observed = self.scores
        observed_max = max(observed) if observed else 0.0
        candidates: list[float] = []
        if self.max_score is not None and self.max_score > 0:
            candidates.append(float(self.max_score))
        if isinstance(self.rubric, Mapping):
            rubric_max = _parse_score(self.rubric.get("max_score"))
            if rubric_max is not None and rubric_max > 0:
                candidates.append(rubric_max)
        default = BENCHMARK_MAX_SCORES.get(self.benchmark)
        if default is not None:
            candidates.append(default)
        base = max(candidates) if candidates else 0.0
        if observed_max > base:
            return observed_max
        if base > 0:
            return base
        return observed_max if observed_max > 0 else 1.0

    @property
    def scale_step(self) -> float:
        values = self.scores
        if any(abs(value - round(value)) > 1e-9 for value in values):
            return 0.5
        return BENCHMARK_SCORE_STEPS.get(self.benchmark, 1.0)

    @property
    def normalized_scores_by_grader(self) -> dict[str, float]:
        scale = self.scale_max
        if scale <= 0:
            return dict(self.scores_by_grader)
        return {
            grader: score / scale
            for grader, score in self.scores_by_grader.items()
        }

    @property
    def normalized_scores(self) -> list[float]:
        return list(self.normalized_scores_by_grader.values())

    @property
    def reference_score(self) -> float | None:
        reconciled = self.reconciled_grades
        if reconciled:
            return float(reconciled[-1].score)
        return self.mean_score

    @property
    def spread_fraction(self) -> float | None:
        spread = self.spread
        if spread is None:
            return None
        return spread / self.scale_max if self.scale_max > 0 else None

    def disagreement(self, minimum: float = 0.0) -> bool:
        fraction = self.spread_fraction
        return fraction is not None and fraction > minimum

    def to_record(self) -> dict[str, Any]:
        record = dict(self.metadata)
        record.update(
            {
                "id": self.item_id,
                "benchmark": self.benchmark,
                "case_id": self.case_id,
                "model": self.model,
                "study": self.study,
                "response": self.response,
                "final_diagnosis": self.final_diagnosis or None,
                "case_vignette": self.case_vignette or None,
                "question_text": self.question_text or None,
                "question_number": self.question_number,
                "max_score": self.max_score,
                "rubric": self.rubric,
                "test_plan": self.test_plan or None,
                "grade": json.dumps(
                    [
                        {
                            "grader": grade.grader,
                            "score": grade.score,
                            **dict(grade.raw),
                        }
                        for grade in self.grades
                    ]
                ),
            }
        )
        return record
