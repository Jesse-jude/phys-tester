"""Provider-agnostic LLM judge adapters and response parsing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .schema import GradedItem

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_SCORE_RE = re.compile(r'"?score"?\s*(?::|=|is|of)?\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r'"?confidence"?\s*(?::|=|is|of)?\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)

_SCALE_HINTS: dict[str, dict[str, Any]] = {
    "cpc_bond": {
        "max": 5.0,
        "allowed": [0.0, 2.0, 3.0, 4.0, 5.0],
        "description": "Bond score for differential diagnosis quality: 5 means the correct diagnosis is in the differential; 0 means it is absent or harmful.",
    },
    "bi_triage": {
        "max": 5.0,
        "allowed": [0.0, 2.0, 3.0, 4.0, 5.0],
        "description": "Bond score for triage or differential diagnosis quality.",
    },
    "cpc_management": {
        "max": 2.0,
        "allowed": [0.0, 1.0, 2.0],
        "description": "Testing-plan score: 2 matches or appropriately extends the reference plan; 1 is reasonable but differs; 0 is not clinically useful.",
    },
    "diagnostic_reasoning": {
        "max": 19.0,
        "step": 1.0,
        "description": "Structured diagnostic-reasoning rubric covering candidate diagnoses, supporting and opposing evidence, the final diagnosis, and next steps.",
    },
    "r_idea": {
        "max": 10.0,
        "step": 1.0,
        "description": "R-IDEA rubric for clinical reasoning documentation: interpretive summary, differential diagnosis, explanation of the leading diagnosis, and explanation of alternatives.",
    },
    "management_reasoning": {
        "step": 0.5,
        "description": "Case-specific management rubric scored in rubric points; scores are multiples of 0.5 up to the item's maximum.",
    },
}

_CONFIDENCE_WORDS = {
    "high": 0.9,
    "medium": 0.6,
    "moderate": 0.6,
    "low": 0.3,
    "very high": 0.95,
    "very low": 0.2,
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class Judgment:
    score: float | None
    confidence: float | None
    justification: str = ""
    raw: str = ""
    model: str = ""
    error: str | None = None
    latency_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Judge(Protocol):
    name: str

    def judge(self, item: GradedItem) -> Judgment:
        ...


def _confidence(value: Any, default: float = 0.5) -> float | None:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[text]
    number = _finite(value)
    if number is None:
        return default
    return max(0.0, min(1.0, number))


def parse_judgment(raw: str | None, *, default_confidence: float = 0.5) -> Judgment:
    if raw is None or not str(raw).strip():
        return Judgment(score=None, confidence=None, raw=raw or "", error="empty_response")
    text = _THINK_RE.sub("", str(raw)).strip()
    candidates = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    obj = _OBJECT_RE.search(text)
    if obj:
        candidates.append(obj.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            score = _finite(parsed.get("score"))
            confidence = _confidence(parsed.get("confidence"), default_confidence)
            justification = parsed.get("justification") or parsed.get("reason") or ""
            if score is None:
                return Judgment(
                    score=None,
                    confidence=confidence,
                    justification=str(justification),
                    raw=str(raw),
                    error="missing_score",
                )
            return Judgment(
                score=score,
                confidence=confidence,
                justification=str(justification),
                raw=str(raw),
            )

    score_match = _SCORE_RE.search(text)
    confidence_match = _CONFIDENCE_RE.search(text)
    score = _finite(score_match.group(1)) if score_match else None
    confidence = _confidence(
        confidence_match.group(1) if confidence_match else None, default_confidence
    )
    if score is None:
        return Judgment(
            score=None,
            confidence=confidence,
            raw=str(raw),
            error="unparsed_score",
        )
    return Judgment(score=score, confidence=confidence, raw=str(raw))


def _scale_for(item: GradedItem) -> dict[str, Any]:
    hint = _SCALE_HINTS.get(item.benchmark, {})
    maximum = item.scale_max
    allowed = hint.get("allowed")
    step = hint.get("step")
    if allowed is not None and maximum > max(allowed):
        allowed = None
    return {
        "min": 0.0,
        "max": maximum,
        "step": step if step is not None else item.scale_step,
        "allowed": allowed,
        "description": hint.get("description", "Use the published rubric for this task."),
    }


def build_judge_prompt(item: GradedItem) -> tuple[str, str]:
    scale = _scale_for(item)
    rubric = item.rubric
    if isinstance(rubric, str):
        rubric_text = rubric
    elif rubric is None:
        rubric_text = "No case-specific rubric was supplied. Apply the task rubric described above."
    else:
        rubric_text = json.dumps(rubric, indent=2)

    allowed_text = (
        f"Allowed scores: {scale['allowed']}."
        if scale["allowed"]
        else f"Score range: {scale['min']} to {scale['max']}, step {scale['step']}."
    )
    context: list[str] = [
        f"Benchmark: {item.benchmark}",
        f"Scoring scale: {scale['description']}",
        allowed_text,
    ]
    if item.case_vignette:
        context.append(f"Case vignette:\n{item.case_vignette}")
    if item.question_text:
        context.append(f"Question:\n{item.question_text}")
    if item.final_diagnosis:
        context.append(f"Reference final diagnosis:\n{item.final_diagnosis}")
    if item.test_plan:
        context.append(f"Reference testing plan:\n{item.test_plan}")
    context.append(f"Rubric:\n{rubric_text}")
    context.append(f"Response to grade:\n{item.response}")

    system = (
        "You are an independent physician grader evaluating an open-ended clinical "
        "response. Score only the response against the supplied rubric. Do not infer "
        "scores from other graders, model identity, or study arm. If the rubric is "
        "ambiguous or the response is borderline, lower your confidence. Return only "
        "a JSON object with exactly three keys: score (number), confidence (0 to 1), "
        "and justification (one or two sentences)."
    )
    user = "\n\n".join(context) + "\n\nReturn the JSON verdict now."
    return system, user


@dataclass
class MockJudge:
    scores: Mapping[str, float] | None = None
    confidence: float | Mapping[str, float] | Callable[[GradedItem], float] = 0.9
    default_score: float | None = None
    score_key: str = "mock_judge_score"
    confidence_key: str = "mock_judge_confidence"
    name: str = "mock"

    def _score(self, item: GradedItem) -> float | None:
        if self.scores and item.item_id in self.scores:
            return float(self.scores[item.item_id])
        value = item.metadata.get(self.score_key)
        if value is not None:
            return _finite(value)
        if self.default_score is not None:
            return float(self.default_score)
        reference = item.reference_score
        return float(reference) if reference is not None else None

    def _confidence(self, item: GradedItem) -> float:
        value = item.metadata.get(self.confidence_key)
        if value is not None:
            return float(value)
        if callable(self.confidence):
            return float(self.confidence(item))
        if isinstance(self.confidence, Mapping):
            return float(self.confidence.get(item.item_id, 0.5))
        return float(self.confidence)

    def judge(self, item: GradedItem) -> Judgment:
        score = self._score(item)
        if score is None:
            return Judgment(
                score=None,
                confidence=0.0,
                model=self.name,
                error="mock_score_missing",
                metadata={"mock": True},
            )
        return Judgment(
            score=score,
            confidence=max(0.0, min(1.0, self._confidence(item))),
            justification="Mock judgment for offline tests and demos.",
            model=self.name,
            metadata={"mock": True},
        )


class OpenAICompatibleJudge:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        retries: int = 2,
        extra_body: Mapping[str, Any] | None = None,
        default_confidence: float = 0.5,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.extra_body = dict(extra_body or {})
        self.default_confidence = default_confidence
        self.name = model

    def judge(self, item: GradedItem) -> Judgment:
        system, user = build_judge_prompt(item)
        started = time.perf_counter()
        try:
            content, usage = self._chat(system, user)
        except Exception as exc:
            return Judgment(
                score=None,
                confidence=0.0,
                model=self.model,
                error=str(exc),
                latency_s=_finite(time.perf_counter() - started),
            )
        judgment = parse_judgment(content, default_confidence=self.default_confidence)
        judgment.model = self.model
        judgment.latency_s = _finite(time.perf_counter() - started)
        judgment.metadata.update({"usage": usage})
        return judgment

    def _chat(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["max_completion_tokens"] = self.max_tokens
        else:
            body["max_tokens"] = self.max_tokens
        body.update(self.extra_body)
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content")
                if content is None:
                    content = message.get("reasoning_content") or ""
                usage = data.get("usage", {}) or {}
                return str(content), dict(usage)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(1.5**attempt)
        raise RuntimeError(f"judge request failed: {last_error}")


class CachingJudge:
    def __init__(self, judge: Judge, path: str | Path = ".judge_cache.json") -> None:
        self.judge = judge
        self.path = Path(path)
        self.name = f"cached:{getattr(judge, 'name', 'judge')}"
        self._cache: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    def _key(self, item: GradedItem) -> str:
        payload = json.dumps(
            {
                "judge": getattr(self.judge, "name", type(self.judge).__name__),
                "item_id": item.item_id,
                "benchmark": item.benchmark,
                "response": item.response,
                "question": item.question_text,
                "vignette": item.case_vignette,
                "rubric": item.rubric,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def judge(self, item: GradedItem) -> Judgment:
        key = self._key(item)
        cached = self._cache.get(key)
        if cached is not None:
            return Judgment(**cached)
        judgment = self.judge.judge(item)
        self._cache[key] = judgment.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        return judgment
