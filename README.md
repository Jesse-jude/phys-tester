# Phys-tester

A grader-quality layer for physician-graded clinical AI evaluations.

Open-ended clinical grading is expensive and noisy. Physicians disagree, graders drift, and
frontier LLM judges often disagree with the physicians they are meant to replace. Phys-tester
turns a physician-graded dataset into a quality audit and a routing policy:

- measures inter-rater agreement per task and per grader pair,
- flags lenient, severe, volatile, or drifting graders,
- identifies item-level disagreements and hard cases,
- pre-screens items with a provider-agnostic LLM judge,
- routes disagreements and uncertain items to physicians, and
- estimates auto-accept coverage, precision, and cost savings.

It is designed around the public [GRAND-ROUNDS](https://huggingface.co/datasets/tbuckley/GRAND-ROUNDS)
format, but the loader accepts any JSON, JSONL, CSV, or Parquet file with the same core fields.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,hub]"
```

On macOS or Linux, activate with `source .venv/bin/activate`. The `hub` extra is only needed
to download GRAND-ROUNDS. Core analysis needs `numpy`, `scipy`, `pandas`, and `pyarrow`.

## Quickstart without data

Generate a synthetic six-task panel with injected leniency, severity, volatility, and drift:

```bash
phys-tester demo --out data/demo.json
phys-tester audit --data data/demo.json
phys-tester route --data data/demo.json --judge mock
```

This writes:

```text
reports/audit.json
reports/audit.md
reports/routes.jsonl
reports/routing.json
reports/routing.md
```

The mock judge reads `mock_judge_score` and `mock_judge_confidence` from the demo metadata.
It exists only for offline demos and tests; it is not a clinically meaningful judge.

## GRAND-ROUNDS

GRAND-ROUNDS is gated. Accept the terms on the dataset page, authenticate, then fetch it:

```bash
hf auth login
phys-tester fetch --dest data/grand_rounds
phys-tester audit --data data/grand_rounds/combined_dataset.json
```

Or from Python:

```python
from phys_tester import load_grand_rounds

items = load_grand_rounds("data/grand_rounds")
```

To use a real judge, point the OpenAI-compatible adapter at any hosted or local endpoint:

```bash
set OPENAI_API_KEY=...
phys-tester route ^
  --data data/grand_rounds/combined_dataset.json ^
  --judge openai ^
  --model your-judge-model ^
  --base-url https://api.openai.com/v1 ^
  --mode audit ^
  --out reports/routes.jsonl
```

On POSIX shells use `\` instead of `^` for line continuation. The adapter works with OpenAI,
OpenRouter, vLLM, and other OpenAI-compatible servers. It does not require the `openai`
package.

The judge sees the case vignette, question, reference diagnosis or plan when available, the
rubric, and the response. It never sees physician scores. It returns
`{"score", "confidence", "justification"}`. The public GRAND-ROUNDS release withholds the
Landmark Diagnostic Cases vignettes, so diagnostic-reasoning judgments use the response,
reference diagnosis, and rubric only.

## Grader Quality Console

`phys-tester console` turns an audit into the JSON payload consumed by the React console in
`Grader-Quality-Console`. The console has no backend; it fetches `console-data.json` from its
`public/` directory.

```bash
phys-tester console ^
  --data data/grand_rounds/combined_dataset.json ^
  --audit reports/grand_rounds_audit.json ^
  --dataset-name "GRAND-ROUNDS public release" ^
  --max-cases 80 ^
  --out "Grader-Quality-Console/Grader-Quality-Console/artifacts/grader-quality/public/console-data.json"
```

`--audit` reuses an existing audit so the console export is cheap. The payload includes overview
metrics, per-task agreement and review pressure, all grader/task diagnostics, and the highest-
disagreement cases with scores, flags, and response excerpts. Pass `--routing reports/routing.json`
to populate the pre-screen page with real judge decisions; without it, the page reports that the
judge was not run and shows the audit review policy instead.

The console displays only what is in the export. It never calls a model or writes to the audit.

### Run the console locally

The console source lives in the nested `Grader-Quality-Console/Grader-Quality-Console` Replit
workspace. A small npm wrapper in `Grader-Quality-Console/` runs it without pnpm or the missing
Replit workspace libraries:

```bash
cd Grader-Quality-Console
npm install
npm run dev
```

Open http://127.0.0.1:5173. The wrapper points Vite at the console source, so edits to
`App.tsx` and `index.css` are picked up directly. From the nested repo you can also use
`npm run dev:local`. `npm run build` writes the static bundle to
`Grader-Quality-Console/Grader-Quality-Console/artifacts/grader-quality/dist/public`, and
`npm run preview` serves it on http://127.0.0.1:4173.

Regenerate `public/console-data.json` after rebuilding the audit; the dev server does not run
the Python pipeline.

## Input format

Each row is one scored response with a list of physician grades:

```json
{
  "id": "entry-1",
  "benchmark": "cpc_bond",
  "case_id": "case-42",
  "model": "GPT-model",
  "study": "kanjee2023",
  "response": "Most likely diagnosis: ...",
  "final_diagnosis": "Tuberculous meningitis",
  "max_score": 5,
  "grade": "[{\"grader\": \"P01\", \"score\": 4}, {\"grader\": \"P02\", \"score\": 5}, {\"grader\": \"RECONCILED\", \"score\": 4}]"
}
```

Supported grade entries use `grader`/`rater`/`rater_id` and `score`/`total`/`value`, either
as a JSON string or a native list. Reconciled or final scores (`RECONCILED`, `FINAL`,
`ADJUDICATED`, or `reconciled: true`) are excluded from the independent panel and used as the
reference score when present. Common GRAND-ROUNDS aliases such as `prediction`, `task`,
`vignette`, `diagnosis`, `rubric`, and `test_plan` are recognized.

Run `phys-tester audit --help`, `phys-tester route --help`, or `phys-tester console --help` for all options.

## What the audit computes

### Panel agreement

Per task:

- **Pairwise QWK**: score-weighted Cohen's kappa averaged over grader pairs that share items.
- **Pair ICC**: mean two-way absolute-agreement ICC across grader pairs.
- **Krippendorff's alpha**: interval and nominal versions; handles sparse panels where not
  every grader scored every item.
- **Exact pair agreement** and **mean absolute difference** across within-item pairs.
- **Mean spread**: within-item range divided by the item scale.

Panel ICC(2,1) is also written to JSON, but it is only estimable when at least one item was
scored by every grader. GRAND-ROUNDS panels are often disjoint by study, so pair ICC and
Krippendorff alpha are the primary sparse-panel summaries.

Management-reasoning questions have different maximum credit (for example 6, 9, or 19 points),
so management scores are divided by each item's maximum before agreement, residual, and
judge-agreement calculations. Spread is always reported relative to the item scale. Raw
management scores are retained in the item-level outputs.

### Grader diagnostics

For every grader in every task:

- leave-one-out panel consensus and residual,
- mean and median residual,
- robust bias z-score against the panel median, gated on a connected panel, consistent
  partner differences, and a minimum effect size,
- outlier rate using the pooled residual scale,
- pair QWK with other graders,
- drift Spearman rho and slope against grading order,
- first-half versus second-half residual shift, and
- flags: `lenient`, `severe`, `volatile`, `drifting`, `low_agreement`.

Default thresholds are configurable on the CLI and in `GraderDiagnosticsConfig`:

| Signal | Default |
| --- | --- |
| Minimum scores to assess a grader | 8 |
| Minimum graders for bias z | 3 |
| Bias z | 2.5 |
| Bias minimum effect | 5% of task scale |
| Bias minimum linked partners | 2 |
| Residual outlier z | 3.0 |
| Outlier rate | 15% |
| Drift rho | 0.40 |
| Drift minimum scores | 12 |
| Drift minimum half-split shift | 10% of task scale |
| Low pair QWK | 0.40 |

Bias z answers "is this grader's mean residual unusual relative to the rest of the panel?"
It is deliberately panel-relative and conservative. A lenient or severe flag requires a connected
panel component of at least three graders, consistent signed differences with at least two
partners, and a mean residual of at least 5% of the task scale. With only two graders, a constant
difference cannot be separated from genuine disagreement. GRAND-ROUNDS panels are often
disconnected by study, so graders who never shared an item are never compared, and a large
residual that comes from case mix is not reported as bias.

Drift is also panel-relative. Grading order is the input row order. If the export contains a
timestamp or explicit sequence field, sort the records before loading; otherwise drift reflects
file order and is often confounded with case mix. A drift flag is a prompt to inspect the
sequence, not proof of behavior change.

### Disagreement and hard cases

An item is marked for human review when it has fewer than the minimum number of graders, a
within-item range above a configurable fraction of the task scale, a high IQR, or a bimodal
score pattern. The default spread threshold is 20% of the task scale and the IQR threshold is
15%.

## Routing policy

`--mode audit` is for already physician-graded data. An item goes to a human when:

1. the judge errors, returns no score, or returns an off-scale score;
2. the response is empty;
3. fewer than `--min-graders` physicians scored it;
4. physician disagreement exceeds `--max-spread-fraction`;
5. judge confidence is below `--confidence`; or
6. the judge score is outside the human tolerance band.

Otherwise it is auto-accepted. Tolerance defaults to
`max(0.10 * task_scale, 0.50 * score_step)`, and can be overridden with `--tolerance`.

`--mode triage` is for new, ungraded items. It routes on judge confidence, judge errors, and
basic response checks. Physician disagreement is ignored unless
`--use-physician-grades-in-triage` is set, because ungraded items have no panel scores.

### Calibration

`route` also produces a confidence calibration curve. For each confidence threshold it
reports confidence-only coverage and precision against the physician reference. The
recommended threshold is the one that maximizes auto coverage subject to the target precision
(default 90%). This curve ignores the audit-mode physician and tolerance gates, so the routed
auto rate can differ from confidence-only coverage.

Calibration is in-sample by default. For a real deployment, fit the threshold on a held-out
case split and evaluate it on unseen cases.

### Cost model

The routing summary estimates:

```text
manual cost = n_items * human_cost_per_item
actual cost = n_items * judge_cost_per_item + n_human * human_cost_per_item
savings     = manual cost - actual cost
```

Defaults are `$25` per human review and `$0.01` per judge call. These are placeholders; pass
`--human-cost` and `--judge-cost` to match your setting.

## Validation plan on GRAND-ROUNDS

The public release contains 4,339 entries and 7,395 physician score entries across five tasks
(7,193 independent scores plus 202 adjudicated scores); the full benchmark has 9,217 scores
across six tasks and 11 physicians. A defensible
validation uses cases, not responses, as the split unit:

1. Audit the full release to establish panel agreement and grader flags per task.
2. Split cases into train and held-out folds. Keep all responses for a case in the same fold.
3. Fit judge confidence thresholds on train cases only.
4. On held-out cases, report auto coverage, auto-accept precision, hard-case recall,
   judge-physician QWK/ICC/MAE, and cost per graded item.
5. Compare against three baselines: all-human grading, confidence-only triage without
   physician-disagreement gating, and random routing at the same human workload.
6. Re-run the audit on the held-out fold to check that grader flags are stable.
7. Report results separately by task; scales and disagreement rates differ substantially.

The BIDMC emergency-department task is excluded from the public release because it contains
protected health information. Do not assume results from the five public tasks transfer to it.

## Python API

```python
from phys_tester import (
    OpenAICompatibleJudge,
    RoutingConfig,
    load_items,
    run_audit,
    run_routing,
)

items = load_items("data/grand_rounds/combined_dataset.json")

audit = run_audit(items)
judge = OpenAICompatibleJudge("your-judge-model", base_url="https://api.openai.com/v1")
routing = run_routing(
    items,
    judge,
    config=RoutingConfig(mode="audit", target_precision=0.90),
)

print(audit["agreement"]["overall"])
print(routing["summary"]["auto_rate"], routing["summary"]["auto_precision"])
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

Tests cover grade parsing, reconciled-score handling, QWK/ICC/Krippendorff, synthetic grader
flags, item disagreement, routing modes, calibration, judge parsing, input/output, and the CLI.

## Limitations

- Sparse or disjoint grader assignments make complete-panel ICC unestimable. Use pair QWK,
  pair ICC, and Krippendorff alpha instead.
- Bias detection needs a connected panel of at least three graders with enough shared items. With
  two graders, a systematic difference is observationally identical to disagreement, and
  disconnected study panels cannot be compared directly.
- Drift is confounded with case mix, study, and item order. Order defaults to input row order,
  not wall-clock grading order. Inspect the sequence before acting.
- QWK and interval Krippendorff alpha treat scores as numeric. Ordinal models or task-specific
  decision rules may be more appropriate for some rubrics.
- Judge confidence is self-reported and can be poorly calibrated. The calibration curve and
  held-out validation are not optional for deployment.
- The mock judge is for tests and demos only.
- This is an evaluation-quality tool, not a diagnostic or clinical decision-support system.

## Project layout

```text
phys_tester/
  schema.py        normalized records and grade parsing
  io.py            JSON/JSONL/CSV/Parquet and GRAND-ROUNDS loading
  agreement.py     QWK, ICC, Krippendorff alpha, judge comparison
  quality.py       grader bias, volatility, drift, and flags
  disagreement.py  item-level hard-case detection
  judge.py         provider-agnostic judge adapters and parsing
  routing.py       audit/triage routing, calibration, cost summary
  pipeline.py      end-to-end audit and routing workflows
  report.py        JSON and Markdown report rendering
  console.py       Grader Quality Console payload export
  cli.py           `phys-tester` command
  demo.py          synthetic six-task grading panel
tests/             unit and end-to-end tests
```
