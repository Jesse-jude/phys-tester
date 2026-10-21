"""Command-line interface for phys-tester."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .console import build_console_payload
from .demo import write_demo
from .disagreement import DisagreementConfig
from .io import download_grand_rounds, load_items, write_json, write_jsonl
from .judge import CachingJudge, MockJudge, OpenAICompatibleJudge
from .pipeline import run_audit, run_routing
from .quality import GraderDiagnosticsConfig
from .report import audit_markdown, routing_markdown
from .routing import RoutingConfig


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="JSON, JSONL, CSV, or Parquet input")
    parser.add_argument("--benchmark", help="Only load one benchmark/task")
    parser.add_argument("--limit", type=int, help="Maximum number of items to load")


def _add_audit_args(parser: argparse.ArgumentParser) -> None:
    _add_data_args(parser)
    parser.add_argument("--out", default="reports/audit.json", help="JSON report path")
    parser.add_argument("--markdown", default="reports/audit.md", help="Markdown report path")
    parser.add_argument("--min-common", type=int, default=3, help="Minimum shared items per grader pair")
    parser.add_argument("--min-scores", type=int, default=8, help="Minimum scores before flagging a grader")
    parser.add_argument("--bias-z", type=float, default=2.5, help="Robust z threshold for leniency/severity")
    parser.add_argument("--bias-min-effect", type=float, default=0.05, help="Minimum |residual| as a fraction of task scale for bias flags")
    parser.add_argument("--bias-min-partners", type=int, default=2, help="Minimum linked partners with consistent sign for bias flags")
    parser.add_argument("--outlier-rate", type=float, default=0.15, help="Residual outlier-rate threshold")
    parser.add_argument("--drift-rho", type=float, default=0.40, help="Absolute Spearman threshold for drift")
    parser.add_argument("--low-qwk", type=float, default=0.40, help="Pair QWK threshold for low agreement")
    parser.add_argument("--spread-fraction", type=float, default=0.20, help="Item range/scale threshold")
    parser.add_argument("--iqr-fraction", type=float, default=0.15, help="Item IQR/scale threshold")


def _add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge", choices=["mock", "openai"], default="mock")
    parser.add_argument("--model", help="Model id for the OpenAI-compatible judge")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", help="API key (defaults to OPENAI_API_KEY)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cache", help="Optional JSON cache path for judge calls")
    parser.add_argument("--mock-score-key", default="mock_judge_score")
    parser.add_argument("--mock-confidence", type=float, default=0.9)
    parser.add_argument("--mock-confidence-key", default="mock_judge_confidence")
    parser.add_argument("--mock-default-score", type=float)


def _build_judge(args: argparse.Namespace):
    if args.judge == "mock":
        judge = MockJudge(
            confidence=args.mock_confidence,
            default_score=args.mock_default_score,
            score_key=args.mock_score_key,
            confidence_key=args.mock_confidence_key,
        )
    else:
        if not args.model:
            raise ValueError("--model is required when --judge openai is used")
        judge = OpenAICompatibleJudge(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    if args.cache:
        judge = CachingJudge(judge, args.cache)
    return judge


def _cmd_audit(args: argparse.Namespace) -> int:
    items = load_items(args.data, benchmark=args.benchmark, limit=args.limit)
    grader_config = GraderDiagnosticsConfig(
        min_scores=args.min_scores,
        bias_z=args.bias_z,
        bias_min_effect_fraction=args.bias_min_effect,
        bias_min_partners=args.bias_min_partners,
        outlier_rate=args.outlier_rate,
        drift_rho=args.drift_rho,
        low_qwk=args.low_qwk,
        min_pair_common=args.min_common,
    )
    disagreement_config = DisagreementConfig(
        spread_fraction=args.spread_fraction,
        iqr_fraction=args.iqr_fraction,
    )
    report = run_audit(
        items,
        agreement_min_common=args.min_common,
        grader_config=grader_config,
        disagreement_config=disagreement_config,
    )
    write_json(report, args.out)
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(audit_markdown(report), encoding="utf-8")

    summary = report["graders"]["summary"]
    print(
        f"Audited {report['dataset']['n_items']:,} items, "
        f"{report['dataset']['n_scores']:,} scores, "
        f"{summary['n_unique_graders']} graders."
    )
    print(f"Flagged grader/task rows: {len(summary['flagged'])}")
    print(f"Wrote {args.out} and {args.markdown}")
    return 0


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _cmd_route(args: argparse.Namespace) -> int:
    items = load_items(args.data, benchmark=args.benchmark, limit=args.limit)
    judge = _build_judge(args)
    config = RoutingConfig(
        mode=args.mode,
        confidence_threshold=args.confidence,
        max_spread_fraction=args.max_spread_fraction,
        min_graders=args.min_graders,
        judge_tolerance=args.tolerance,
        use_physician_grades_in_triage=args.use_physician_grades_in_triage,
        human_cost_per_item=args.human_cost,
        judge_cost_per_item=args.judge_cost,
        target_precision=args.target_precision,
    )
    report = run_routing(items, judge, config=config)
    write_jsonl(report["decisions"], args.out)
    write_json(report, args.report)
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(routing_markdown(report), encoding="utf-8")

    summary = report["summary"]
    print(
        f"Routed {summary['n_items']:,} items: "
        f"{summary['n_auto']:,} auto-accepted "
        f"({_pct(summary['auto_rate'])}), {summary['n_human']:,} sent to humans."
    )
    if summary["auto_precision"] is not None:
        print(f"Auto-accept precision: {_pct(summary['auto_precision'])}")
    selection = report["calibration"]["selection"] if report.get("calibration") else None
    if selection:
        print(
            f"Recommended confidence threshold: {selection['threshold']:.2f} "
            f"(confidence-only coverage {_pct(selection['coverage'])}, precision "
            f"{_pct(selection['precision'])})"
        )
    print(f"Wrote {args.out}, {args.report}, and {args.markdown}")
    return 0


def _cmd_console(args: argparse.Namespace) -> int:
    items = load_items(args.data, benchmark=args.benchmark, limit=args.limit)
    if args.audit:
        with open(args.audit, encoding="utf-8") as handle:
            audit = json.load(handle)
    else:
        audit = run_audit(items)
    routing = None
    if args.routing:
        with open(args.routing, encoding="utf-8") as handle:
            routing = json.load(handle)
    payload = build_console_payload(
        audit,
        items,
        routing=routing,
        dataset_name=args.dataset_name,
        max_cases=args.max_cases,
        max_response_chars=args.max_response_chars,
    )
    write_json(payload, args.out)
    print(
        f"Wrote console data for {len(payload['cases'])} review cases "
        f"and {len(payload['graders'])} grader/task rows to {args.out}"
    )
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    path = download_grand_rounds(
        args.dest,
        filename=args.file,
        repo_id=args.repo,
        token=args.token,
    )
    items = load_items(path, benchmark=args.benchmark, limit=args.limit)
    print(f"Downloaded {path}")
    print(f"Loaded {len(items):,} items")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    path = write_demo(args.out, n_items=args.n_items, seed=args.seed)
    print(f"Wrote synthetic demo data to {path}")
    print(f"Try: phys-tester audit --data {path}")
    print(
        f"Then: phys-tester route --data {path} --judge mock --mode audit "
        "--out reports/routes.jsonl"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phys-tester",
        description="Audit physician graders, route disagreements, and calibrate an LLM judge.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Measure agreement and flag drifting or outlier graders")
    _add_audit_args(audit_parser)
    audit_parser.set_defaults(func=_cmd_audit)

    route_parser = subparsers.add_parser("route", help="Backtest judge pre-screening and human routing")
    _add_data_args(route_parser)
    _add_judge_args(route_parser)
    route_parser.add_argument("--mode", choices=["audit", "triage"], default="audit")
    route_parser.add_argument("--confidence", type=float, default=0.70)
    route_parser.add_argument("--target-precision", type=float, default=0.90)
    route_parser.add_argument("--tolerance", type=float, help="Absolute judge-human tolerance override")
    route_parser.add_argument("--max-spread-fraction", type=float, default=0.20)
    route_parser.add_argument("--min-graders", type=int, default=2)
    route_parser.add_argument(
        "--use-physician-grades-in-triage",
        action="store_true",
        help="Also gate triage on physician disagreement when scores exist",
    )
    route_parser.add_argument("--human-cost", type=float, default=25.0)
    route_parser.add_argument("--judge-cost", type=float, default=0.01)
    route_parser.add_argument("--out", default="reports/routes.jsonl", help="Per-item routing decisions")
    route_parser.add_argument("--report", default="reports/routing.json", help="Routing summary JSON")
    route_parser.add_argument("--markdown", default="reports/routing.md", help="Routing summary Markdown")
    route_parser.set_defaults(func=_cmd_route)

    fetch_parser = subparsers.add_parser("fetch", help="Download the gated GRAND-ROUNDS dataset")
    fetch_parser.add_argument("--dest", default="data/grand_rounds")
    fetch_parser.add_argument("--file", default="combined_dataset.json")
    fetch_parser.add_argument("--repo", default="tbuckley/GRAND-ROUNDS")
    fetch_parser.add_argument("--token", help="Hugging Face token (defaults to HF_TOKEN)")
    fetch_parser.add_argument("--benchmark")
    fetch_parser.add_argument("--limit", type=int)
    fetch_parser.set_defaults(func=_cmd_fetch)

    console_parser = subparsers.add_parser("console", help="Export audit data for the Grader Quality Console")
    _add_data_args(console_parser)
    console_parser.add_argument("--audit", help="Existing audit JSON (skips re-auditing)")
    console_parser.add_argument("--routing", help="Optional routing JSON with judge pre-screen results")
    console_parser.add_argument("--out", default="reports/console-data.json")
    console_parser.add_argument("--dataset-name", default="GRAND-ROUNDS")
    console_parser.add_argument("--max-cases", type=int, default=100)
    console_parser.add_argument("--max-response-chars", type=int, default=600)
    console_parser.set_defaults(func=_cmd_console)

    demo_parser = subparsers.add_parser("demo", help="Generate synthetic grading data")
    demo_parser.add_argument("--out", default="data/demo.json")
    demo_parser.add_argument("--n-items", type=int, default=240)
    demo_parser.add_argument("--seed", type=int, default=7)
    demo_parser.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
