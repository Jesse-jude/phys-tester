"""Grader-quality tooling for physician-graded clinical AI data."""

__version__ = "0.1.0"

from .agreement import (
    AgreementReport,
    agreement_report,
    compare_scores,
    icc_2_1,
    krippendorff_alpha,
    pairwise_agreement,
    quadratic_weighted_kappa,
)
from .disagreement import DisagreementConfig, disagreement_report, item_disagreement
from .io import download_grand_rounds, load_grand_rounds, load_items, write_json, write_jsonl
from .pipeline import dataset_summary, run_audit, run_routing
from .judge import (
    CachingJudge,
    Judgment,
    MockJudge,
    OpenAICompatibleJudge,
    parse_judgment,
)
from .quality import GraderDiagnosticsConfig, grader_report
from .routing import (
    RoutingConfig,
    calibration_curve,
    route_items,
    select_confidence_threshold,
    summarize_routing,
)
from .schema import Grade, GradedItem

__all__ = [
    "AgreementReport",
    "CachingJudge",
    "DisagreementConfig",
    "Grade",
    "GradedItem",
    "GraderDiagnosticsConfig",
    "Judgment",
    "MockJudge",
    "OpenAICompatibleJudge",
    "RoutingConfig",
    "agreement_report",
    "calibration_curve",
    "compare_scores",
    "dataset_summary",
    "disagreement_report",
    "download_grand_rounds",
    "grader_report",
    "icc_2_1",
    "item_disagreement",
    "krippendorff_alpha",
    "load_grand_rounds",
    "load_items",
    "pairwise_agreement",
    "parse_judgment",
    "quadratic_weighted_kappa",
    "route_items",
    "run_audit",
    "run_routing",
    "select_confidence_threshold",
    "summarize_routing",
    "write_json",
    "write_jsonl",
]
