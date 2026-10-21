"""Markdown rendering for audit and routing reports."""

from __future__ import annotations

from typing import Any, Mapping


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{number:.1%}"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def audit_markdown(report: Mapping[str, Any]) -> str:
    dataset = report["dataset"]
    agreement = report["agreement"]
    graders = report["graders"]
    disagreement = report["disagreement"]

    lines = [
        "# Grader Quality Audit",
        "",
        "## Dataset",
        "",
        f"- Items: **{dataset['n_items']:,}** ({dataset['n_scored_items']:,} scored)",
        f"- Physician scores: **{dataset['n_scores']:,}**",
        f"- Tasks: **{dataset['n_tasks']}**",
        f"- Mean scores per item: **{_fmt(dataset['mean_scores_per_item'], 2)}**",
        f"- Items without scores: **{dataset['items_without_scores']:,}**",
        "",
        _table(
            ["Task", "Items", "Scores", "Graders"],
            [
                [name, f"{task['n_items']:,}", f"{task['n_scores']:,}", task["n_graders"]]
                for name, task in agreement["tasks"].items()
            ],
        ),
        "",
        "## Inter-rater agreement",
        "",
        _table(
            [
                "Task",
                "Scored items",
                "QWK",
                "Pair ICC",
                "Krippendorff alpha",
                "Exact pair agreement",
                "Mean spread",
            ],
            [
                [
                    name,
                    task["n_scored_items"],
                    _fmt(task["mean_pairwise_qwk"]),
                    _fmt(task["mean_pairwise_icc"]),
                    _fmt(task["krippendorff_alpha_interval"]),
                    _fmt(task["pairwise_exact"], percent=True),
                    _fmt(task["mean_spread_fraction"], percent=True),
                ]
                for name, task in agreement["tasks"].items()
            ],
        ),
        "",
        "QWK is the score-weighted pair agreement averaged over grader pairs. "
        "Pair ICC is the mean two-way absolute-agreement ICC across grader pairs. "
        "Krippendorff alpha handles sparse panels where not every grader scored "
        "every item. Mean spread is the within-item range divided by the item scale. "
        "Management-reasoning scores are divided by each item's maximum before "
        "agreement and residual calculations because maximum credit varies by question.",
        "",
        "## Grader diagnostics",
        "",
        f"- Grader/task rows: **{graders['summary']['n_grader_task_rows']}**",
        f"- Unique graders: **{graders['summary']['n_unique_graders']}**",
        f"- Flagged rows: **{len(graders['summary']['flagged'])}**",
        "",
    ]

    if graders["summary"].get("status_counts"):
        lines.extend(
            [
                _table(
                    ["Status", "Rows"],
                    [
                        [status, count]
                        for status, count in sorted(
                            graders["summary"]["status_counts"].items()
                        )
                    ],
                ),
                "",
            ]
        )

    if graders["summary"]["flag_counts"]:
        lines.extend(
            [
                _table(
                    ["Flag", "Count"],
                    [
                        [flag, count]
                        for flag, count in sorted(
                            graders["summary"]["flag_counts"].items(),
                            key=lambda entry: (-entry[1], entry[0]),
                        )
                    ],
                ),
                "",
            ]
        )

    rows = []
    for row in graders["graders"]:
        if row["status"] == "ok":
            continue
        rows.append(
            [
                row["benchmark"],
                row["grader"],
                row["n_scores"],
                _fmt(row["robust_z_bias"], 2) if row.get("bias_estimable") else "-",
                _fmt(row["pair_qwk"]),
                _fmt(row["drift_rho"], 2),
                _fmt(row["outlier_rate"], percent=True),
                ", ".join(row["flags"]) or row["status"],
            ]
        )
    if rows:
        lines.append(
            _table(
                [
                    "Task",
                    "Grader",
                    "Scores",
                    "Bias z",
                    "Pair QWK",
                    "Drift rho",
                    "Outlier rate",
                    "Flags",
                ],
                rows,
            )
        )
        lines.append("")
    else:
        lines.extend(["No grader/task rows crossed the diagnostic thresholds.", ""])

    lines.extend(
        [
            "Bias z standardizes each grader's mean leave-one-out residual against "
            "the panel median, using the pooled residual scale and the grader's "
            "number of scores. It is only shown and flagged when the grader's "
            "connected panel component has at least three graders, the grader "
            "shares enough items with at least two partners, those partners show a "
            "consistent signed difference, and the residual is at least 5% of the "
            "task scale. With two graders a constant difference is observationally "
            "identical to disagreement, and disconnected study panels can make "
            "case mix look like leniency. Drift rho is the Spearman correlation "
            "between a grader's residual and grading order, and is only flagged "
            "when the first-to-second-half residual shift is also material. Drift "
            "can reflect changing case mix as well as grader behavior, so inspect "
            "the sequence before acting.",
            "",
            "## Disagreement and hard cases",
            "",
            _table(
                [
                    "Task",
                    "Items",
                    "Needs human",
                    "Rate",
                    "Mean spread",
                    "Top flags",
                ],
                [
                    [
                        name,
                        summary["n_items"],
                        summary["n_needs_human"],
                        _fmt(summary["needs_human_rate"], percent=True),
                        _fmt(summary["mean_spread_fraction"], percent=True),
                        ", ".join(
                            f"{flag}={count}"
                            for flag, count in sorted(
                                summary["flag_counts"].items(),
                                key=lambda entry: (-entry[1], entry[0]),
                            )[:3]
                        )
                        or "-",
                    ]
                    for name, summary in disagreement["summary"].items()
                ],
            ),
            "",
            "An item needs human review when it has too few grades, a high "
            "range or IQR relative to the task scale, or a bimodal score pattern.",
            "",
            "## Reading the report",
            "",
            "- QWK and ICC are panel-level summaries; inspect pair agreement "
            "for individual grader conflicts.",
            "- A grader/task row is flagged when its bias, volatility, drift, or "
            "pair agreement crosses a configurable threshold.",
            "- Low coverage is reported but not flagged: graders are often "
            "assigned to subsets of cases by study design.",
        ]
    )
    return "\n".join(lines) + "\n"


def routing_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    cost = summary["cost"]
    lines = [
        "# Grader Routing Backtest",
        "",
        "## Summary",
        "",
        f"- Mode: **{summary['mode']}**",
        f"- Items: **{summary['n_items']:,}**",
        f"- Auto-accepted: **{summary['n_auto']:,}** ({_fmt(summary['auto_rate'], percent=True)})",
        f"- Sent to humans: **{summary['n_human']:,}** ({_fmt(summary['human_rate'], percent=True)})",
        f"- Auto-accept precision (labeled): **{_fmt(summary['auto_precision'], percent=True)}**",
        f"- Hard-case recall: **{_fmt(summary['hard_case_recall'], percent=True)}**",
        f"- Estimated human cost saved: **{_fmt(cost['estimated_savings'], 2)}** "
        f"({_fmt(cost['savings_fraction'], percent=True)})",
        "",
        "## Reasons for human review",
        "",
    ]
    if summary["reason_counts"]:
        lines.append(
            _table(
                ["Reason", "Count"],
                [
                    [reason, count]
                    for reason, count in sorted(
                        summary["reason_counts"].items(),
                        key=lambda entry: (-entry[1], entry[0]),
                    )
                ],
            )
        )
    else:
        lines.append("No items were routed to humans.")
    lines.append("")

    lines.extend(
        [
            "## By task",
            "",
            _table(
                ["Task", "Items", "Auto", "Human", "Auto rate", "Auto precision"],
                [
                    [
                        name,
                        task["n_items"],
                        task["n_auto"],
                        task["n_human"],
                        _fmt(task["auto_rate"], percent=True),
                        _fmt(task["auto_precision"], percent=True),
                    ]
                    for name, task in summary["by_task"].items()
                ],
            ),
            "",
            "## Judge agreement with physician reference",
            "",
            _table(
                ["Task", "N", "QWK", "ICC(2,1)", "MAE", "Spearman"],
                [
                    [
                        name,
                        task["n"],
                        _fmt(task["qwk"]),
                        _fmt(task["icc_2_1"]),
                        _fmt(task["mae"]),
                        _fmt(task["spearman"]),
                    ]
                    for name, task in summary["judge_agreement"].items()
                ],
            ),
            "",
        ]
    )

    calibration = report.get("calibration")
    if calibration:
        selection = calibration["selection"]
        lines.extend(
            [
                "## Confidence calibration",
                "",
                "The curve below ignores the audit-mode physician and tolerance "
                "gates; it answers how well judge confidence predicts correctness "
                "for a triage-only policy. The routed auto rate above still includes "
                "the audit gates.",
                "",
                f"- Target auto-accept precision: **{_fmt(selection['target_precision'], percent=True)}**",
                f"- Recommended confidence threshold: **{selection['threshold']:.2f}**",
                f"- Confidence-only precision: **{_fmt(selection['precision'], percent=True)}**",
                f"- Confidence-only coverage: **{_fmt(selection['coverage'], percent=True)}**",
                f"- Target met: **{_fmt(selection['meets_target'])}**",
                "",
            ]
        )
        at_recommended = calibration.get("at_recommended_threshold")
        if at_recommended:
            lines.extend(
                [
                    "At the recommended threshold, this backtest would auto-accept "
                    f"**{_fmt(at_recommended['auto_rate'], percent=True)}** of items at "
                    f"**{_fmt(at_recommended['auto_precision'], percent=True)}** precision.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Method notes",
            "",
            "- Audit mode sends an item to a human when physicians disagree, when "
            "the panel is too small, when the judge is unconfident, or when the "
            "judge falls outside the human tolerance band.",
            "- Triage mode is for ungraded items: it routes on judge confidence "
            "alone unless `use_physician_grades_in_triage` is enabled.",
            "- Calibration is in-sample unless you fit the threshold on a held-out "
            "case split. Treat the recommended threshold as a starting point.",
            "- Auto-accept precision is measured only on items with a physician "
            "reference score.",
            "- Management-reasoning judge agreement is scale-normalized by each "
            "item's maximum; MAE and QWK are on a 0-1 scale for that task.",
        ]
    )
    return "\n".join(lines) + "\n"
