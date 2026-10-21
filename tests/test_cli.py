from __future__ import annotations

import json

from phys_tester.cli import main
from phys_tester.demo import write_demo


def test_cli_audit_and_route(tmp_path):
    data = write_demo(tmp_path / "demo.json", n_items=36, seed=11)
    audit_json = tmp_path / "audit.json"
    audit_md = tmp_path / "audit.md"
    assert main(
        [
            "audit",
            "--data",
            str(data),
            "--out",
            str(audit_json),
            "--markdown",
            str(audit_md),
            "--bias-min-effect",
            "0.10",
            "--bias-min-partners",
            "3",
        ]
    ) == 0
    assert audit_json.exists()
    assert audit_md.exists()
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    assert report["dataset"]["n_items"] == 36
    assert "agreement" in report and "graders" in report and "disagreement" in report

    routes = tmp_path / "routes.jsonl"
    routing_json = tmp_path / "routing.json"
    routing_md = tmp_path / "routing.md"
    assert main(
        [
            "route",
            "--data",
            str(data),
            "--judge",
            "mock",
            "--mode",
            "audit",
            "--out",
            str(routes),
            "--report",
            str(routing_json),
            "--markdown",
            str(routing_md),
        ]
    ) == 0
    assert routes.exists()
    assert routing_json.exists()
    assert routing_md.exists()
    routing = json.loads(routing_json.read_text(encoding="utf-8"))
    assert routing["summary"]["n_items"] == 36
    assert routing["calibration"]["selection"]["threshold"] is not None
    decisions = [json.loads(line) for line in routes.read_text(encoding="utf-8").splitlines()]
    assert len(decisions) == 36
    assert {decision["route"] for decision in decisions} <= {"auto_accept", "human_review"}

    console_json = tmp_path / "console.json"
    assert main(
        [
            "console",
            "--data",
            str(data),
            "--audit",
            str(audit_json),
            "--routing",
            str(routing_json),
            "--out",
            str(console_json),
            "--dataset-name",
            "Demo panel",
            "--max-cases",
            "5",
            "--max-response-chars",
            "120",
        ]
    ) == 0
    console = json.loads(console_json.read_text(encoding="utf-8"))
    assert console["version"] == 1
    assert console["dataset"]["name"] == "Demo panel"
    assert console["dataset"]["items"] == 36
    assert len(console["cases"]) <= 5
    assert console["preScreen"]["available"] is True
    assert console["preScreen"]["decisions"]


def test_cli_demo(tmp_path):
    path = tmp_path / "demo.json"
    assert main(["demo", "--out", str(path), "--n-items", "12", "--seed", "2"]) == 0
    assert path.exists()
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 12
