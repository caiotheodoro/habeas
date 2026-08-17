"""Tests for habeas_model.benchmark_eval."""

import json
import random

from habeas_forge import generate

from habeas_model.benchmark_eval import _to_forge_verdict, run_eval


class _EchoOracleProvider:
    """Test double: returns the correct verdict verbatim (scoring sanity
    check), simulating a perfect model."""

    def __init__(self, tasks_by_content):
        self._by_content = tasks_by_content

    def complete(self, system: str, user: str, image_b64: str) -> str:
        return self._by_content[user]


class _UnparseableProvider:
    def complete(self, system: str, user: str, image_b64: str) -> str:
        return "not json at all"


def test_to_forge_verdict_roundtrip():
    raw = json.dumps({
        "verdict": "FLAG",
        "violations": [{"type": "DOC_EXPIRED", "severity": "HIGH", "field": "x",
                        "observed": "o", "expected": "e", "cfr": "c",
                        "correction": "r"}],
    })
    v = _to_forge_verdict(raw)
    assert v is not None
    assert v.verdict == "FLAG"
    assert v.violations[0].type.value == "DOC_EXPIRED"


def test_to_forge_verdict_unparseable_returns_none():
    assert _to_forge_verdict("garbage") is None


def test_run_eval_perfect_provider_scores_1(tmp_path):
    rng = random.Random(4)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks = [generate.task(rng, seed=4, n_violations=1) for _ in range(3)]
    with open(tasks_path, "w") as f:
        for t in tasks:
            f.write(t.model_dump_json() + "\n")

    from habeas_model.dataset_builder import _user_content, build_record
    by_content = {}
    for t in tasks:
        img = build_record(t)["image_b64"]
        raw = json.dumps({
            "verdict": t.expected.verdict,
            "violations": [
                {"type": v.type.value, "severity": v.severity.value, "field": v.field,
                 "observed": v.observed, "expected": v.expected, "cfr": v.cfr,
                 "correction": v.correction}
                for v in t.expected.violations
            ],
        })
        by_content[_user_content(t, img)] = raw

    out_path = tmp_path / "results.jsonl"
    summary = run_eval(str(tasks_path), _EchoOracleProvider(by_content), str(out_path))
    assert summary["parse_rate"] == 1.0
    assert summary["severity_weighted_recall"] == 1.0
    assert summary["n_tasks"] == 3.0

    # resumability: re-running with the same out_path is a no-op read, not a
    # re-score (checkpoint already has all task_ids)
    lines_before = out_path.read_text().count("\n")
    run_eval(str(tasks_path), _EchoOracleProvider(by_content), str(out_path))
    lines_after = out_path.read_text().count("\n")
    assert lines_before == lines_after


def test_run_eval_unparseable_provider_scores_0_recall(tmp_path):
    rng = random.Random(9)
    tasks_path = tmp_path / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        t = generate.task(rng, seed=9, n_violations=1)
        f.write(t.model_dump_json() + "\n")
    out_path = tmp_path / "results.jsonl"
    summary = run_eval(str(tasks_path), _UnparseableProvider(), str(out_path))
    assert summary["parse_rate"] == 0.0
