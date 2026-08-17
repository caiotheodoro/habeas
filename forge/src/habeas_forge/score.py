"""Scoring per CONTRACTS.md §2."""

from __future__ import annotations

from .schema import SEVERITY_WEIGHTS, ViolationType, Verdict
from .verify import types_of


def score_predictions(expected: Verdict, predicted: Verdict | None) -> dict[str, float]:
    parsed = predicted is not None
    if not parsed:
        return {"caught": 0.0, "total": 0.0, "fp": 0.0, "parsed": 0.0}
    exp = {(v.type, v.severity) for v in expected.violations}
    pred = {(v.type, v.severity) for v in predicted.violations}
    total = sum(SEVERITY_WEIGHTS[s] for _, s in exp)
    caught = sum(SEVERITY_WEIGHTS[s] for t, s in exp if (t, s) in pred)
    fp = sum(1 for t, s in pred if (t, s) not in exp)
    return {"caught": caught, "total": total, "fp": fp, "parsed": 1.0}


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    n = len(results)
    if not n:
        return {}
    total_w = sum(r["total"] for r in results)
    return {
        "n_tasks": float(n),
        "parse_rate": sum(r["parsed"] for r in results) / n,
        "severity_weighted_recall": sum(r["caught"] for r in results) / total_w
        if total_w else 1.0,
        "n_violation_tasks": float(sum(1 for r in results if r["total"] > 0)),
        "false_positives_per_task": sum(r["fp"] for r in results) / n,
    }


def class_recall(expected: list[Verdict], predicted: list[Verdict | None],
                 cls: ViolationType) -> float:
    denom = caught = 0
    for e, p in zip(expected, predicted):
        if cls in types_of(e):
            denom += 1
            if p is not None and cls in types_of(p):
                caught += 1
    return caught / denom if denom else 1.0