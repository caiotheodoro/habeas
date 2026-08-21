"""Scoring per CONTRACTS.md §2."""

from __future__ import annotations

from .schema import SEVERITY_WEIGHTS, ViolationType, Verdict
from .verify import types_of


def score_predictions(expected: Verdict, predicted: Verdict | None) -> dict[str, float]:
    parsed = predicted is not None
    if not parsed:
        return {"caught": 0.0, "total": 0.0, "fp": 0.0, "parsed": 0.0,
                "verdict_correct": 0.0, "cfr_correct": 0.0, "cfr_total": 0.0}
    exp = {(v.type, v.severity) for v in expected.violations}
    pred = {(v.type, v.severity) for v in predicted.violations}
    total = sum(SEVERITY_WEIGHTS[s] for _, s in exp)
    caught = sum(SEVERITY_WEIGHTS[s] for t, s in exp if (t, s) in pred)
    fp = sum(1 for t, s in pred if (t, s) not in exp)
    # Citation exact-match (README.md's promotion gate target, >95%):
    # among the violation *instances* the model actually caught (correct
    # type+severity — a miss or a false positive can't be "cited
    # correctly," there's nothing to compare), does its `cfr` string match
    # the oracle's exactly? Denominator is instance count, not
    # severity-weighted, since citation correctness is a per-instance
    # binary property, not a recall-style weighted quantity.
    pred_cfr = {(v.type, v.severity): v.cfr for v in predicted.violations}
    caught_instances = [v for v in expected.violations if (v.type, v.severity) in pred]
    cfr_total = float(len(caught_instances))
    cfr_correct = sum(1.0 for v in caught_instances
                      if pred_cfr.get((v.type, v.severity)) == v.cfr)
    # Violation-instance overlap (caught/total/fp) says nothing about
    # whether the top-level verdict string itself is right or even
    # self-consistent (e.g. "PASS" with violations listed, or "FLAG" with
    # none) — track that separately so callers don't have to infer it.
    verdict_correct = 1.0 if predicted.verdict == expected.verdict else 0.0
    return {"caught": caught, "total": total, "fp": fp, "parsed": 1.0,
            "verdict_correct": verdict_correct, "cfr_correct": cfr_correct,
            "cfr_total": cfr_total}


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    n = len(results)
    if not n:
        return {}
    total_w = sum(r["total"] for r in results)
    cfr_total = sum(r.get("cfr_total", 0.0) for r in results)
    return {
        "n_tasks": float(n),
        "parse_rate": sum(r["parsed"] for r in results) / n,
        "severity_weighted_recall": sum(r["caught"] for r in results) / total_w
        if total_w else 1.0,
        "n_violation_tasks": float(sum(1 for r in results if r["total"] > 0)),
        "false_positives_per_task": sum(r["fp"] for r in results) / n,
        "verdict_accuracy": sum(r["verdict_correct"] for r in results) / n,
        "citation_exact_match": sum(r.get("cfr_correct", 0.0) for r in results) / cfr_total
        if cfr_total else 1.0,
    }


def reward(expected: Verdict, predicted: Verdict | None,
          fp_penalty: float = 0.3, verdict_bonus: float = 0.2,
          unparseable_reward: float = -1.0) -> float:
    """Scalar RLVR reward — unbounded below via fp_penalty*fp (an
    over-flagging model on a clean-PASS task can score below -1.0; there is
    no floor beyond `unparseable_reward` -1.0 minus fp penalties), bounded
    above at 1.0 + verdict_bonus. Outcome-verifier-only, no PRM,
    no PRM, per docs/methodology.md §3.2. Pure function of (expected,
    predicted); never takes raw text (that's a habeas_model-layer concern,
    see habeas_model.benchmark_eval.to_forge_verdict).

    Unparseable output scores worst (no partial credit for malformed JSON).
    A clean expected PASS (total == 0) has no caught/total ratio to reward,
    so it's scored purely on verdict correctness. Every other case rewards
    severity-weighted recall, penalizes false positives, and adds/subtracts
    a verdict_bonus for whether the top-level PASS|FLAG call itself was
    right — closes the gap where instance-level recall could be perfect
    while the verdict string contradicts it (or the oracle).
    """
    r = score_predictions(expected, predicted)
    if not r["parsed"]:
        return unparseable_reward
    if r["total"] == 0.0:
        return (1.0 if r["verdict_correct"] else -1.0) - fp_penalty * r["fp"]
    recall = r["caught"] / r["total"]
    bonus = verdict_bonus if r["verdict_correct"] else -verdict_bonus
    return recall - fp_penalty * r["fp"] + bonus


def class_recall(expected: list[Verdict], predicted: list[Verdict | None],
                 cls: ViolationType) -> float:
    denom = caught = 0
    for e, p in zip(expected, predicted):
        if cls in types_of(e):
            denom += 1
            if p is not None and cls in types_of(p):
                caught += 1
    return caught / denom if denom else 1.0