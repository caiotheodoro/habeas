"""Tests for habeas_model.rlvr_reward — plain Python, no GPU/Modal."""

import json
import random

from habeas_forge import generate
from habeas_model.rlvr_reward import oracle_reward_func, rlvr_reward


def test_rlvr_reward_matches_forge_score_reward():
    rng = random.Random(20)
    task = generate.task(rng, seed=20, n_violations=1)
    raw = json.dumps({
        "verdict": task.expected.verdict,
        "violations": [
            {"type": v.type.value, "severity": v.severity.value, "field": v.field,
             "observed": v.observed, "expected": v.expected, "cfr": v.cfr,
             "correction": v.correction}
            for v in task.expected.violations
        ],
    })
    r = rlvr_reward(raw, task.expected)
    assert r > 1.0  # perfect match: recall 1.0 + verdict_bonus


def test_rlvr_reward_unparseable():
    rng = random.Random(21)
    task = generate.task(rng, seed=21, n_violations=1)
    assert rlvr_reward("not json", task.expected) == -1.0


def test_oracle_reward_func_batch_aligned():
    rng = random.Random(22)
    tasks = [generate.task(rng, seed=22, n_violations=1) for _ in range(3)]
    completions = []
    expected_verdict = []
    for t in tasks:
        completions.append(json.dumps({
            "verdict": t.expected.verdict,
            "violations": [
                {"type": v.type.value, "severity": v.severity.value, "field": v.field,
                 "observed": v.observed, "expected": v.expected, "cfr": v.cfr,
                 "correction": v.correction}
                for v in t.expected.violations
            ],
        }))
        expected_verdict.append(t.expected.model_dump_json())
    rewards = oracle_reward_func(completions, expected_verdict)
    assert len(rewards) == 3
    assert all(r > 1.0 for r in rewards)
