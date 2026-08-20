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


def test_rlvr_reward_empty_string_unparseable():
    rng = random.Random(23)
    task = generate.task(rng, seed=23, n_violations=1)
    assert rlvr_reward("", task.expected) == -1.0


def test_rlvr_reward_malformed_json_unparseable():
    rng = random.Random(24)
    task = generate.task(rng, seed=24, n_violations=1)
    assert rlvr_reward("{not: valid, json!!", task.expected) == -1.0


def test_rlvr_reward_extra_fields_do_not_break_parsing():
    # Pydantic ignores unrecognized extra fields by default — a completion
    # padded with junk keys should score identically to the clean version,
    # not error out and score as unparseable (that would make the reward
    # brittle to harmless format drift, not a real correctness signal).
    rng = random.Random(25)
    task = generate.task(rng, seed=25, n_violations=1)
    v = task.expected.violations[0]
    clean = {
        "verdict": task.expected.verdict,
        "violations": [{"type": v.type.value, "severity": v.severity.value,
                        "field": v.field, "observed": v.observed,
                        "expected": v.expected, "cfr": v.cfr,
                        "correction": v.correction}],
    }
    padded = json.loads(json.dumps(clean))
    padded["confidence"] = 0.99
    padded["violations"][0]["extra_note"] = "unexpected field"
    r_clean = rlvr_reward(json.dumps(clean), task.expected)
    r_padded = rlvr_reward(json.dumps(padded), task.expected)
    assert r_padded == r_clean


def test_rlvr_reward_no_free_lunch_for_pass_with_violations_listed():
    # A completion claiming PASS while listing violations is internally
    # inconsistent — verdict_correct is False whenever expected is FLAG,
    # so the reward must not reach the fully-correct ceiling even if the
    # listed violation instances happen to match.
    rng = random.Random(26)
    task = generate.task(rng, seed=26, n_violations=1)
    assert task.expected.verdict == "FLAG"  # n_violations=1 guarantees FLAG
    v = task.expected.violations[0]
    inconsistent = json.dumps({
        "verdict": "PASS",
        "violations": [{"type": v.type.value, "severity": v.severity.value,
                        "field": v.field, "observed": v.observed,
                        "expected": v.expected, "cfr": v.cfr,
                        "correction": v.correction}],
    })
    correct = json.dumps({
        "verdict": "FLAG",
        "violations": [{"type": v.type.value, "severity": v.severity.value,
                        "field": v.field, "observed": v.observed,
                        "expected": v.expected, "cfr": v.cfr,
                        "correction": v.correction}],
    })
    assert rlvr_reward(inconsistent, task.expected) < rlvr_reward(correct, task.expected)


def test_rlvr_reward_no_free_lunch_for_flag_with_empty_violations():
    # Claiming FLAG (verdict_correct) but listing zero violations should
    # score worse than actually catching the violation — verdict_bonus
    # alone must not be enough to reach the ceiling.
    rng = random.Random(27)
    task = generate.task(rng, seed=27, n_violations=1)
    empty = json.dumps({"verdict": "FLAG", "violations": []})
    correct = json.dumps({
        "verdict": task.expected.verdict,
        "violations": [{"type": v.type.value, "severity": v.severity.value,
                        "field": v.field, "observed": v.observed,
                        "expected": v.expected, "cfr": v.cfr,
                        "correction": v.correction}
                       for v in task.expected.violations],
    })
    assert rlvr_reward(empty, task.expected) < rlvr_reward(correct, task.expected)


def test_rlvr_reward_duplicated_violations_do_not_inflate_reward():
    # score_predictions dedupes via a set of (type, severity) pairs —
    # repeating the same correct violation many times must not score
    # higher than listing it once (no reward for padding the list).
    rng = random.Random(28)
    task = generate.task(rng, seed=28, n_violations=1)
    v = task.expected.violations[0]
    entry = {"type": v.type.value, "severity": v.severity.value,
            "field": v.field, "observed": v.observed, "expected": v.expected,
            "cfr": v.cfr, "correction": v.correction}
    once = json.dumps({"verdict": task.expected.verdict, "violations": [entry]})
    five_times = json.dumps({"verdict": task.expected.verdict,
                             "violations": [entry] * 5})
    assert rlvr_reward(five_times, task.expected) == rlvr_reward(once, task.expected)


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


def test_oracle_reward_func_conversational_completion_shape():
    # GRPOTrainer wraps completions as [{"role": "assistant", "content":
    # text}] (not a plain string) whenever the dataset is conversational
    # (has a "prompt" column of role/content dicts, which build_rlvr_prompt
    # produces) — found live on the first RLVR smoke run
    # (habeas-rlvr-0820-0244, TypeError inside to_forge_verdict). Must be
    # handled identically to the plain-string shape.
    rng = random.Random(29)
    task = generate.task(rng, seed=29, n_violations=1)
    v = task.expected.violations[0]
    raw = json.dumps({
        "verdict": task.expected.verdict,
        "violations": [{"type": v.type.value, "severity": v.severity.value,
                        "field": v.field, "observed": v.observed,
                        "expected": v.expected, "cfr": v.cfr,
                        "correction": v.correction}],
    })
    conversational = [[{"role": "assistant", "content": raw}]]
    plain = [raw]
    expected_verdict = [task.expected.model_dump_json()]
    assert (oracle_reward_func(conversational, expected_verdict)
            == oracle_reward_func(plain, expected_verdict))
