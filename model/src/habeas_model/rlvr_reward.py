"""RLVR reward plumbing: raw model completion -> forge oracle reward.

Plain Python, no Modal/GPU dependency — unit-testable in isolation, unlike
the actual GRPO training step (cloud/modal_rlvr.py). Reuses
`habeas_model.schema.to_forge_verdict` (raw text -> Verdict) and
`habeas_forge.score.reward` (Verdict pair -> scalar) rather than
reimplementing either — single source of truth shared with benchmark_eval
and dataset_builder's teacher-trace filtering.
"""

from __future__ import annotations

from habeas_forge.schema import Verdict
from habeas_forge.score import reward

from .schema import to_forge_verdict


def rlvr_reward(raw_completion: str, expected: Verdict) -> float:
    predicted = to_forge_verdict(raw_completion)
    return reward(expected, predicted)


def oracle_reward_func(completions: list, expected_verdict: list[str],
                       **kwargs) -> list[float]:
    """TRL `GRPOTrainer(reward_funcs=...)`-compatible callable.

    `expected_verdict` is a per-example dataset column (JSON-serialized
    forge `Verdict`, see `dataset_builder.build_rlvr_prompt`) — TRL repeats
    every other dataset column to match `num_generations` per prompt, so
    `completions` and `expected_verdict` are already aligned 1:1.

    `completions` shape depends on whether the dataset is conversational
    (has a "prompt" column of role/content dicts, which ours does): TRL's
    GRPOTrainer wraps each completion as `[{"role": "assistant",
    "content": text}]` rather than a plain string in that case (see
    `grpo_trainer.py`'s `_calculate_rewards` — `is_conversational(inputs[0])`
    branch). A plain-string reward_func assuming raw text broke on the
    very first live RLVR smoke run (`TypeError: expected string or
    bytes-like object` inside `to_forge_verdict`) — handle both shapes.
    """
    out = []
    for completion, exp_json in zip(completions, expected_verdict):
        raw = completion[0]["content"] if isinstance(completion, list) else completion
        expected = Verdict.model_validate_json(exp_json)
        out.append(rlvr_reward(raw, expected))
    return out
