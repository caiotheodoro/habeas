"""Contamination monitoring per CONTRACTS.md §5."""

from __future__ import annotations

from .schema import Task


def build_train_index(tasks: list[Task]) -> dict[str, str]:
    return {t.signature: t.task_id for t in tasks}


def probe(index: dict[str, str], eval_tasks: list[Task]) -> dict:
    leaked = [t.task_id for t in eval_tasks if t.signature in index]
    return {"leaked": leaked, "n_leaked": len(leaked), "n_eval": len(eval_tasks)}


def split_overlap(train: list[Task], eval_tasks: list[Task]) -> int:
    return len(probe(build_train_index(train), eval_tasks)["leaked"])