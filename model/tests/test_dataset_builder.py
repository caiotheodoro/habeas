"""Tests for habeas_model.dataset_builder."""

import json
import random

from habeas_forge import generate
from habeas_model.dataset_builder import build_dataset, build_record


def test_build_record_matches_expected_verdict():
    rng = random.Random(1)
    task = generate.task(rng, seed=7, n_violations=1)
    record = build_record(task)
    assert record["task_id"] == task.task_id
    assert len(record["messages"]) == 3
    assert record["messages"][0]["role"] == "system"
    target = json.loads(record["messages"][2]["content"])
    assert target["verdict"] == task.expected.verdict
    assert len(target["violations"]) == len(task.expected.violations)
    assert record["image_b64"]


def test_build_dataset_writes_jsonl(tmp_path):
    rng = random.Random(2)
    tasks_path = tmp_path / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for _ in range(5):
            t = generate.task(rng, seed=2, n_violations=1)
            f.write(t.model_dump_json() + "\n")
    out_path = tmp_path / "sft.jsonl"
    n = build_dataset(str(tasks_path), str(out_path))
    assert n == 5
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        rec = json.loads(line)
        assert "messages" in rec and "task_id" in rec
