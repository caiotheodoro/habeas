"""Tests for habeas_model.train_cli's data-loading step.

Only exercises `_load_sft_records` (bytes/JSONL -> list[dict] with a real
`images` column) — deliberately does NOT call `run_sft` itself, which
needs torch/transformers/trl and a real model to actually train; that's a
Modal-only, GPU-only path this test suite can't and shouldn't run.
"""

import json
import random

from habeas_forge import generate
from habeas_model.dataset_builder import build_record
from habeas_model.train_cli import _load_sft_records
from PIL import Image


def test_load_sft_records_decodes_images(tmp_path):
    rng = random.Random(10)
    task = generate.task(rng, seed=10, n_violations=1)
    record = build_record(task)
    data_path = tmp_path / "sft.jsonl"
    data_path.write_text(json.dumps(record) + "\n")

    loaded = _load_sft_records(str(data_path), smoke=False)
    assert len(loaded) == 1
    assert "image_b64" not in loaded[0]  # popped, replaced by images
    assert "images" in loaded[0]
    assert len(loaded[0]["images"]) == 1
    assert isinstance(loaded[0]["images"][0], Image.Image)
    assert loaded[0]["messages"][0]["role"] == "system"


def test_load_sft_records_smoke_caps_to_eight(tmp_path):
    rng = random.Random(11)
    tasks = [generate.task(rng, seed=11, n_violations=1) for _ in range(12)]
    data_path = tmp_path / "sft.jsonl"
    with open(data_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(build_record(t)) + "\n")

    all_records = _load_sft_records(str(data_path), smoke=False)
    smoke_records = _load_sft_records(str(data_path), smoke=True)
    assert len(all_records) == 12
    assert len(smoke_records) == 8
