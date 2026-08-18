"""QLoRA SFT training, shared by the Modal app and the GCP spot fallback.

`run_sft` is a plain, filesystem-path-based function — no Modal/GCP-specific
code — so both `cloud/modal_train.py` (bytes -> temp file -> run_sft ->
volume commit) and this module's own CLI (used by `cloud/gcp_spot.sh`) call
the exact same training logic. Keeping one code path here is required by
docs/TRAINING_PLAN.md §1's reproducibility contract: one `habeas_model`
commit SHA to record, not two silently-diverging training implementations.

Input: a JSONL file matching `dataset_builder.build_record()`'s output
shape (`{"task_id", "messages": [...], "image_b64": str}`). If starting
from a raw forge `Task` JSONL instead, build the SFT JSONL first via
`python -m habeas_model.dataset_builder build --tasks-file ... --out ...`.
"""

from __future__ import annotations

import base64
import io
import json

import click


def _load_sft_records(data_path: str, smoke: bool) -> list[dict]:
    records = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if smoke:
        records = records[:8]
    from PIL import Image
    for r in records:
        img_b64 = r.pop("image_b64", None)
        if not img_b64:
            r["images"] = []
            continue
        img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        if smoke:
            # The full-size rendered page (~700x920) produces enough vision
            # tokens to OOM an L4 even with a 4-bit model and max_length=1024
            # — found via an actual smoke run (fp32 upcast inside trl's
            # chunked cross-entropy: "CUDA out of memory. Tried to allocate
            # 4.74 GiB"), and reducing max_length alone had zero effect,
            # confirming vision tokens (not text length) were the driver.
            # Smoke only validates the tooling, not fidelity, so downscale.
            img = img.resize((224, 224))
        r["images"] = [img]
    return records


def run_sft(data_path: str, out_dir: str, smoke: bool = False, epochs: int = 2) -> None:
    """Train a QLoRA adapter on Qwen3.8-27B from an SFT-record JSONL.

    `smoke=True` caps both the data slice (first 8 records) and the step
    count (max_steps=5) — belt-and-suspenders so a misconfigured smoke run
    can't accidentally become a full, costly training pass.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    records = _load_sft_records(data_path, smoke)
    ds = Dataset.from_list(records)  # columns: task_id, messages, images

    # 4-bit is about VRAM (27B params in bf16 is ~54GB, doesn't fit an L4's
    # 24GB) — orthogonal to smoke vs. real, so always on regardless of
    # `smoke` (a pre-existing `load_in_4bit=not smoke` bug here would have
    # OOM'd every smoke run on real hardware). The installed transformers
    # version rejects a bare `load_in_4bit=` kwarg on from_pretrained
    # (TypeError: unexpected keyword argument) — found by an actual smoke
    # run on GCP, see docs/DECISIONS.md — needs quantization_config instead.
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.8-27B", device_map="auto", dtype="bfloat16",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))

    trainer = SFTTrainer(
        model=model, processing_class=processor,
        args=SFTConfig(
            # max_seq_length was renamed to max_length in the installed trl
            # (found via an actual smoke run — TypeError otherwise).
            # Smoke mode uses a shorter max_length: the fp32 upcast inside
            # trl's chunked cross-entropy loss (h.float() @ w.float().t())
            # OOM'd an L4 at 4096 with the 4-bit model already using ~17GB
            # of 22GB usable VRAM — found via an actual smoke run. Real
            # (non-smoke) runs may need a bigger GPU or further tuning
            # (smaller chunks, activation offload) at 4096; not solved
            # here since it's out of scope for validating the tooling.
            output_dir=out_dir, max_length=1024 if smoke else 4096,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            gradient_checkpointing=True, bf16=True, logging_steps=10,
            num_train_epochs=1 if smoke else epochs,
            max_steps=5 if smoke else -1,
        ),
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(f"{out_dir}-final")


@click.command()
@click.option("--data", "data_path", required=True, help="SFT-record JSONL path")
@click.option("--out", "out_dir", required=True, help="checkpoint output dir")
@click.option("--smoke", is_flag=True, default=False)
@click.option("--epochs", default=2)
def main(data_path: str, out_dir: str, smoke: bool, epochs: int) -> None:
    run_sft(data_path, out_dir, smoke=smoke, epochs=epochs)
    click.echo(f"wrote checkpoint to {out_dir}-final")


if __name__ == "__main__":
    main()
