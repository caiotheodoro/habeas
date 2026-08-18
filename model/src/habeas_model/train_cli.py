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
import os

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


def run_sft(data_path: str, out_dir: str, smoke: bool = False, epochs: int = 2,
           max_steps: int | None = None, save_steps: int = 20) -> None:
    """Train a QLoRA adapter on Qwen3.8-27B from an SFT-record JSONL.

    `smoke=True` caps both the data slice (first 8 records) and image size
    (224x224) plus `max_length=1024` — a cheap tooling-only validation that
    intentionally doesn't reflect real training's GPU-memory footprint.

    `max_steps` (independent of `smoke`) lets a caller bound a run to a
    handful of steps while still using the real config — max_length=4096,
    full-resolution images — the intended way to validate real-scale GPU
    memory fit on a small task count before committing to a full run (the
    smoke config is deliberately too small to answer that question).

    Resumable: saves a checkpoint to `out_dir` every `save_steps` (default
    20, keeping the 3 most recent). If `out_dir` already has a checkpoint
    when called — e.g. the process was interrupted (preemption) and
    restarted, `out_dir` persisting on the same boot disk across a GCE
    preemptible-VM stop/restart cycle — training resumes from there instead
    of starting over. Matters most for a real full run: only preemptible
    A100 quota is available in this project as of 2026-08-18 (see
    docs/DECISIONS.md), and a real run is long enough (~30hr estimated at
    the observed ~135s/step) that losing all progress to one preemption
    would be expensive.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from transformers.trainer_utils import get_last_checkpoint
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
            # use_liger_kernel: trl's default chunked cross-entropy loss
            # still upcasts the full lm_head weight matrix to fp32 per
            # chunk (h.float() @ w.float().t()) — a FIXED ~4.74GiB
            # allocation independent of batch/sequence/image size (found
            # empirically: identical OOM size across three attempts with
            # different max_length and image resolution). That, on top of
            # the 4-bit model's ~17GB footprint, doesn't fit an L4's 22GB
            # usable VRAM. Liger Kernel's fused CE loss avoids materializing
            # the full fp32 logits matrix at all — the actual fix, not a
            # sequence/image-size workaround.
            output_dir=out_dir, max_length=1024 if smoke else 4096,
            use_liger_kernel=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            gradient_checkpointing=True, bf16=True, logging_steps=10,
            num_train_epochs=1 if smoke else epochs,
            max_steps=max_steps if max_steps is not None else (5 if smoke else -1),
            save_strategy="steps", save_steps=save_steps, save_total_limit=3,
        ),
        train_dataset=ds,
    )
    resume_from = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    if resume_from:
        print(f"resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(f"{out_dir}-final")


@click.command()
@click.option("--data", "data_path", required=True, help="SFT-record JSONL path")
@click.option("--out", "out_dir", required=True, help="checkpoint output dir")
@click.option("--smoke", is_flag=True, default=False)
@click.option("--epochs", default=2)
@click.option("--max-steps", default=None, type=int,
             help="Bound a run to N steps regardless of --smoke (for a "
                  "real-config, small-data GPU-memory validation).")
@click.option("--save-steps", default=20,
             help="Checkpoint interval — also the resume granularity if "
                  "the run is interrupted and restarted against the same "
                  "--out dir.")
def main(data_path: str, out_dir: str, smoke: bool, epochs: int, max_steps: int | None,
         save_steps: int) -> None:
    run_sft(data_path, out_dir, smoke=smoke, epochs=epochs, max_steps=max_steps,
           save_steps=save_steps)
    click.echo(f"wrote checkpoint to {out_dir}-final")


if __name__ == "__main__":
    main()
