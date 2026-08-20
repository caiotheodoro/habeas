"""RLVR (GRPO/GSPO/DAPO) training, shared by the Modal app and the GCP
spot fallback — same split as train_cli.py's run_sft: a plain,
filesystem-path-based function with no Modal/GCP-specific code, so both
`cloud/modal_rlvr.py` (bytes -> temp file -> run_rlvr -> volume commit)
and this module's own CLI (used by `cloud/gcp_rlvr.sh`) call the exact
same training logic. Required by docs/TRAINING_PLAN.md §1's
reproducibility contract — one `habeas_model` commit SHA to record, not
two silently-diverging training implementations (the same reason
train_cli.py exists instead of duplicating SFT's logic in modal_train.py).

Input: an RLVR-prompt JSONL matching `dataset_builder.build_rlvr_prompt()`'s
output shape (`{"task_id", "prompt": [...], "images": [b64], "expected_verdict",
"chat_template_kwargs"}`) — build it via `python -m habeas_model.dataset_builder
prompts --tasks-file ... --out ...`. Deliberately distinct from the SFT-trace
JSONL (no assistant turn, no gold trace) so RLVR data structurally cannot
leak into SFT training (methodology.md: "RLVR data never mixed into SFT").
"""

from __future__ import annotations

import base64
import io
import json

import click


def _load_rlvr_records(prompts_path: str, smoke: bool) -> list[dict]:
    records = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if smoke:
        records = records[:8]
    from PIL import Image
    for r in records:
        r["images"] = [Image.open(io.BytesIO(base64.b64decode(b)))
                       for b in r["images"]]
    return records


def run_rlvr(prompts_path: str, base_adapter: str, out_dir: str,
             smoke: bool = False, iters: int = 200, group_size: int = 8,
             max_steps: int | None = None) -> None:
    """Train an RL-tuned adapter on top of an SFT adapter via GRPO/GSPO
    against the forge oracle reward.

    `smoke=True` caps the prompt set to 8 records and bounds training to
    5 steps — a cheap tooling-only validation, mirroring run_sft's smoke
    path (same rationale: catch API/dependency/OOM bugs before any real
    spend, not a fidelity check).

    `max_steps` (independent of `smoke`) lets a caller bound a run to a
    handful of steps at the real prompt/image scale — the way to validate
    real GPU-memory fit before committing to a full run, same pattern as
    train_cli.run_sft's `--max-steps`.
    """
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from .rlvr_reward import oracle_reward_func

    records = _load_rlvr_records(prompts_path, smoke)
    ds = Dataset.from_list(records)  # columns: task_id, prompt, images, expected_verdict, chat_template_kwargs

    # Same VRAM/API constraints as train_cli.run_sft (see its comments and
    # docs/DECISIONS.md): 4-bit required to fit the base model,
    # quantization_config not a bare load_in_4bit kwarg, dtype= not
    # torch_dtype=.
    base = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.8-27B", device_map="auto", dtype="bfloat16",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    model = PeftModel.from_pretrained(base, base_adapter, is_trainable=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=oracle_reward_func,
        processing_class=processor,
        args=GRPOConfig(
            output_dir=out_dir,
            # GSPO (sequence-level importance sampling, the algorithm
            # Qwen3 itself trains with) + DAPO's clip values. loss_type is
            # "grpo" (sequence-length-normalized), NOT "dapo" — TRL itself
            # warns at runtime that pairing importance_sampling_level=
            # "sequence" with loss_type="dapo" sums per-token contributions
            # in a way that doesn't reproduce a true per-sequence objective,
            # and says explicitly: "to reproduce the GSPO paper's setup,
            # set loss_type='grpo'". Found live on the first RLVR smoke run
            # (see docs/DECISIONS.md) — "dapo" was this file's first guess
            # based on it being TRL's plain default, corrected once GSPO's
            # own pairing requirement became clear from the library's own
            # warning, not from a second literature pass.
            importance_sampling_level="sequence",
            loss_type="grpo",
            epsilon=0.2, epsilon_high=0.28,
            num_generations=group_size,
            max_steps=max_steps if max_steps is not None else (5 if smoke else iters),
            # Same fixed fp32-lm_head-upcast OOM as SFT (see train_cli.py's
            # comment / docs/DECISIONS.md) applies here too — GRPOConfig
            # exposes the same fused-loss escape hatch.
            use_liger_kernel=True,
        ),
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(f"{out_dir}-final")


@click.command()
@click.option("--prompts", "prompts_path", required=True, help="RLVR-prompt JSONL path")
@click.option("--base-adapter", required=True, help="SFT adapter dir to RL-tune")
@click.option("--out", "out_dir", required=True, help="checkpoint output dir")
@click.option("--smoke", is_flag=True, default=False)
@click.option("--iters", default=200)
@click.option("--group-size", default=8, help="num_generations (G).")
@click.option("--max-steps", default=None, type=int,
             help="Bound a run to N steps regardless of --smoke (for a "
                  "real-config, small-data GPU-memory validation).")
def main(prompts_path: str, base_adapter: str, out_dir: str, smoke: bool,
        iters: int, group_size: int, max_steps: int | None) -> None:
    run_rlvr(prompts_path, base_adapter, out_dir, smoke=smoke, iters=iters,
             group_size=group_size, max_steps=max_steps)
    click.echo(f"wrote checkpoint to {out_dir}-final")


if __name__ == "__main__":
    main()
