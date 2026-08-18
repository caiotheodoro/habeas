"""Modal RLVR app: GRPO (Dr. GRPO + DAPO decoupled clip) against the forge
oracle reward. SFT adapter -> RL-tuned adapter. See docs/TRAINING_PLAN.md
§Stage 3.

Run: modal run cloud/modal_rlvr.py --prompts data/rlvr-prompts.jsonl \
       --base-adapter /checkpoints/sft-final --smoke

`prompts` is an RLVR-prompt JSONL (see `habeas_model.dataset_builder.
build_rlvr_prompt` / the `prompts` CLI subcommand) — deliberately distinct
from the SFT-trace JSONL `modal_train.py` consumes, so RLVR data can never
be pointed at the SFT dataset by accident (methodology.md: "RLVR data
never mixed into SFT").

TRL API notes (verified against installed trl==0.24.0 during design, see
docs/DECISIONS.md — re-verify against whatever trl version actually
resolves at image-build time, since cloud/Dockerfile pins `trl>=0.16` with
no upper bound):
- `GRPOConfig.loss_type="dr_grpo"` is a native option (Dr. GRPO: no
  per-token length normalization).
- `GRPOConfig(epsilon=0.2, epsilon_high=1.0)` is DAPO's decoupled clip —
  two native, separate fields.
- Dynamic sampling of non-saturated prompt groups (DAPO's "drop/resample
  zero-reward-variance groups") is **not** natively supported — GRPOTrainer
  only logs `frac_reward_zero_std`, it doesn't filter. Deliberately shipped
  without it in v1 (monitor the metric; a custom GRPOTrainer subclass is a
  documented follow-up if saturation turns out to matter empirically, not
  built preemptively).
"""

from __future__ import annotations

import modal

app = modal.App("habeas-rlvr")
vol = modal.Volume.from_name("habeas-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile")


@app.function(image=image, gpu="L4", volumes={"/checkpoints": vol},
              timeout=60 * 60 * 12)
def rlvr(prompts: bytes, base_adapter: str, iters: int = 200,
         group_size: int = 8, smoke: bool = False) -> str:
    import base64
    import io
    import json

    from datasets import Dataset
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    from habeas_model.rlvr_reward import oracle_reward_func

    records = [json.loads(line) for line in prompts.decode("utf-8").splitlines()
              if line.strip()]
    if smoke:
        records = records[:8]
    for r in records:
        r["images"] = [Image.open(io.BytesIO(base64.b64decode(b)))
                       for b in r.pop("images")]
    ds = Dataset.from_list(records)  # columns: task_id, prompt, images, expected_verdict

    base = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.8-27B", device_map="auto", torch_dtype="bfloat16")
    model = PeftModel.from_pretrained(base, base_adapter, is_trainable=True)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.8-27B")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=oracle_reward_func,
        processing_class=processor,
        args=GRPOConfig(
            output_dir="/checkpoints/rlvr",
            loss_type="dr_grpo",
            epsilon=0.2, epsilon_high=1.0,
            num_generations=group_size,
            max_steps=5 if smoke else iters,
        ),
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model("/checkpoints/rlvr-final")
    vol.commit()
    return "done"
