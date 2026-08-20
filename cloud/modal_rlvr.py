"""Modal RLVR app: GSPO (sequence-level importance sampling) + DAPO
decoupled clip against the forge oracle reward. SFT adapter -> RL-tuned
adapter. See docs/TRAINING_PLAN.md §Stage 3.

Run: modal run cloud/modal_rlvr.py --prompts data/rlvr-prompts.jsonl \
       --base-adapter /checkpoints/sft-final --smoke

`prompts` is an RLVR-prompt JSONL (see `habeas_model.dataset_builder.
build_rlvr_prompt` / the `prompts` CLI subcommand) — deliberately distinct
from the SFT-trace JSONL `modal_train.py` consumes, so RLVR data can never
be pointed at the SFT dataset by accident (methodology.md: "RLVR data
never mixed into SFT").

TRL API notes (re-verified 2026-08-20 against the installed model/.venv
trl source directly, correcting two wrong values from the original
design pass — see docs/DECISIONS.md's RLVR-research entry):
- `GRPOConfig(importance_sampling_level="sequence")` — GSPO (sequence-
  level importance sampling instead of GRPO's noisy token-level ratio).
  This is the algorithm Qwen3 itself trains with, natively supported in
  TRL, and directly relevant since our base model *is* Qwen3.8-27B
  (arXiv:2507.18071).
- `GRPOConfig.loss_type="grpo"` (sequence-length-normalized), **not**
  `"dapo"` despite `"dapo"` being TRL's plain default — TRL itself warns
  at runtime that pairing `importance_sampling_level="sequence"` with
  `loss_type="dapo"` sums per-token contributions in a way that doesn't
  reproduce a true per-sequence objective, and says explicitly to set
  `loss_type="grpo"` to reproduce GSPO's actual paper setup. Found live
  on the first RLVR smoke run (`habeas-rlvr-0820-0244`, see
  docs/DECISIONS.md) — `"dapo"` was this file's first guess based on it
  being TRL's plain default; corrected once GSPO's own pairing
  requirement became clear from the library's own warning.
- `GRPOConfig(epsilon=0.2, epsilon_high=0.28)` is DAPO's actual paper
  clip value — the earlier `epsilon_high=1.0` here was simply wrong,
  confirmed against the installed `grpo_config.py` docstring ("Paper DAPO
  recommends 0.28").
- Dynamic sampling of non-saturated prompt groups (DAPO's "drop/resample
  zero-reward-variance groups") is **not** natively supported — GRPOTrainer
  only logs `frac_reward_zero_std`, it doesn't filter. Deliberately shipped
  without it in v1 (monitor the metric; a custom GRPOTrainer subclass is a
  documented follow-up if saturation turns out to matter empirically, not
  built preemptively).
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("habeas-rlvr")
vol = modal.Volume.from_name("habeas-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile(str(Path(__file__).parent / "Dockerfile"))


@app.function(image=image, gpu="L4", volumes={"/checkpoints": vol},
              timeout=60 * 60 * 12)
def rlvr(prompts: bytes, base_adapter: str, iters: int = 200,
         group_size: int = 8, smoke: bool = False) -> str:
    import tempfile

    from habeas_model.rlvr_cli import run_rlvr

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
        f.write(prompts)
        prompts_path = f.name
    run_rlvr(prompts_path, base_adapter, "/checkpoints/rlvr", smoke=smoke,
             iters=iters, group_size=group_size)
    vol.commit()
    return "done"
