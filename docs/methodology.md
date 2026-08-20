# Habeas — Methodology (shared across the four forge repos)

This is the recipe that already beat a frontier model with a 1.7B LoRA
(ReconForge, `~/Documents/personal/reconforge`) — now scaled to a 28B
multimodal Qwen on cloud free credits.

## 1. Seeded synthetic data + verifier-as-oracle

- A seeded generator draws true facts, injects a **known** violation set,
  and derives "as printed" fields that realize exactly that set.
- A **deterministic oracle** recomputes the violations from the printed
  fields — no model, no LLM judge.
- A **self-check gate** runs every candidate task through the oracle and
  regenerates on disagreement, so oracle agreement is a property.
- Same seed → same tasks (byte-identical signatures).

## 2. Contamination control

- Task signature = SHA-256 over sorted ground-truth fields.
- Train/val/benchmark are signature-disjoint; a leak probe must fire 1.0 on
  intentional leaks and 0.0 on clean sets (ROC).

## 3. Training (SOTA 2026)

1. **SFT cold-start**: QLoRA 4-bit on verifier-filtered traces distilled from
   a stronger model. RLVR data never mixed in.
2. **RLVR**: GRPO with GSPO sequence-level importance sampling
   (`importance_sampling_level="sequence"` — the algorithm Qwen3 itself
   trains with, natively supported in TRL; fixes GRPO's noisy
   single-sample-per-token importance ratio), paired with
   `loss_type="grpo"` (sequence-length-normalized) **not** `"dapo"` — TRL
   warns at runtime that "dapo"'s per-token summing doesn't reproduce a
   true per-sequence objective when combined with sequence-level
   importance sampling, and says explicitly to use `loss_type="grpo"` to
   reproduce GSPO's actual paper setup (found live on the first RLVR
   smoke run, see docs/DECISIONS.md) — with DAPO's actual paper clip
   values (`epsilon=0.2, epsilon_high=0.28`, not 1.0), dynamic sampling of
   non-saturated prompts, G≈8–16, outcome-verifier-as-oracle only (no PRM —
   narrow, machine-verifiable output; safest against reward hacking).
   **Verifier hardening** before any GPU spend: fuzz the reward function
   against adversarial/edge-case completions (empty, malformed JSON,
   verdict/violation-list mismatches, duplicated violation types) and
   assert no free lunch for a degenerate completion; log
   `frac_reward_zero_std` during training to catch verifier saturation.
3. **Self-play**: ReST-EM / expert iteration rounds + one s1-style curation
   pass (~1k high-difficulty traces). Candidate upgrade once this stage is
   built: best-of-N candidate selection per prompt (continuous
   logprob-based scoring + a cheap pivot-tournament ranking, see
   LLM-as-a-Verifier, arXiv:2607.05391) ahead of the oracle-filter step,
   instead of one shot per prompt — the oracle filter stays the actual
   correctness gate either way, this only changes what's offered to it.
4. **Judges**: pairwise, position-swapped, temp 0, majority of 3, bootstrap
   CIs, kappa ≥ 0.85 vs a golden-100 set. Only for residual prose. Pin the
   judge contract as `(judge_model_id, rubric_version,
   prompt_template_hash)` and bump any field only deliberately, never as
   a side effect of a vendor swap; track judge-vs-human kappa as a
   first-class metric with monthly re-checks; never let a candidate model
   judge itself. Candidate scoring upgrade once built: continuous
   logprob-based scores (LLM-as-a-Verifier, arXiv:2607.05391) instead of
   discrete pairwise verdicts, if calibration against the golden-100 set
   shows it separates quality better.

## 4. Evaluation

- Core metrics are deterministic (oracle-scored). Head-to-head vs frontier
  (Qwen3.8-2.4T-A95B, DeepSeek v4-flash) and vs the base model zero-shot, on
  identical inputs and scoring code.
- Frontier eval runs concurrent with JSONL checkpointing (resumable).

## 5. Cloud (free credits)

- **Modal primary**: `dev-caiotheodoro` profile, $30/mo recurring credit, L4
  24GB ≈ 37 GPU-hrs/mo, free 1 TiB volume for resumable checkpoints.
- **GCP fallback**: `cambio-curitiba-498923` (dev.caiotheodoro@gmail.com),
  Compute API enabled, GPU quota limit 1000. Spot L4/A100 marathon.
- **Gate before training**: smoke LoRA on the base model to validate the
  3-day-old Qwen3.8-27B DeltaNet fine-tuning tooling; fallback TRL+PEFT or
  the FP8 base.

## 6. Honest reporting

Every repo records negative results in `docs/DECISIONS.md` with evidence.
Base-model and eval numbers are self-measured on self-built benchmarks; that
is the point — the methodology is published with the numbers.