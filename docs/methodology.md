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
2. **RLVR**: GRPO with the Dr. GRPO fix (drop per-token length normalization),
   DAPO-style decoupled clip (high 1.0 / low 0.2), dynamic sampling of
   non-saturated prompts, G≈8–16, outcome-verifier-as-oracle only (no PRM —
   narrow, machine-verifiable output; safest against reward hacking).
3. **Self-play**: ReST-EM / expert iteration rounds + one s1-style curation
   pass (~1k high-difficulty traces).
4. **Judges**: pairwise, position-swapped, temp 0, majority of 3, bootstrap
   CIs, kappa ≥ 0.85 vs a golden-100 set. Only for residual prose.

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