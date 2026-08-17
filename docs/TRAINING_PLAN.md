# Habeas — P4 Training Plan (SFT cold-start → RLVR → self-play → judges)

Design-only document — no cloud jobs have been launched from this plan.
Follows `docs/methodology.md` (shared recipe across specula/suture/plumb/
habeas) and applies `mlops-pipeline-design` discipline: data/code/model
versioned independently, promotion gated on a fixed eval set, rollback is a
pointer flip, not a redeploy.

## 0. Gate status (blocking, shared)

Smoke LoRA on Modal — validates the 3-day-old Qwen3.8-27B DeltaNet
fine-tuning tooling before any real spend — is **not yet run** in specula
(the lead repo) or here. Checked `specula/docs/DECISIONS.md`: no entry.
**Do not start Stage 2 (real SFT) until either specula records a green
smoke-LoRA decision, or habeas runs its own smoke LoRA independently** (a
deliberate scope decision the user deferred earlier this session — revisit
if specula's timeline is unclear). Stages 1 and the dataset/reward-function
design work below have no cloud-spend dependency and can proceed now.

## 1. Reproducibility contract (per training run, logged in DECISIONS.md)

Every SFT/RLVR run must record, before it's called "done":
- **Data snapshot**: exact `data/train.jsonl`/`val.jsonl`/`golden.jsonl`
  provenance — generator seed, `n`, and the git commit of `forge/` at
  generation time (task signatures are seed+code deterministic, but the
  generator itself can change — see the P2 entropy-bug fix as a concrete
  example of why this matters).
- **Code commit**: the exact `habeas_model`/`habeas_forge` commit SHA.
- **Env**: `model/uv.lock` hash (already committed) + the Modal image
  digest (`cloud/Dockerfile` build hash).
- **Hyperparams**: full config (see Stage 2/3 below), not just "QLoRA".
- **Seed**: training seed, separate from data-generation seed.

This closes the "pipeline ran without error ≠ validated" gap and the
"versioned the model but not the data snapshot" mistake called out in the
mlops-pipeline-design skill.

## 2. Stage 1 — scale up the training corpus

Current `data/train.jsonl` has 301 tasks (from a 400-task pilot). That's
enough to validate the pipeline (which is what P0-P3 did) but too small for
a real SFT run. Before Stage 2:
- Generate a larger pilot: `cli pilot --seed 7 --n 5000` (or higher — no
  code change needed, existing CLI), re-split, re-leakprobe. Pick `n` based
  on the smoke-LoRA GPU-hours budget (Modal: ~37 GPU-hrs/mo on the free
  tier) once Stage 0 clears and gives a real per-step timing.
- Re-verify `data/golden.jsonl` (seed 777) stays zero-overlap against the
  larger train/val — rerun the same 3-way check used in the P2 commit.
- Stratification is already handled by `cli split` (deciles × violation
  class) — no design change, just scale.

## 3. Stage 2 — SFT cold-start

Per methodology.md: QLoRA 4-bit on **verifier-filtered traces distilled
from a stronger model**. Concretely:

1. **Teacher traces**: implement a teacher `Provider` (reusing the
   `habeas_model.benchmark_eval.Provider` protocol already built) pointed
   at a frontier model (Qwen3.8-2.4T-A95B or DeepSeek v4-flash — same
   models named in the head-to-head target). Run it over `train.jsonl` via
   `run_eval`-style batched calls, but **keep only traces whose predicted
   `Verdict` matches the oracle exactly** (severity-weighted recall == 1.0,
   zero false positives) — that's the "verifier-filtered" step; RLVR data
   must never be mixed into this set (methodology.md, explicit).
2. **Dataset**: `habeas_model.dataset_builder.build_dataset()` already
   produces the right chat-format shape; swap the assistant turn from
   ground truth to the filtered teacher trace (a small extension — a
   `target_source: Literal["oracle", "teacher"]` param) so both a
   ground-truth-only baseline and a teacher-distilled run are buildable
   from the same code path (no train/serve skew between them).
3. **Fix the Modal wiring bug found during P3**: `cloud/modal_train.py`'s
   `train()` currently hardcodes `train_dataset=[]` and never uses its
   `data: bytes` param (flagged in `docs/HANDOFF.md`). Fix: deserialize
   `data` into a HF `datasets.Dataset` from the SFT JSONL, wire it into
   `SFTTrainer`. `cloud/gcp_spot.sh`'s startup script references
   `python3 -m habeas_model.train --data data/train.jsonl` — that module
   doesn't exist yet either; either build a thin CLI wrapper around
   `dataset_builder` + the Modal `train()` call, or update the script to
   match whatever entrypoint is actually built (don't leave the two
   inconsistent).
4. **Starting hyperparams** (already scaffolded in `modal_train.py`, keep
   unless the smoke run says otherwise): LoRA r=32, alpha=64, dropout=0.05,
   target_modules=all-linear, seq_len=4096, per-device batch=1,
   grad_accum=4, bf16, gradient checkpointing on.
5. **Registry**: Modal volume `habeas-checkpoints` already exists (per
   HANDOFF.md environment section) — adopt a `current -> <run-id>` pointer
   file convention (not implemented yet) so promotion/rollback is a pointer
   flip, not a redeploy, per the skill's rollback guidance.

## 4. Stage 3 — RLVR (GRPO / Dr. GRPO, DAPO-style)

`cloud/modal_rlvr.py` does not exist yet in habeas (specula's own copy is
still a `NotImplementedError` stub — habeas would need to build this from
scratch, not port a working implementation).

- **Reward**: outcome-verifier-only, no PRM (methodology.md — narrow,
  machine-verifiable output, safest against reward hacking). Concretely:
  `reward = severity_weighted_recall - fp_penalty * false_positives_per_task`
  computed via the **existing** `habeas_forge.score.score_predictions`/
  `summarize` — reuse, don't reimplement (avoids train/serve/eval skew
  across SFT, RLVR, and benchmark scoring all using the same scorer).
- **Algorithm**: Dr. GRPO fix (drop per-token length normalization — avoids
  the reward-hacking-via-verbosity failure mode), DAPO decoupled clip
  (high=1.0, low=0.2), dynamic sampling of non-saturated prompts, G≈8–16
  rollouts per prompt.
- **Prompts**: sample from `train.jsonl` (never `val.jsonl`/`golden.jsonl`);
  keep an explicit RLVR-only data pointer distinct from the SFT trace set
  per methodology.md's "RLVR data never mixed into SFT" rule (and vice
  versa — SFT traces shouldn't leak into RLVR prompt sampling either, though
  the base tasks can overlap since only the *verdict trace*, not the task
  itself, is the sensitive artifact here).
- **Base checkpoint**: the SFT adapter from Stage 2, not the base model.

## 5. Stage 4 — self-play (ReST-EM / expert iteration + s1 curation)

- ReST-EM rounds: generate, oracle-filter, retrain — same filter discipline
  as Stage 2's teacher-distillation step (reuse, don't reimplement).
- One s1-style curation pass: ~1k highest-difficulty traces (use `Task`'s
  existing `difficulty`/`ocr_noise_level` fields already threaded through
  the generator as the difficulty signal — no new field needed).

## 6. Stage 5 — evaluation (golden benchmark + judges)

- Core metrics: `habeas_model.benchmark_eval.run_eval` against
  `data/golden.jsonl` (seed 777, already generated, zero-overlap-verified)
  — already built and tested, just needs a real `Provider` wired in (see
  HANDOFF.md's "P3 remainder").
- Judges (residual prose fields only — `observed`/`correction` text
  quality, never core verdicts): pairwise, position-swapped, temp 0,
  majority of 3, bootstrap CIs, kappa ≥ 0.85 vs a golden-100 hand-labeled
  subset. Not yet built — new `model/` module, out of scope until Stage 2/3
  land (no point calibrating a judge before there's a model to judge).
- Head-to-head: Qwen3.8-2.4T-A95B, DeepSeek v4-flash, base Qwen3.8-27B
  zero-shot — same `run_eval` harness, same system prompt, identical inputs
  per CONTRACTS.md §6.

## 7. Promotion gate (CI-style, before calling any checkpoint "the model")

A checkpoint is promotable only if it clears every README.md target on the
golden benchmark:

| Metric | Target |
|---|---|
| Oracle agreement on adversarial suite | 100% |
| Severity-weighted violation recall | > 0.95 |
| Citation exact-match | > 95% |
| Timeliness/reverification flags | 100% |
| Parse rate | 100% |

No partial-credit promotion. A checkpoint that clears 4/5 targets stays
`current`-unpromoted; the run and its shortfall get a DECISIONS.md entry
(honest reporting is a repo-wide rule, not optional for bad news).

## 8. Concrete next actions

1. Resolve Stage 0 (smoke LoRA gate) — check specula's status before
   duplicating spend, or explicitly decide to run habeas's own.
2. Scale up pilot data (Stage 1) — no gate dependency, can start now.
3. Build the teacher-distillation trace pipeline + fix `modal_train.py`'s
   unused-`data` bug (Stage 2) — no gate dependency for the *code*, only
   for the *real training run*.
4. Build `cloud/modal_rlvr.py` for habeas (currently doesn't exist) with
   the reward function reusing `habeas_forge.score` (Stage 3) — code-only,
   no gate dependency.
5. Wire a real `Provider` (local vLLM/MLX or frontier API) into
   `benchmark_eval` for actual head-to-head numbers once a trained
   checkpoint exists.

Items 2–4 are independent of each other and of Stage 0 — a reasonable
fanout for the next work session once scoped further.
