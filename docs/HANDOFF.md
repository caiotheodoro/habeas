# Habeas — Handoff (fresh-agent bootstrap)

**What this is:** a fine-tune of Qwen3.8-27B (28B multimodal, Apache-2.0) that
validates Form I-9 + presented documents → audit report citing M-274
Handbook / 8 CFR 274a.2. Synthetic-only (heavy PII — never use real forms).
Specs: `README.md` (atomic spec), `CONTRACTS.md` (fixed benchmark contracts),
`docs/methodology.md` (shared recipe across specula, suture, plumb, habeas),
`docs/TRAINING_PLAN.md` (P4 SFT/RLVR/self-play/judges plan, design-only —
no cloud jobs launched from it yet).

## Repo map

```
forge/    habeas_forge: seeded I-9 packet generator + M-274/8 CFR rules
          engine (verifier-as-oracle) + contamination monitor + CLI
model/    habeas_model: dataset_builder.py (forge Task -> SFT chat-format
          JSONL, base64 rendered page per task), benchmark_eval.py
          (Provider adapter protocol + concurrent/checkpointed eval +
          forge-oracle scoring), schema.py = model-output contract +
          parser + SYSTEM_PROMPT
cloud/    Modal app (L4 24GB, QLoRA) + Dockerfile + GCP spot script
eval/     deterministic golden harness notes
docs/     DECISIONS.md, BENCHMARK.md (report template), HANDOFF.md,
          methodology.md
```

## Status (2026-08-17)

- **P0 scaffold**: complete — README, CONTRACTS, forge (schema, rules-engine
  oracle, generator, score, contamination, cli), model, cloud, docs.
- **P1 oracle**: complete — citations verified against eCFR/govinfo 8 CFR
  274a.2 (CFR-2025-title8-vol1); remote-examination (E-Verify) branch live
  (`REMOTE_EXAM_INVALID`, 8 CFR 274a.2(b)(1)(ix)); OCR-noise augmentation
  live (rendering-layer only, `Task.ocr_noise_level`). See DECISIONS.md.
- **Verified green**: `make validate` (8 tests; every violation type
  reachable, including REMOTE_EXAM_INVALID); 400-task pilot (287 FLAG, 82
  remote-exam packets), split overlap 0, contamination leak-probe ROC clean.
- **P2 golden benchmark**: complete — `data/golden.jsonl` (seed 777, 1000
  tasks, 728 FLAG), zero overlap vs train/val confirmed. Fixed a generator
  entropy bug found in the process (see DECISIONS.md).
- **P3 dataset builder**: scaffolding complete —
  `habeas_model.dataset_builder.build_dataset()` (also a small CLI: `python
  -m habeas_model.dataset_builder build --tasks-file ... --out ...`) turns
  forge Task JSONL into chat-format SFT records (system/user/assistant +
  base64 rendered page); image rendering is seeded from the task signature
  for reproducibility. `habeas_model.benchmark_eval` adds a minimal
  `Provider` adapter protocol, concurrent + JSONL-checkpointed (resumable)
  eval, and scoring via `habeas_forge.score`. `model/tests/` added (6
  tests green) — needed `uv sync --extra dev` in `model/` (heavy deps:
  torch/transformers/trl/peft) and `click`+`numpy` added to
  `model/pyproject.toml`. **Smoke LoRA on Modal not run** — deferred,
  shared gate, specula is the lead repo for that.
- **P4 Stage 0 (smoke-LoRA gate): CLEARED ✓ (2026-08-18)** — a real SFT
  smoke run completed end-to-end on a GCP on-demand L4
  (`habeas-train-0818-0342`, us-east1-b, project
  `project-ddef13eb-b20f-47e0-af0`): 5/5 training steps, checkpoint
  written (`adapter_model.safetensors`, 499MB, verified on disk), clean
  exit. Took 10 fixed bugs across 7 live iterations to get there — see
  `docs/DECISIONS.md`'s "SMOKE-LORA GATE CLEARED" entry for the full list
  and each bug's own entry for detail. Instance torn down after
  confirmation (on-demand billing). **Modal was abandoned for this attempt**
  — its local client proved unreliable for long-lived connections in this
  session (unrelated to the training code); `cloud/modal_train.py`/
  `modal_rlvr.py` carry the same code fixes but haven't themselves been
  live-verified on Modal.
- **P4 Stages 1-3 (real SFT/RLVR)**: code built and unit-tested (see P3
  entry above), but **not live-verified at real scale** — the smoke run
  used 8 synthetic tasks and 224×224 downscaled images specifically to fit
  an L4; real training's `max_length=4096` / full-resolution images / full
  ~1600-task corpus GPU-memory footprint has not itself been tested. Do
  not assume it "just works" by extrapolation from the smoke pass.

## Next actions

1. **P1 remainder**: M-274 Handbook chapter numbers (as opposed to 8 CFR
   subsections, all verified) were cross-referenced via search excerpts, not
   a full chapter-by-chapter PDF read — worth a follow-up pass against the
   current M-274 PDF directly if a discrepancy surfaces in practice.
2. **P3 remainder**: wire a real provider (local vLLM/MLX or frontier API)
   behind the `Provider` protocol in `benchmark_eval.py` for actual
   head-to-head runs (current tests use an in-memory test double).
3. **P4 real SFT run**: now unblocked (gate cleared) — needs its own GPU-
   sizing decision (L4 may or may not be sufficient at full `max_length`/
   resolution even with liger-kernel; verify before assuming) and, per
   methodology.md, teacher-distilled verifier-filtered traces rather than
   the smoke run's raw oracle-target data.

## Bootstrap (fresh agent)

```sh
cd ~/Documents/personal/habeas
make sync && make validate
cd forge && uv run python -m habeas_forge.cli pilot --seed 7 --n 400 --out data/pilot.jsonl
cd forge && uv run python -m habeas_forge.cli split --pilot data/pilot.jsonl --out-train data/train.jsonl --out-val data/val.jsonl
cd forge && uv run python -m habeas_forge.cli leakprobe --train data/train.jsonl --eval-file data/val.jsonl
```

## Environment (already configured)

- **Modal**: profile `dev-caiotheodoro` (token verified 2026-08-17, $30/mo free
  credit, L4 24GB). Volume `habeas-checkpoints`.
- **gcloud**: active account `dev.caiotheodoro@gmail.com`, project
  `cambio-curitiba-498923`, Compute API enabled. **GPU quota is actually 0
  globally** (`GPUS_ALL_REGIONS` limit 0.0, confirmed 2026-08-17 via
  `gcloud compute project-info describe` — no GPU quota line present at
  all) — the "GPU quota 1000" claimed here previously was wrong/stale.
  `cloud/gcp_spot.sh` cannot actually launch an L4 instance until a quota
  increase is requested and approved (manual GCP process, not scriptable).
- uv + Python ≥3.11; macOS: `make sync` runs `chflags -R nohidden .venv`.

## Parallel repos

specula (food-label compliance — lead), suture (policy-issuance QC), plumb
(AIA pay-app review). Same scaffold/methodology; independently operable.
