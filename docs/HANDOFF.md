# Habeas — Handoff (fresh-agent bootstrap)

**What this is:** a fine-tune of Qwen3.8-27B (28B multimodal, Apache-2.0) that
validates Form I-9 + presented documents → audit report citing M-274
Handbook / 8 CFR 274a.2. Synthetic-only (heavy PII — never use real forms).
Specs: `README.md` (atomic spec), `CONTRACTS.md` (fixed benchmark contracts),
`docs/methodology.md` (shared recipe across specula, suture, plumb, habeas).

## Repo map

```
forge/    habeas_forge: seeded I-9 packet generator + M-274/8 CFR rules
          engine (verifier-as-oracle) + contamination monitor + CLI
model/    habeas_model: dataset builder (stub), benchmark eval (stub),
          schema.py = model-output contract + parser + SYSTEM_PROMPT
cloud/    Modal app (L4 24GB, QLoRA) + Dockerfile + GCP spot script
eval/     deterministic golden harness notes
docs/     DECISIONS.md, BENCHMARK.md (report template), HANDOFF.md,
          methodology.md
```

## Status (2026-08-17)

- **P0 scaffold**: complete — README, CONTRACTS, forge (schema, rules-engine
  oracle, generator, score, contamination, cli), model, cloud, docs.
- **Verified green**: `make validate` (5 tests; every violation type
  reachable); 400-task pilot (298 FLAG), split overlap 0, contamination
  leak-probe ROC clean.
- **P1+ not started**: citation verification, remote-examination branch,
  golden benchmark, dataset builder, SFT, RLVR, head-to-head.

## Next actions

1. **P1**: verify exact citations and both I-9 editions (2023-08-01,
   2025-01-20) against M-274 / eCFR; add the remote-examination (E-Verify)
   branch; OCR-noise augmentation.
2. **P2**: build golden benchmark (seed 777) via the CLI.
3. **P3**: dataset builder + **smoke LoRA on Modal** to validate Qwen3.8-27B
   DeltaNet fine-tuning tooling (shared gate; specula is the lead).
4. Wire a provider adapter into `benchmark_eval._predict_one` for head-to-head.

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
  `cambio-curitiba-498923`, Compute API enabled, GPU quota 1000 (spot L4
  fallback).
- uv + Python ≥3.11; macOS: `make sync` runs `chflags -R nohidden .venv`.

## Parallel repos

specula (food-label compliance — lead), suture (policy-issuance QC), plumb
(AIA pay-app review). Same scaffold/methodology; independently operable.
