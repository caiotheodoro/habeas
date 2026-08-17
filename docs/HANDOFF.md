# Attest — Handoff

## Status (2026-08-17)
- **P0 scaffold**: README, CONTRACTS, forge (schema, M-274/8 CFR oracle,
  generator, score, contamination, cli — 5 tests green, pilot/split/leakprobe
  green), model, cloud, docs.
- **P1 pending**: verify exact citations and both editions against
  M-274/eCFR; remote-examination (E-Verify) branch; OCR-noise augmentation.
- **P2+ pending**: golden benchmark (seed 777), dataset builder, SFT, RLVR,
  head-to-head.

## Next actions
1. P1 citation verification + remote-examination branch.
2. Smoke LoRA on Modal (DeltaNet tooling gate).

## Environment
- Modal profile `dev-caiotheodoro`; gcloud dev.caiotheodoro@gmail.com /
  cambio-curitiba-498923 (Compute API on, GPU quota 1000).
- `make sync` then `make validate`.
