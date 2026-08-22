# Habeas — Benchmark Report

_Template. Full head-to-head populated after P6 (frontier comparison)._

Interim SFT-only checkpoint result, **full 1000-task golden set**
(2026-08-22, see docs/DECISIONS.md "Citation exact-match metric +
definitive full-golden eval" — supersedes the earlier 150-task
subsample figure):

| Model | Severity-w. recall | Verdict accuracy | Citation exact | FP/task | Parse |
|---|---|---|---|---|---|
| **Habeas SFT-only (Qwen3.8-27B QLoRA, pre-RLVR)** | 0.614 | 0.931 | 0.915 | 0.251 | 1.000 |
| **Habeas (teacher-distilled SFT, 399/1597 records)** | 0.398 | 0.780 | — | 0.740 | 0.960 |
| **Habeas (Qwen3.8-27B QLoRA + RLVR)** | — | — | — | — | — |
| Qwen3.8-2.4T-A95B (frontier) | — | — | — | — | — |
| DeepSeek v4-flash (frontier) | — | — | — | — | — |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — | — |

Teacher-distilled row is a **negative result, confounded** (~4x less
data/fewer training steps than the oracle-only row, not a clean
technique comparison, and only measured on a 150-task subsample — see
docs/DECISIONS.md's "Teacher distillation attempted" entry). Not
promoted; oracle-only SFT remains the current best artifact.

Run book: `cd forge && uv run python -m habeas_forge.cli pilot --seed 777 --n 1000 --out data/benchmark.jsonl` then `model` bench eval. Contamination: train/bench overlap = 0; leak probe ROC green.
