# Habeas — Benchmark Report

_Template. Full head-to-head populated after P6 (frontier comparison)._

Interim SFT-only checkpoint result, 150/1000-task golden subsample
(2026-08-20, see docs/DECISIONS.md "Eval pipeline + first SFT eval
numbers" for methodology/caveats — not a random sample, not the full
golden set):

| Model | Severity-w. recall | Verdict accuracy | FP/task | Parse |
|---|---|---|---|---|
| **Habeas SFT-only (Qwen3.8-27B QLoRA, pre-RLVR)** | 0.615 | 0.927 | 0.313 | 0.993 |
| **Habeas (Qwen3.8-27B QLoRA + RLVR)** | — | — | — | — |
| Qwen3.8-2.4T-A95B (frontier) | — | — | — | — |
| DeepSeek v4-flash (frontier) | — | — | — | — |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — |

Run book: `cd forge && uv run python -m habeas_forge.cli pilot --seed 777 --n 1000 --out data/benchmark.jsonl` then `model` bench eval. Contamination: train/bench overlap = 0; leak probe ROC green.
