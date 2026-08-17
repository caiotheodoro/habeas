# I9forge — Benchmark Report

_Template. Populated after P6 (head-to-head vs frontier)._

| Model | Severity-w. recall | HIGH recall | Citation acc | Parse |
|---|---|---|---|---|
| **I9forge (Qwen3.8-27B QLoRA + RLVR)** | — | — | — | — |
| Qwen3.8-2.4T-A95B (frontier) | — | — | — | — |
| DeepSeek v4-flash (frontier) | — | — | — | — |
| Base Qwen3.8-27B (zero-shot) | — | — | — | — |

Run book: `cd forge && uv run python -m i9forge_forge.cli pilot --seed 777 --n 1000 --out data/benchmark.jsonl` then `model` bench eval. Contamination: train/bench overlap = 0; leak probe ROC green.
