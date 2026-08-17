# Attest — Decision Log

Append entries as decisions are made. Contract defaults are fixed; revise
only with measured evidence.

## Format

```
## YYYY-MM-DD — <study id> — <title>
- Decision: ...
- Rationale: ...
- Evidence: ...
- Alternatives rejected: ...
```

## 2026-08-17 — P0 — Base model + cloud
- Decision: Qwen3.8-27B (Apache-2.0, VL) QLoRA 4-bit on Modal L4 (primary)
  with GCP cambio-curitiba-498923 spot fallback.
- Rationale: shared stack; document scans are multimodal-native; the 2025-01-20
  I-9 edition postdates most frontier training.
- Evidence: HF card; Modal token verified; GCP Compute API enabled.

## 2026-08-17 — P0 — Oracle: M-274 / 8 CFR 274a.2 rules engine
- Decision: oracle implements edition, combination, document-validity,
  expiry, timeliness (3 business days), reverification, category, and
  consistency rules; generator builds valid packets and injects violations;
  self-check gate requires oracle == injected set.
- Rationale: static federal rules = deterministic oracle; the 3-business-day
  deadline and List A/B/C table are the load-bearing rules.
- Evidence: forge tests (5 green; every violation type reachable).

## 2026-08-17 — P0 — Synthetic-only data
- Decision: no real PII or real forms, ever; packets are procedurally
  generated with placeholder identities; rendering uses no real document
  imagery.
- Rationale: I-9 data is heavy PII/immigration data; synthetic preserves the
  methodology without any privacy surface.
- Evidence: generate.py (placeholder names/numbers).
