# Attest

**Fine-tune a multimodal Qwen3.8-27B (28B, Apache-2.0) on Modal/GCP free
credits so it beats frontier models at Form I-9 compliance validation.**

Form I-9 + presented identity/employment-authorization documents (scans) go
in → an audit-ready report comes out: every field problem, document
acceptability issue, timeliness violation, and reverification gap, each cited
to the M-274 Handbook / 8 CFR 274a.2.

Fourth of four parallel repos (`plumb`, `seam`, `tally`,
`attest`) on one shared methodology (`docs/methodology.md`). Independent and
operable in parallel.

## Why this niche

- **Static federal rules, zero drift.** One jurisdiction, one form, two
  editions (2023-08-01, 2025-01-20 — the latter postdates most frontier
  training), one static rule set (M-274 + 8 CFR 274a.2). Every check is
  deterministic given the inputs — the oracle is a rules engine.
- **Vision-native:** the model reads actual presented documents (passports,
  EADs, DLs, SSN cards) — the exact regime where the multimodal 27B wins.
- **Real money at stake:** statutory fines per form (~$288–2,861 paperwork,
  up to ~$28,619 substantive per 8 CFR 274a.10, 2025-adjusted); ICE audit
  waves; 3-business-day deadline automation alone sells.
- **Competition (research-verified):** LawLogix / I-9 Advantage / Tracker are
  workflow SaaS, not LLM-native validation with citation-grade output.

## The process (end-to-end, fully automatable)

1. Ingest employee Section 1 + Section 2 + Supplement A/B data, plus scans of
   presented documents.
2. Validate every field; classify documents against the fixed List A/B/C
   acceptability table; check category-code/expiry/attestation consistency.
3. Enforce timeliness (Section 2 ≤ 3 business days from hire) and
   reverification (Supplement B when work authorization expires); apply the
   remote-examination (E-Verify) branch.
4. Emit audit report JSON: `PASS | FLAG` + `{type, severity, field, observed,
   expected, cfr, correction}`. Human signs off.

## Benchmark contracts

See `CONTRACTS.md` — fixed violation taxonomy + severity weights, scoring,
contamination, benchmark rules.

## Directory layout

```
forge/      seeded I-9/document generator + M-274/8 CFR oracle + contamination
            monitor + golden benchmark (pkg attest_forge)
model/      multimodal dataset builder, QLoRA SFT, GRPO RLVR, benchmark eval
cloud/      Modal app + Dockerfile + GCP spot scripts
eval/       deterministic golden harness
docs/       DECISIONS.md, BENCHMARK.md, HANDOFF.md, methodology.md
```

## Training stack & cloud

Shared stack (`docs/methodology.md`): QLoRA 4-bit SFT → GRPO/Dr.GRPO
(DAPO-style) RLVR against the I-9 oracle → ReST-EM → s1 curation → benchmark.
Modal `dev-caiotheodoro` primary, GCP fallback. Smoke LoRA first (DeltaNet
gate).

## Targets (golden benchmark, held-out seed, zero contamination)

| Metric | Target |
|---|---|
| Oracle agreement on adversarial suite | **100%** |
| Severity-weighted violation recall | **> 0.95** |
| Citation (M-274 / 8 CFR) exact-match | **> 95%** |
| Timeliness / reverification flags | **100%** |
| Parse rate | 100% |

Head-to-head vs Qwen3.8-2.4T-A95B, DeepSeek v4-flash, base Qwen3.8-27B.

## Phases

P0 scaffold (current) → P1 rules engine + hand-verified cases → P2
generator + golden set → P3 dataset + SFT → P4 RLVR → P5 self-play/judges →
P6 head-to-head + writeup.

## Risks (honest)

- **Heavy PII** — synthetic-only training data, no real forms ever;
  encryption/access controls; validator positioning with human sign-off.
- **Tamper / physical-document fraud** out of scope (validator, not
  authenticator).
- DHS rulemaking churn is a tailwind (remote-examination procedure).