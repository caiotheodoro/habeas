# I9forge — Benchmark & Scoring Contracts (FIXED)

Changes require a DECISIONS.md entry with evidence. Generator, verifier,
scorer, contamination monitor, and eval all conform.

## 1. Task definition

One task = a Form I-9 (Sections 1, 2, 3, Supplement A/B) + presented
documents (scans), plus structured ground truth for oracle tracing.

Model emits audit report JSON:

```json
{
  "verdict": "PASS" | "FLAG",
  "violations": [
    {
      "type": "VIOLATION_TYPE",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "field": "SECTION2.DOCUMENT",
      "observed": "List A passport expired 2025-06-01",
      "expected": "unexpired List A document",
      "cfr": "8 CFR 274a.2(b)(1)(v)(A)",
      "correction": "obtain an unexpired List A document"
    }
  ]
}
```

## 2. Violation taxonomy + severity weights

| Class | id | Severity | Weight |
|---|---|---|---|
| No acceptable document combination (needs List A OR List B+List C) | `COMBINATION_INVALID` | HIGH | 1.0 |
| Document not on the acceptable lists / wrong list for type | `DOC_INVALID` | HIGH | 1.0 |
| List A/B document expired | `DOC_EXPIRED` | HIGH | 1.0 |
| Work authorization expired without Supplement B reverification | `REVERIFICATION` | HIGH | 1.0 |
| Section 2 completed after the 3-business-day deadline | `TIMELINESS` | MEDIUM | 0.6 |
| Required Section 1/2 field incomplete | `FIELD_INCOMPLETE` | MEDIUM | 0.6 |
| Wrong form edition | `EDITION_WRONG` | MEDIUM | 0.6 |
| Attestation category inconsistent with documents | `CATEGORY_MISMATCH` | MEDIUM | 0.6 |
| Minor data-entry inconsistency (name/DOB across sections) | `DATA_INCONSISTENT` | LOW | 0.3 |

Severity-weighted recall `R_w = Σ w·caught / Σ w` over violation-bearing
tasks; caught iff type matches (HIGH classes also match severity). Precision =
caught / emitted (unmatched = false positive). Unparseable = parse miss.

## 3. Oracle (verifier-as-oracle)

`i9forge_forge.verify` implements M-274 Handbook + 8 CFR 274a.2:

- **Edition**: Section 1 must use the 2023-08-01 or 2025-01-20 edition.
- **Combination**: exactly one List A document, OR one List B + one List C.
- **Document validity**: passport → List A; EAD/parole → List A; driver's
  license → List B; state ID → List B; SSN card / birth certificate → List C.
  A doc's declared list must match its type (DOC_INVALID).
- **Expiry**: List A and List B documents must be unexpired as of the
  Section 2 date (DOC_EXPIRED).
- **Timeliness**: Section 2 must be completed ≤ 3 business days after hire
  (TIMELINESS).
- **Reverification**: when work authorization expires, Supplement B must
  record a reverification (REVERIFICATION).
- **Category**: the Section 1 attestation category must be consistent with the
  presented documents (CATEGORY_MISMATCH).
- **Completeness**: required Section 1/2 fields present (FIELD_INCOMPLETE);
  cross-section name/DOB consistency (DATA_INCONSISTENT).

Generator builds a valid I-9 packet then injects N violations; self-check
gate requires oracle == injected set; regenerate on mismatch.

## 4. Data & splits

- **Train/val** stratified by (difficulty decile × violation class), seed 7;
  signature-disjoint; build refuses on overlap > 0.
- **Benchmark**: ~1,000 held-out tasks, seed 777, zero overlap.
- **Difficulty** 0–1: near-miss documents (expiring soon, similar doc types),
  OCR noise, subtle data-entry inconsistencies.

## 5. Contamination monitor

Task signature = SHA-256 over sorted (field, value) pairs of the packet's
ground truth (PII placeholders only — fully synthetic). Leak probe fires 1.0
on leaked sets, 0.0 on clean.

## 6. Benchmark rules

Same system prompt for every model; non-thinking verdict output; frontier
scored with identical inputs + scoring code; unparseable = miss; concurrent
eval with JSONL checkpointing.

## 7. Judge calibration (residual prose only)

Pairwise, position-swapped, temp 0, majority ≥3, bootstrap CIs, kappa ≥ 0.85
vs golden-100. Never for core verdicts.