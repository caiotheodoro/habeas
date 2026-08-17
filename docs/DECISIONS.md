# Habeas — Decision Log

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

## 2026-08-17 — P4 Stage 1 — Scale up training corpus (seed 7, n=2000)
- Decision: regenerated `data/pilot.jsonl` at `n=2000` (was 400), re-split
  (train=1597, val=403, overlap=0), re-leakprobed (clean 0/403, leaked
  10/10 planted). `n=2000` chosen as a reasonable middle ground pending
  real per-step GPU timing from the (not-yet-run) smoke LoRA — cheap to
  regenerate at a different `n` once that timing exists.
- Rationale: 301 training tasks (the P0/P1/P2-era pilot size) validates the
  pipeline but is too small for a real SFT run; docs/TRAINING_PLAN.md §Stage
  1 calls for scaling before Stage 2.
- Evidence: `contamination.split_overlap` confirms `data/golden.jsonl`
  (seed 777, unchanged) stays zero-overlap against the new, larger
  train/val — same check used for the original P2 golden-benchmark commit.
- Alternatives rejected: leaving `n` at 400 and scaling later — no
  advantage to deferring a free, local, deterministic regeneration step.

## 2026-08-17 — P3 — Dataset builder + benchmark_eval provider adapter
- Decision: `habeas_model.dataset_builder` reads forge `Task` JSONL and
  emits chat-format SFT records (system = `SYSTEM_PROMPT`, user = form/doc
  facts + base64 rendered page, assistant = oracle-derived `VerdictOut`
  JSON); `habeas_model.benchmark_eval` adds a minimal `Provider` protocol
  (`complete(system, user, image_b64) -> str`), concurrent
  (`ThreadPoolExecutor`) + JSONL-checkpointed eval (`run_eval`, skips
  already-scored `task_id`s on re-run — resumable per CONTRACTS.md §6), and
  scoring via `habeas_forge.score.score_predictions`/`summarize`. Both
  modules import `habeas_forge` via `sys.path` insertion (`model/tests/
  conftest.py`), matching forge's own existing test-time pattern, since
  both packages are `[tool.uv] package = false` (no build backend, so no
  normal path-dependency wiring is possible between them).
- Rationale: HANDOFF.md's "wire a provider adapter into
  benchmark_eval._predict_one" next-action required the module to exist
  first; scaffolding it against the current train/val data now (rather than
  waiting on the smoke-LoRA gate) lets both be validated independently.
  Image rendering during dataset-build/eval is seeded from
  `int(task.signature[:16], 16)` (not a fresh `random.Random()`) so the same
  task always renders identical bytes — required for CONTRACTS.md §6's
  "identical inputs" guarantee across eval runs/models; found and fixed
  during test-writing (a non-seeded render made `_user_content`'s reported
  image length nondeterministic per call).
- Evidence: `model/tests/` — 6 tests green (`test_dataset_builder.py`,
  `test_benchmark_eval.py`, incl. a `run_eval` resumability check and a
  perfect-provider severity-weighted-recall==1.0 sanity check). `uv sync
  --extra dev` run in `model/` (torch/transformers/trl/peft + click/numpy
  added to `model/pyproject.toml`). Independently reviewed via
  `opencode run`.
- Alternatives rejected: persisting rendered images to disk at pilot-
  generation time in forge and loading them in `model/` — deferred; on-
  demand re-render from `task.form` (now deterministic) is simpler and
  avoids a forge/CLI change outside this workstream's scope, at the cost of
  the model/ render not being byte-identical to whatever image forge's own
  `image_form_sha256` was computed from (acceptable: same form content,
  different but reproducible noise draw).

## 2026-08-17 — P2 — Golden benchmark (seed 777) + generator entropy fix
- Decision: generated the golden benchmark via
  `habeas_forge.cli pilot --seed 777 --n 1000 --out data/golden.jsonl`
  (1000 tasks, 728 FLAG). Fixed a generator bug discovered during zero-
  overlap verification: `generate_packet()`'s `COMBINATION_INVALID`
  injection fallback (when stripping documents leaves the list empty) used
  a hardcoded document (`number="D1"`, fixed expiration) instead of a
  randomized number — a zero-entropy fallback that, combined with the
  generator's otherwise-small "valid packet" field space (64 name
  combinations × 2 editions × 2 categories × fixed dates), produced
  byte-identical `FormI9` signatures across independent seeds. Confirmed:
  golden (seed 777) vs train/val (seed 7) had exactly 1 overlap each before
  the fix, both traced to this fallback path; 0/0 after randomizing the
  fallback document number.
- Rationale: CONTRACTS.md §4 requires the benchmark be zero-overlap with
  train/val; any zero-entropy code path in the generator is a latent
  contamination-monitor blind spot regardless of dataset size.
- Evidence: `contamination.split_overlap(train, golden)` /
  `split_overlap(val, golden)` both 0 after the fix (were 1/1 before);
  `make validate` still 8/8 green; independently reviewed via `opencode run`.
- Alternatives rejected: re-seeding golden generation until no collision
  appears (masks the bug rather than fixing it; a real deployment could hit
  the same fallback and produce a genuinely duplicate task).

## 2026-08-17 — P1 — Citation verification: 8 CFR 274a.2 / M-274
- Decision: verified all 9 pre-existing `verify.py` citation strings against
  authoritative source text; corrected 4. Final citations: EDITION_WRONG →
  M-274 Ch.1 / 8 CFR 274a.2(a)(2) [corrected: was (a)]; FIELD_INCOMPLETE →
  8 CFR 274a.2(b)(1)(i)(A) [corrected: was (b)(1)(i)]; DATA_INCONSISTENT →
  M-274 Ch.4 (Completing Section 2) / Ch.9 (Correcting Errors) [chapter
  numbers from search excerpts, not a full PDF read — see HANDOFF.md
  follow-up]; COMBINATION_INVALID → 8 CFR 274a.2(b)(1)(v) [confirmed
  unchanged]; DOC_INVALID → M-274 Ch.4 / 8 CFR 274a.2(b)(1)(v)(A)-(C)
  [refined: was (v) alone]; DOC_EXPIRED → 8 CFR 274a.2(b)(1)(v) [corrected:
  was (v)(A), which is actually the List A document enumeration, not the
  expiration clause]; TIMELINESS → 8 CFR 274a.2(b)(1)(ii) [confirmed
  unchanged]; REVERIFICATION → 8 CFR 274a.2(b)(1)(vii) [confirmed unchanged,
  text matches verbatim]; CATEGORY_MISMATCH → 8 CFR 274a.2(b)(3)
  [corrected: was (b)(1)(i)(A), which is the Section-1-completion clause,
  not the attestation-under-penalty-of-perjury clause].
- Rationale: oracle citations must trace to real authority, not the
  placeholder guesses made during P0 scaffold; CONTRACTS.md §3 requires the
  oracle to cite M-274/8 CFR precisely.
- Evidence: govinfo.gov CFR-2025-title8-vol1 §274a.2 full text (PDF pages
  808-813, fetched 2026-08-17) for all 8 CFR subsections; WebSearch excerpts
  of the current M-274 Handbook table of contents / "9.0 Correcting Errors
  or Missing Information on Form I-9" for the two M-274 chapter cites (not a
  full handbook PDF read — flagged as a P1-remainder follow-up in
  HANDOFF.md).
- Alternatives rejected: keeping placeholder citations (violates
  "verifier-as-oracle must be traceable").

## 2026-08-17 — P1 — Remote-examination (E-Verify) branch
- Decision: model `FormI9.remote_examination` / `everify_enrolled` /
  `remote_copies_retained`; add `ViolationType.REMOTE_EXAM_INVALID` (HIGH,
  weight 1.0) firing when remote exam is used without E-Verify enrollment or
  without retained document copies, cited to 8 CFR 274a.2(b)(1)(ix) (the DHS
  alternative-documentation-examination-procedure clause).
- Rationale: 8 CFR 274a.2(b)(1)(ix) is a distinct compliance branch not
  covered by the existing physical-exam rules; reusing an existing type
  would conflate two different failure classes. Both sub-failures
  (no E-Verify enrollment, copies not retained) reuse one type since they're
  the same compliance class ("remote exam performed out of compliance").
- Evidence: `forge/tests/test_forge.py::test_remote_exam_gate` +
  `test_every_violation_type_reachable` (8 tests green); 400-task pilot
  shows 82 remote-exam packets, 36 REMOTE_EXAM_INVALID occurrences,
  stratified split overlap 0.
- Alternatives rejected: reusing DOC_INVALID/FIELD_INCOMPLETE for remote-exam
  failures (would conflate document-classification/completeness failures
  with a procedural-alternative failure, weakening the taxonomy).

## 2026-08-17 — P1 — OCR-noise augmentation
- Decision: `render_form()` applies gaussian sensor noise, blur, slight
  rotation, and JPEG re-quantization scaled by `ocr_noise_level` (derived
  from `task()`'s `difficulty` param); recorded on `Task.ocr_noise_level`.
  Noise is strictly a rendering-layer perturbation applied only after
  `oracle_gate` has already passed on the clean `FormI9` — it never touches
  ground-truth fields, `signature()`, or `verify()`.
- Rationale: CONTRACTS.md §4 names OCR noise as a difficulty axis; the
  self-check gate (`oracle_gate`) must stay a pure function of `FormI9`, so
  noise had to be isolated to the image-rendering seam that already existed
  between `form` and `image_form_sha256`.
- Evidence: `forge/tests/test_forge.py::test_ocr_noise_does_not_affect_oracle`
  (same seed, noise 0.0 vs 1.0 → identical form/signature/expected),
  `test_ocr_noise_produces_valid_image`; `test_deterministic_same_seed`
  still green unmodified.
- Alternatives rejected: perturbing `FormI9` field values (e.g. character
  substitutions in `name_section1`) — would require the oracle to model OCR
  error tolerance, turning a rendering artifact into a ground-truth
  ambiguity; rejected as out of scope for this increment.
