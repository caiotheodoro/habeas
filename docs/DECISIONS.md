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

## 2026-08-17 — P4 — Modal SFT/RLVR wiring + shared train_cli
- Decision: `cloud/modal_train.py`'s `train()` previously ignored its own
  `data: bytes` param and trained on a hardcoded empty dataset. Fixed by
  extracting the actual training logic into a new, plain,
  filesystem-path-based `habeas_model.train_cli.run_sft(data_path, out_dir,
  smoke, epochs)` — shared by both `modal_train.py` (bytes → temp file →
  `run_sft` → `vol.commit()`) and a new `train_cli.py` CLI entrypoint used
  by the GCP spot fallback, so the two cloud paths can't silently diverge
  (docs/TRAINING_PLAN.md §1's one-code-path reproducibility requirement).
  `run_sft` decodes each SFT record's `image_b64` into a `PIL.Image` under
  an `images` column (required by TRL's multimodal `SFTTrainer` collator —
  not optional) and passes `processing_class=AutoProcessor(...)`, not just
  a tokenizer. Smoke mode caps both the data slice (8 records) and step
  count (`max_steps=5`) — belt-and-suspenders.
- New `cloud/modal_rlvr.py` (didn't exist in habeas before — specula's own
  copy is still a stub): GRPO RLVR against `habeas_model.rlvr_reward.
  oracle_reward_func`, which wraps `habeas_model.schema.to_forge_verdict` +
  `habeas_forge.score.reward` into TRL's `reward_funcs(completions,
  **kwargs)` signature. Uses `GRPOConfig(loss_type="dr_grpo", epsilon=0.2,
  epsilon_high=1.0)` — Dr. GRPO and DAPO's decoupled clip are both native
  TRL options (verified against installed `trl==0.24.0`), no custom
  subclass needed. DAPO's dynamic sampling (drop/resample zero-reward-
  variance prompt groups) is **not** natively supported by TRL and is
  deliberately not implemented in v1 — ships without it, monitoring
  TRL's own `frac_reward_zero_std` metric instead; a custom `GRPOTrainer`
  subclass is a documented follow-up only if that metric shows it matters
  empirically. Loads the SFT adapter via `PeftModel.from_pretrained(base,
  base_adapter, is_trainable=True)`, never the raw base model.
- Fixed `cloud/Dockerfile`'s `CMD` (referenced a nonexistent
  `habeas_model.train` module) to `habeas_model.train_cli`, added `ENV
  PYTHONPATH` for the two `package = false` projects. Fixed
  `cloud/gcp_spot.sh`'s startup script (same dangling-module bug) to run
  `dataset_builder build` then `train_cli.py`; also fixed a separate real
  gap found while doing this — `data/` is gitignored, so a fresh clone on
  a GCE box has no pilot/train/val files at all — added the missing
  `pilot`/`split` regeneration step (seed 7, matching the Stage 1 entry
  above) before dataset building.
- Also fixed (unrelated pre-existing issue, found incidentally while
  testing these files import cleanly): `modal.gpu.L4(count=1)` is
  deprecated against the installed Modal SDK (1.2.6) — `DeprecationError:
  use gpu="L4" instead`. Updated both `modal_train.py` and the new
  `modal_rlvr.py` to the current `gpu="L4"` string form.
- Rationale: one training code path (not two), correct multimodal data
  wiring (the `images` column gap would have silently crashed or trained
  on garbage at real-run time), and RLVR built on native TRL features
  rather than reinvented ones.
- Evidence: `model/tests/test_train_cli.py` (`_load_sft_records` decode +
  smoke-cap tests, isolated from the actual untestable-without-GPU
  `run_sft` training call), `model/tests/test_rlvr_reward.py`
  (`oracle_reward_func` batch-alignment + unparseable-completion tests) —
  17/17 model tests green total. Both `cloud/modal_train.py` and
  `cloud/modal_rlvr.py` verified to import cleanly (no errors, no
  deprecation warnings) with the locally-installed Modal SDK; `bash -n`
  clean on `gcp_spot.sh`. Independently reviewed via `opencode run` —
  traced the gcp_spot.sh path-consistency (cd forge/cd .. sequencing) and
  the TRL reward_funcs kwarg-passing convention, found no bugs in what it
  examined before the review process ended without a final written verdict
  (a recurring flakiness this session, not specific to this diff).
- Flagged, not verified: real multimodal image-column format against the
  actual Qwen3.8-27B processor (no GPU/model weights available locally —
  same flag as the original P4 plan); whether TRL 0.24.0's `GRPOTrainer`
  genuinely supports multimodal `images` columns end-to-end (its GRPO
  implementation has historically been text-first) — a smoke-run-time risk
  to watch, not resolvable by static review.

## 2026-08-17 — P4 — dataset_builder teacher-trace source + RLVR prompts
- Decision: `build_record(task, target_source="oracle"|"teacher", teacher=
  None)` — default `"oracle"` unchanged; `"teacher"` calls a `Provider`
  (moved from `benchmark_eval.py` into `dataset_builder.py`, both now
  reuse one definition — `benchmark_eval` imports it back), parses via
  `habeas_model.schema.to_forge_verdict` (promoted from `benchmark_eval.
  _to_forge_verdict`, kept as a back-compat alias there), and
  verifier-filters per docs/methodology.md's "SFT cold-start on
  verifier-filtered traces distilled from a stronger model." `build_dataset`
  threads the same params and skips filtered-out (`None`) records.
  Separately, `build_rlvr_prompt`/`build_rlvr_prompts` (+ a `prompts` CLI
  subcommand) produce a **structurally distinct** RLVR-prompt JSONL (no
  assistant turn at all) from `train.jsonl` `Task`s, so RLVR data cannot be
  pointed at the SFT trace file by construction (methodology.md's "RLVR
  data never mixed into SFT").
- **Bug found and fixed during independent review** (`opencode run`): the
  first verifier-filter implementation checked exact-match via
  `habeas_forge.score.score_predictions`'s `caught`/`total`/`fp` — but
  those are severity-weighted sums over a *set* of (type, severity) pairs,
  so a task with two violations sharing the same (type, severity) (e.g.
  two `DOC_EXPIRED`) would pass the "exact match" check even if the
  teacher's trace reported only one of them. Fixed by comparing
  `collections.Counter((type, severity) for v in violations)` — a true
  multiset comparison — directly in `build_record`, independent of
  `score_predictions`. Confirmed via a live demonstration
  (`score_predictions` returns `caught==total==1.0, fp==0` for a 2-vs-1
  duplicate-violation pair) and empirically confirmed **currently
  unreachable via the generator** (`_valid_packet` never produces more
  than one expirable List A/B document, and the `REMOTE_EXAM_INVALID`
  injection branch always sets exactly one of its two sub-conditions bad,
  never both) — fixed anyway since it's a latent correctness gap that
  self-play or a generator change could make reachable later, and the fix
  was cheap and file-local.
- Rationale: a "verifier-filtered trace" claim needs to actually be exact;
  set-based aggregate scoring (fine for `score_predictions`'s intended use
  — recall/precision reporting at the benchmark level) is the wrong tool
  for a hard pass/fail filter gating what enters the SFT dataset.
- Evidence: `model/tests/test_dataset_builder.py::
  test_build_record_teacher_source_duplicate_violation_undercount_filtered`
  (regression test, hand-constructs a duplicate-violation task) + all other
  `dataset_builder`/`rlvr_reward`/`train_cli` tests — 17/17 green.
- Alternatives rejected: changing `score_predictions` itself to multiset
  semantics — would be a second CONTRACTS.md-scoped scoring change in one
  session (after the verdict_correct fix) with wider blast radius across
  every consumer of severity-weighted recall; the exact-match need is
  local to this one filter, so fixing it there is narrower and safer.

## 2026-08-18 — P4 Stage 0 — Smoke-LoRA gate: torchvision + SFTConfig API bugs
- Decision: fixed two more real bugs found by re-running the smoke test
  after the quantization_config fix, each one progressing further before
  failing on the next: (1) `AutoProcessor.from_pretrained("Qwen/
  Qwen3.8-27B")` raises `ImportError: Qwen3VLVideoProcessor requires the
  Torchvision library` — the multimodal processor pulls in a video
  sub-processor unconditionally even for image-only use; added
  `torchvision` to `cloud/Dockerfile`, `cloud/gcp_spot.sh`, and
  `model/pyproject.toml` (a sibling repo, specula, hit the identical gap
  independently — cross-repo corroboration this isn't environment-specific
  noise). (2) `SFTConfig(max_seq_length=4096)` raises `TypeError:
  unexpected keyword argument 'max_seq_length'` — renamed to `max_length`
  in the installed trl (confirmed via `inspect.signature` against the
  actual installed package); fixed in `train_cli.py`.
- Rationale: same as the quantization_config entry above — these are
  exactly the class of bug the smoke gate exists to surface before a real,
  costly training run hits them instead.
- Evidence: both found via the same live GCP smoke run
  (`habeas-train-0818-0236`, us-east1-b), redeployed via `git pull` + a
  `systemd-run` transient unit on the running instance (faster than
  recreating — reuses the already-downloaded pip packages and partial HF
  model cache) rather than tearing down/recreating each time. `model`
  tests still 17/17 green (neither change is exercised by local unit tests
  — both are live-execution-only paths).

## 2026-08-18 — P4 Stage 0 — Smoke-LoRA gate: quantization API bug found + fixed
- Decision: fixed `train_cli.run_sft` and `modal_rlvr.rlvr`'s model loading
  — both passed a bare `load_in_4bit=True` kwarg to
  `AutoModelForMultimodalLM.from_pretrained`, which the installed
  `transformers` version rejects outright: `TypeError:
  Qwen3_5ForConditionalGeneration.__init__() got an unexpected keyword
  argument 'load_in_4bit'`. Fixed to `quantization_config=
  BitsAndBytesConfig(load_in_4bit=True)`. Also fixed the deprecated
  `torch_dtype=` kwarg to `dtype=` in both places (a warning, not a crash,
  but flagged during the same run). Added `bitsandbytes` to
  `model/pyproject.toml`'s dependencies (was only ever installed via the
  Dockerfile/gcp_spot.sh pip lines, not declared for local dev).
- Rationale: this is exactly what the smoke-LoRA gate exists to catch — a
  static-analysis/code-review pass couldn't have found this (the earlier
  Plan sub-agent's design correctly flagged 4-bit as required for VRAM, but
  had no way to verify the exact `from_pretrained` kwarg contract against
  the real installed transformers version without executing it on real
  hardware).
- Evidence: caught via an actual GCP L4 on-demand smoke run (instance
  `habeas-train-0818-0236`, us-east1-b, project
  `project-ddef13eb-b20f-47e0-af0` — Modal was abandoned for this attempt
  after repeated local client connection instability, unrelated to the
  training code itself; see the same date's Modal-vs-GCP entry below for
  that context). Full traceback captured via `gcloud compute ssh ...
  --command='tail /root/train.log'`. Fix applied, redeployed to the same
  running instance for re-verification (see follow-up entry once the rerun
  completes).
- Alternatives rejected: none — this is a straightforward API-contract fix,
  not a design decision.

## 2026-08-18 — P4 Stage 0 — Modal local-client instability -> GCP pivot
- Decision: after ~2 hours of Modal `modal run --detach` attempts
  consistently crashing 2-4 minutes into the image build (an asyncio/h2/
  grpclib `AttributeError` on 'H2Connection' object, reproduced across the
  system Python 3.9 install AND a from-scratch clean Python 3.12 venv, both
  authenticated correctly, both showing identical remote-side build
  progress via streamed logs before crashing) — while all short one-shot
  Modal CLI calls (`app list`, `container list`, `profile current`) worked
  reliably throughout — concluded this is local-environment network/OS
  behavior (plausibly background-process socket throttling) specific to
  this automated session, not a Modal service issue or a bug in
  `cloud/modal_train.py`/`modal_rlvr.py`. Pivoted to the GCP spot/on-demand
  fallback (`cloud/gcp_spot.sh`), which runs entirely server-side via a
  startup script — no persistent local streaming connection required,
  sidestepping the failure mode entirely. GCP's own smoke run succeeded
  through pip install, data generation, and model download on the first
  code-complete attempt (the only failure was the `load_in_4bit` bug
  above, a real code issue, not an infra one).
- Also found and fixed along the way: `gcp_spot.sh`'s `--metadata=` flag
  broke on commas in embedded comments (gcloud parses `--metadata` as a
  comma-separated dict) — switched to `--metadata-from-file`, which is also
  the more robust pattern generally; GPU-attached GCP VMs require
  `--maintenance-policy=TERMINATE` even when non-preemptible (no live
  migration support for GPUs) — the script's new `PREEMPTIBLE` toggle
  handles this correctly either way; this project's actual GPU quota was 0
  (not the "1000" HANDOFF.md previously claimed) until manually requested
  via the console — see HANDOFF.md's corrected environment section; the
  first preemptible attempt that did get scheduled was preempted mid-run
  before completing, so the script now defaults to on-demand
  (`PREEMPTIBLE=false`) for reliability on short validation runs, with spot
  available via an env var for later real (longer, costlier) training.
- Evidence: `docs/HANDOFF.md` environment section corrected; `cloud/
  gcp_spot.sh` diff (metadata-from-file, PREEMPTIBLE toggle, SMOKE-scoped
  PILOT_N=20).

## 2026-08-17 — P4 Stage 2 — Verdict-consistency scoring fix + RLVR reward()
- Decision: `score_predictions` (CONTRACTS.md §2's scorer) previously only
  compared violation-instance sets (`caught`/`total`/`fp`) and never
  checked whether `predicted.verdict` (PASS|FLAG) itself was right or even
  self-consistent with the listed violations — a model could score
  perfect severity-weighted recall while emitting a self-contradictory
  `PASS` alongside listed violations (or `FLAG` with none). Added a
  `verdict_correct` field (`predicted.verdict == expected.verdict`) to
  `score_predictions`'s return dict and a `verdict_accuracy` aggregate to
  `summarize` — additive, existing keys/tests unaffected (`make validate`:
  12/12 green, up from 8; new `verdict_correct`/`verdict_accuracy` keys
  don't break `test_scoring`'s narrow key-lookup assertion).
- Also added `reward(expected, predicted, fp_penalty=0.3,
  verdict_bonus=0.2, unparseable_reward=-1.0) -> float` in the same file,
  reusing the (now-fixed) `score_predictions` — the scalar RLVR reward
  docs/TRAINING_PLAN.md §Stage 3 calls for. Branches: unparseable → worst
  score (no partial credit, methodology.md's outcome-verifier-only rule);
  clean-PASS expected (total==0, no recall ratio to compute) → scored
  purely on `verdict_correct` minus fp penalty; general case →
  severity-weighted recall minus fp penalty plus/minus a verdict_bonus.
  Pure function of `(Verdict, Verdict | None)` — no dependency on
  `habeas_model` (wrong direction; raw-text parsing into a `Verdict` stays
  a `habeas_model`-layer concern).
- Rationale: this was going to become load-bearing for RLVR reward shaping
  (a reward function built on an incomplete scorer would silently
  under-penalize verdict-contradicting outputs), and the fix benefits SFT
  eval and benchmark eval identically since they share the same scorer —
  closing it once here beats three separate patches later.
- Evidence: `forge/tests/test_forge.py::test_scoring_verdict_consistency`,
  `test_reward_general_case`, `test_reward_clean_pass_case`,
  `test_reward_unparseable` — all green.
- Alternatives rejected: patching only inside the new `reward()` function
  and leaving `score_predictions`/`summarize` untouched — would leave
  benchmark/SFT eval reporting the same blind spot RLVR was fixed for,
  and duplicate the verdict-comparison logic instead of sharing it.

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
