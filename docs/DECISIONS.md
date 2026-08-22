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

## 2026-08-18 — P4 — real full run: batch_size=1 wasted A100 headroom
- Decision: watching the real full run's first 2 steps live (GPU memory
  ~23.8GB/40GB, GPU utilization 36%), noticed `per_device_train_batch_size=1`
  — a default set while validating against the tighter-memory L4 (22GB) —
  was never revisited after moving to the A100 (40GB): ~17GB (41%) sat
  idle, and the low utilization meant the GPU was spending most of its
  time waiting on 4 sequential single-sample passes per step rather than
  computing. Added `batch_size`/`grad_accum` params to `run_sft` (CLI:
  `--batch-size`/`--grad-accum`), exposed via `gcp_spot.sh`'s
  `BATCH_SIZE`/`GRAD_ACCUM` env vars (defaults 1/4, unchanged behavior
  unless overridden). Relaunched the real run with `BATCH_SIZE=2` (keeping
  `GRAD_ACCUM=4`, so effective batch doubles from 4 to 8) — both uses the
  idle headroom directly and roughly halves total optimizer steps for the
  same epoch coverage (more samples processed per step). Estimated
  ~31hr → ~12-15hr.
- Rationale: found by literally watching the run rather than blindly
  trusting the VALIDATE-run-derived defaults carried over unchanged to a
  GPU with nearly double the memory — the same "measure the actual
  hardware, don't extrapolate" discipline that drove the L4→A100 decision
  in the first place, just applied one level deeper.
- Evidence: `nvidia-smi` telemetry from the live run (23.8GB/40GB, 36%
  util) before the change; `model` tests 17/17 green after adding the new
  params. Restarted after only 2 steps (~5 min) — negligible progress
  lost. `--batch-size 3` was considered but not used this round (untested,
  real risk of OOM on a run not being closely babysat) — `--batch-size 2`
  is a confident, well-estimated step from the observed 23.8GB baseline.

## 2026-08-18 — P4 — gcp_spot.sh MODE replaces SMOKE/VALIDATE (bit twice)
- Decision: replaced the `SMOKE`/`VALIDATE` boolean pair with a single
  required `MODE=smoke|validate|real` (`${MODE:?...}` — hard failure with
  a clear message if unset, `case` statement rejects anything else).
- Rationale: the two-boolean design's "SMOKE defaults to true unless
  VALIDATE=true" footgun (logged earlier this same date) was fixed once,
  then bit again in a different shape: launching a *real* run (both
  `SMOKE` and `VALIDATE` meant to be false) without explicitly setting
  `SMOKE=false` silently ran a smoke run instead, because `SMOKE`'s
  default won whenever it wasn't explicitly overridden — the exact same
  underlying defaulting problem, just triggered by omission rather than
  by setting `VALIDATE=true` alone. Two near-misses on the same class of
  bug is a design-not-usage problem: no combination of documentation or
  "remember to set X" discipline reliably prevents a silent wrong-mode
  launch when the interface allows an "obviously wrong" state (both
  unset, defaulting to the cheapest/safest mode) to look identical to
  "I meant this." A single required enum can't have this failure mode —
  you either name a real mode or the script refuses to run.
- Evidence: `MODE` unset → `line N: MODE: Set MODE=smoke|validate|real...`
  (bash's `${VAR:?msg}` parameter expansion, hard exit); `MODE=bogus` →
  explicit rejection message; `MODE=smoke` → parses and runs as before.
  Caught before any GPU time was spent on the wrong config both times
  (deleted the wrongly-configured instance within ~1 min each time).
- Alternatives rejected: adding a third "did you mean it" guard
  (confirmation prompt, `--force` flag) — doesn't fix the root cause
  (silent default winning), just adds friction; the required-enum fix is
  strictly better and no more code.

## 2026-08-18 — P4 — Checkpoint-resume before the real full run
- Decision: before launching a real full SFT run, added
  resume-from-checkpoint support: `train_cli.run_sft` now sets
  `save_strategy="steps", save_steps=20 (default), save_total_limit=3` on
  `SFTConfig`, and checks `transformers.trainer_utils.get_last_checkpoint
  (out_dir)` before calling `trainer.train()` — if a checkpoint already
  exists under `--out`, training resumes from there instead of restarting.
  `cloud/gcp_spot.sh`'s startup script was made idempotent to match: it
  now skips the `git clone` / data-regeneration steps if they already
  exist (a fresh clone would risk resuming against a different code
  version than the run started with; data regeneration is seed-fixed and
  byte-identical, so skipping it is just a restart-speed optimization,
  not a correctness one), and appends to `/root/train.log` instead of
  overwriting it. New `--save-steps` CLI flag exposes the interval.
- Rationale: at the observed ~135s/step, a real full run (full corpus, 2
  epochs) is an estimated ~30 hours, and only *preemptible* A100 quota is
  approved in this project — realistically several preemption-interrupted
  restarts, not one clean run. GCE re-runs the startup-script on every
  boot (including a restart after a preemptible-VM stop cycle), and the
  boot disk (repo, generated data, `/root/checkpoints`) persists across
  that as long as the instance itself isn't deleted — so the recovery
  path is: notice the instance is `TERMINATED`, run `gcloud compute
  instances start <name>`, and the startup-script + `run_sft`'s resume
  logic pick training back up automatically. A full managed-instance-group
  auto-healing setup was considered and deliberately not built — bigger
  scope than warranted for a run this session is actively monitoring
  anyway; manual restart-on-preemption is the accepted fallback, made safe
  by this resume logic rather than by auto-recreation.
- Evidence: `model/tests/` still 17/17 green (the resume-detection guard
  itself — `os.path.isdir` + `get_last_checkpoint` — was sanity-checked
  standalone against both an existing and a missing directory; the actual
  GPU-dependent `trainer.train(resume_from_checkpoint=...)` path can't be
  unit-tested without a GPU, consistent with the rest of `run_sft`).
- Alternatives rejected: syncing checkpoints to a GCS bucket instead of
  relying on local-disk persistence — the classic-preemptible instance
  model already persists the boot disk across a stop cycle (confirmed:
  it's not deleted unless the instance resource itself is deleted), so a
  second, more complex persistence layer wasn't needed for this case;
  would reconsider if the run ever needs to move to a Spot VM with
  `--instance-termination-action=DELETE` instead.

## 2026-08-18 — P4 — REAL CONFIG VALIDATED ON A100 ✓
- Decision: after fixing the boot-disk-size bug (previous entry), reran
  `VALIDATE=true GPU_TYPE=nvidia-tesla-a100 PREEMPTIBLE=true` (only
  preemptible A100 quota was approved) on a fresh `a2-highgpu-1g`
  instance (`habeas-train-0818-1317`, us-central1-a). Full success: all
  5/5 steps completed (`train_runtime: 672.9s`), checkpoint written and
  verified (`adapter_model.safetensors`, 499MB), GPU memory stable at
  ~23.7GB/40GB throughout (comfortable headroom, unlike the L4's
  near-100%-then-OOM). **Real training config (`max_length=4096`,
  full-resolution images) is confirmed to fit an A100 40GB.** This closes
  the caveat both `HANDOFF.md` and `TRAINING_PLAN.md` have carried since
  the Stage 0 smoke gate cleared.
- Rationale: this was the actual question the user's "bigger GPU" decision
  needed answered before committing to a full, multi-hour, multi-task real
  run — now answered with real evidence, not extrapolation.
- Evidence: live `nvidia-smi` telemetry (23.7GB stable across all 5
  steps), `train_runtime`/`train_loss` output, checkpoint file listing on
  the instance. Instance torn down immediately after confirmation.
- Next: a real full SFT run (full ~1600-task corpus, real epochs, no
  `--max-steps`/`--smoke` bound) can now proceed on
  `GPU_TYPE=nvidia-tesla-a100 PREEMPTIBLE=true` — note the preemption risk
  is real for a run this much longer than 11 minutes; consider requesting
  on-demand A100 quota before a very long full run, or accept
  checkpoint-and-resume risk on preemption (not yet built — `run_sft`
  doesn't currently support resuming from a partial checkpoint).

## 2026-08-18 — P4 — A100 VALIDATE attempt: 100GB boot disk was the real culprit
- Decision: launched an A100 40GB (`a2-highgpu-1g`) `VALIDATE=true` run to
  re-check GPU-memory fit after the L4 OOM (see the entry below). The
  model download repeatedly appeared to stall (near-zero network
  throughput for minutes at a time, `CLOSE-WAIT` connections) — initially
  diagnosed as HF anonymous-rate-limiting (repeated "unauthenticated
  requests" warnings) and worked around with a user-supplied `HF_TOKEN`,
  which did seem to help temporarily. The download eventually failed
  outright with `OSError: [Errno 28] No space left on device` —
  `df -h /` showed 93GB/97GB used. **The actual root cause was the boot
  disk, not the network**: the default `--boot-disk-size=100GB` (the DLVM
  image's own stated minimum) left far too little room once ~70GB+ of
  model weights started accumulating during download — the "stalls" were
  very likely disk-nearly-full write blocking, not HF throttling (the
  HF_TOKEN fix may have been solving a real but secondary problem, or
  coincidentally timed with the disk filling further). Fixed by adding
  `GCP_BOOT_DISK_SIZE` (default `300GB`) to `gcp_spot.sh`.
- Rationale: 100GB was sized for a driver+deps DLVM image, never
  re-evaluated once a 70GB+ model entered the picture. A generously-sized
  disk is cheap relative to GPU-hour cost; running out mid-download on an
  A100 wastes far more than the extra disk ever costs.
- Evidence: `df -h /` on the live instance (93G/97G used, 3.9G free);
  exact traceback (`_create_symlink` → `os.symlink` → `ENOSPC`) in the
  instance's systemd journal. Instance torn down, relaunching with the
  disk fix.
- Alternatives rejected: keep diagnosing as a network/HF-token issue —
  would have kept failing at the same disk ceiling regardless of transfer
  speed or auth.

## 2026-08-18 — P4 — VALIDATE run: real config does NOT reliably fit an L4
- Decision: ran `VALIDATE=true` (50 tasks, real `max_length=4096`,
  full-resolution images, `--max-steps 5`) on a fresh on-demand L4
  (`habeas-train-0818-1006`, us-central1-a). Step 1/5 completed
  successfully (21.5GB VRAM used, 151s), but step 2 crashed:
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 340.00
  MiB. GPU 0 has a total capacity of 22.03 GiB` — a *small* allocation
  request, meaning the run was sitting right at the ceiling with
  essentially zero headroom, and the very next micro-batch (different
  tasks → different real token counts within the 4096 cap) tipped it over.
  **Conclusion: an L4 does not reliably fit the real training config, even
  with `use_liger_kernel=True`.** This is exactly the finding the
  `VALIDATE` mode exists to surface cheaply (~15 min, one small instance)
  instead of discovering it hours into a full real run.
- Rationale: confirms the caveat already flagged in `docs/TRAINING_PLAN.md`
  and `HANDOFF.md` — the smoke pass (224×224 images, max_length=1024)
  validated the *tooling*, not real-scale GPU-memory fit, and the two are
  genuinely different questions with different answers here.
- Evidence: live `nvidia-smi`/journal output from the run; instance torn
  down immediately after the OOM was confirmed (no reason to keep paying
  for a dead process).
- **Decision needed from the user, not made unilaterally here**: real
  training needs either (a) a bigger GPU (A100 40GB+ — another manual GCP
  quota-increase round-trip, higher $/hr), or (b) a reduced real config
  that reliably fits an L4 (e.g. `max_length=2048` instead of 4096, and/or
  moderate image downscale — not smoke's 224×224, but something like
  448×448 — trading some context/fidelity for reliability on cheaper
  hardware). Both are legitimate; this is a cost/quality tradeoff call for
  the user, not an engineering-only decision.

## 2026-08-18 — P4 — gcp_spot.sh VALIDATE mode + a SMOKE/VALIDATE footgun
- Decision: added a `VALIDATE=true` mode to `gcp_spot.sh` — real config
  (`max_length=4096`, full-resolution images) on a small task count
  (`VALIDATE_N`, default 50), bounded to `VALIDATE_STEPS` (default 5) via
  a new `train_cli.py --max-steps` override (independent of `--smoke`).
  This is the deliberate middle ground between the Stage 0 smoke gate
  (deliberately undersized, doesn't answer real-scale GPU-memory
  questions) and a full real run — check the real config fits an L4 with a
  handful of cheap steps before committing to hours of real training.
- **Bug caught immediately on first use**: launching with only
  `VALIDATE=true` set (not also `SMOKE=false`) silently ran a **smoke**
  run instead — `SMOKE` defaulted to `true` regardless, and the script's
  `if SMOKE ... elif VALIDATE ...` branching let the default win. Fixed by
  making `SMOKE`'s default conditional on `VALIDATE`: `VALIDATE=true` now
  makes `SMOKE` default to `false` unless explicitly overridden. Verified
  via a standalone bash snippet exercising both branches before relaunching.
- Rationale: this is a real config-footgun class of bug — silent wrong-mode
  execution is worse than a hard failure, since it looks like it's doing
  the right thing (an actual instance came up, an actual run started) while
  answering the wrong question entirely. Caught before any GPU time was
  wasted on the wrong config (deleted immediately, ~1 min after launch).
- Evidence: `bash -c` dry-run of the default-selection logic for both
  `VALIDATE=true` (→ `SMOKE=false`) and unset (→ `SMOKE=true`) cases.

## 2026-08-18 — P4 Stage 0 — SMOKE-LORA GATE CLEARED ✓
- Decision: after 7 live iterations on GCP (each catching and fixing one
  real bug — see the entries below, all from this same session/date),
  `habeas-train-0818-0342` (on-demand L4, us-east1-b, project
  `project-ddef13eb-b20f-47e0-af0`) completed a full smoke SFT run:
  `python -m habeas_model.train_cli --data data/sft-train.jsonl --out
  /root/checkpoints/sft --smoke` — 5/5 training steps, `train_loss: 6.4`
  (a fresh, untrained LoRA's starting loss — expected, not a target),
  clean exit, and a real adapter checkpoint written to
  `/root/checkpoints/sft-final/` (`adapter_model.safetensors`, 499MB,
  confirmed via `ls -la` on the instance). This is the Stage 0 gate
  docs/TRAINING_PLAN.md requires before any real SFT run — **cleared**.
  Bugs found and fixed, in the order hit: (1) `modal.Image.from_dockerfile`
  path resolution, (2) `load_in_4bit=not smoke` → OOM-on-real-hardware bug
  (pre-existing since P0), (3) Modal's local client connection instability
  (pivoted to GCP), (4) `load_in_4bit=True` bare kwarg rejected by
  installed transformers (needs `quantization_config=BitsAndBytesConfig`),
  (5) missing `torchvision` (Qwen3VL's video sub-processor), (6)
  `SFTConfig.max_seq_length` renamed to `max_length`, (7) `gcp_spot.sh`'s
  bare Ubuntu image had **no NVIDIA driver at all** (silent CPU training),
  (8) pip transitively upgraded `torchaudio` past the DLVM image's pinned
  `torch`, an ABI break, (9) `jinja2` too old on the DLVM image, (10) a
  fixed ~4.74GiB fp32-lm_head-upcast OOM in trl's default loss (fixed via
  `use_liger_kernel=True`, not a sequence/image-size workaround — see the
  entry below for the empirical trail on that one specifically).
- Rationale: this is exactly why the gate exists — every one of these 10
  bugs would have surfaced mid-way through a real, multi-hour, real-money
  training run instead, most of them well after real spend had already
  happened. None were catchable by static review alone (each needed the
  actual installed package versions + actual GPU hardware to reproduce).
- Evidence: `/root/checkpoints/sft-final/adapter_model.safetensors`
  (499106600 bytes) on the live instance; full stdout/journal captured at
  each of the 7 iterations, referenced across today's DECISIONS.md
  entries. Instance torn down after confirmation (on-demand billing, no
  reason to keep it running once the artifact is verified).
- Next: this specific smoke checkpoint is disposable (`--smoke` uses 8
  synthetic tasks and 224×224 downscaled images purely to fit an L4) — not
  meant to be a real starting adapter. `docs/TRAINING_PLAN.md`'s Stage
  1/2 (real SFT on the full corpus) is now unblocked to actually execute,
  pending a separate decision on real training's own GPU sizing (the
  liger-kernel fix + max_length=4096 real-mode headroom hasn't itself been
  live-tested at full scale — flagging, not assuming it "just works" at
  10x the data and 4x the sequence length).

## 2026-08-18 — P4 Stage 0 — L4 VRAM fit: liger-kernel fused loss
- Decision: after fixing the driver/torchaudio/jinja2 issues, the smoke
  run reached the actual training step and hit
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.74 GiB`
  inside trl's default chunked cross-entropy loss (`h.float() @
  w.float().t()` in `sft_trainer.py`). Two attempted workarounds — halving
  `max_length` (4096→1024) and downscaling the rendered image
  (700×920→224×224) — both produced the **identical** 4.74GiB allocation
  failure, proving the OOM wasn't driven by sequence length or image size
  at all: it's the cost of upcasting the full `lm_head` weight matrix
  (`vocab_size × hidden_size × 4 bytes`) to fp32 once per loss chunk, a
  fixed cost independent of the batch. On top of the 4-bit model's ~17GB
  footprint, that fixed chunk doesn't fit an L4's ~22GB usable VRAM. Fixed
  by enabling `use_liger_kernel=True` in both `SFTConfig` (`train_cli.py`)
  and `GRPOConfig` (`modal_rlvr.py`) — Liger Kernel's fused CE loss
  computes cross-entropy without ever materializing the full fp32 logits
  matrix, the actual fix rather than a sequence/image-size workaround.
  Added `liger-kernel` to `model/pyproject.toml` (with a `sys_platform ==
  'linux'` marker — it transitively pulls `triton`, which has no macOS
  wheel and broke local `uv sync` entirely until the marker was added),
  `cloud/Dockerfile`, and `cloud/gcp_spot.sh`.
- Rationale: this is a genuine hardware-fit problem for a 27B multimodal
  model on a 24GB-class GPU, not a code bug — the fix needed to actually
  reduce the loss computation's memory footprint, not just shrink inputs
  that weren't the bottleneck. Reverted the earlier (ineffective)
  max_length/image-downscale changes' rationale accordingly — they're kept
  as-is since they're still reasonable smoke-mode economy, just weren't
  the fix for this particular OOM.
- Evidence: identical `Tried to allocate 4.74 GiB` message across 3 live
  smoke attempts with varying `max_length`/image size (ruling out those
  variables empirically, not by inspection); `liger-kernel` installs and
  imports cleanly on the live instance; `model` tests 17/17 green after
  the `sys_platform` marker fix restored local `uv sync`.
- Alternatives rejected: a bigger/different GPU for the smoke gate
  specifically — defeats the point of a *cheap* validation gate; Liger
  Kernel is the standard, well-supported fix for exactly this failure mode
  and required no infrastructure change.

## 2026-08-18 — P4 Stage 0 — gcp_spot.sh had no GPU driver at all
- Decision: after the torchvision/max_length fixes, the smoke run got much
  further (loading model weights) before failing with `ValueError: Your
  setup doesn't support bf16/gpu. You need to assign use_cpu if you want
  to train the model on CPU.` — `nvidia-smi` on the instance: command not
  found; `torch.cuda.is_available()`: `False`. The bare
  `--image-family=ubuntu-2204-lts` image `gcp_spot.sh` used has **no
  NVIDIA driver installed at all** — the L4 accelerator was attached but
  invisible to every process on the box, so the ~25+ minutes of "model
  loading" progress observed in two earlier attempts was actually running
  on CPU the whole time, only caught when `transformers`' `TrainingArgs`
  validation explicitly checked for GPU/bf16 support. Fixed by switching
  to Google's Deep Learning VM image family
  (`pytorch-2-9-cu129-ubuntu-2204-nvidia-580`,
  project `deeplearning-platform-release`) — PyTorch/CUDA/driver
  preinstalled and version-matched, the standard approach for GCP GPU
  workloads rather than manually apt-get installing a driver on a bare
  image. Also removed `torch`/`torchvision` from the startup script's own
  `pip install` line, since a bare `pip install torch` on top of the DLVM
  image risks silently replacing its CUDA-linked build with a mismatched
  or CPU-only wheel from PyPI's default index.
- Rationale: this is the highest-value bug the smoke gate has caught so
  far — a "successful"-looking CPU run would have produced a technically-
  working but catastrophically slow and pointless LoRA adapter, and this
  exact silent-CPU-fallback failure mode is called out generically in
  `mlops-pipeline-design`'s corner-cases checklist ("pipeline ran without
  error ≠ validated").
- Evidence: `nvidia-smi`/`torch.cuda.is_available()` checks on the live
  instance; `gcloud compute images describe-from-family` confirms the new
  image family exists with a 100GB disk (matches the script's existing
  `--boot-disk-size=100GB`, no change needed there).
- Alternatives rejected: manually `apt-get install nvidia-driver-XXX` +
  reboot on the bare image — slower, more fragile (driver/CUDA/torch
  version matching is exactly what the DLVM image exists to solve), and
  reboot-in-startup-script adds its own failure modes.

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

## 2026-08-20 — P4 — Real SFT run COMPLETE ✓
- Decision: full real SFT run (`GPU_TYPE=nvidia-tesla-a100 PREEMPTIBLE=true
  BATCH_SIZE=2 GRAD_ACCUM=4`, 400 steps, full corpus/epochs) launched
  2026-08-18 ~16:22 EDT on instance `habeas-train-0818-1622`, completed
  2026-08-20 ~00:06 UTC. Adapter checkpoint pulled to local
  `checkpoints/sft-final/` (986MB, LoRA adapter only — `checkpoints/` added
  to `.gitignore`, not committed) and the GCE instance deleted.
- Evidence: `train.log` final lines —
  `train_runtime=9389s`, `train_loss≈0.0103` at completion, steady
  `mean_token_accuracy` logged every 10 steps throughout. One preemption
  occurred mid-run (step ~330s region) — checkpoint-resume worked exactly
  as designed: restarted from `checkpoint-360` (the last periodic save
  before the preempt), not from scratch. `checkpoint-400`/`sft-final` both
  present and verified on disk (`du -sh` = 971MB on the instance, 986MB
  after local tar/scp round-trip — difference is filesystem block
  overhead, not data loss).
- Real per-step pace observed: ~231-236s/it steady-state throughout (batch
  size 2 change from earlier this session held up under the full run, not
  just the short validation pass).
- Not yet done: no eval run against `data/golden.jsonl` yet (needs a
  `Provider` adapter wrapping this checkpoint + a GPU to run inference on —
  next action, not done in this entry). RLVR stage (`cloud/modal_rlvr.py`,
  GRPO against `oracle_reward_func`) also not yet started — per
  `docs/TRAINING_PLAN.md`, uses this SFT adapter as its base.
- User directive: HF upload deferred until "the final version" (i.e. after
  eval/RLVR, not this raw SFT-only checkpoint).

## 2026-08-20 — P4 — Eval pipeline (2 live bugs) + first SFT eval numbers
- Built `model/src/habeas_model/local_provider.py` (`LocalHFProvider`,
  base+LoRA via transformers/peft, mirrors train_cli.py's exact 4-bit/
  bfloat16 load config) and `eval_cli.py` wiring it into
  `benchmark_eval.run_eval`. Two live bugs found running this on a GCP
  A100 (instance `habeas-eval-0819-2225`), both fixed and pushed before a
  usable eval completed:
  1. `apply_chat_template(images=...)` — not a recognized parameter; gets
     silently misrouted into `processor.__call__`'s kwargs, raising
     `TypeError: Qwen3VLProcessor: ...`. Fix: images belong inline in the
     message `content` list (`{"type": "image", "image": <PIL>}`) — the
     Jinja template only emits `<|vision_start|><|image_pad|><|vision_end|>`
     for content items carrying an image key, no separate kwarg needed.
  2. Qwen3's chat template defaults `enable_thinking` to **True** when
     unset (`{%- if enable_thinking is undefined or enable_thinking is
     true %}`). Without explicitly passing `enable_thinking=False`, the
     model emitted step-by-step CoT prose instead of the trained
     JSON-only verdict, overrunning `max_new_tokens=512` before reaching
     any parseable output (`predicted: null` on every task). CONTRACTS.md
     §6 requires non-thinking output — this was silently violating it.
  Both fixed via a live debug script isolating one task's raw completion
  before re-running the full eval — same iterate-on-real-hardware pattern
  as the training bugs.
- **First real SFT-adapter eval, 150/1000 golden-set subsample** (seed
  777 tasks, sequential subset — not random sample, a scope-reduction
  decision made with the user to avoid a ~25hr full-1000 run before any
  signal existed at all):
  `parse_rate=0.993`, `verdict_accuracy=0.927`,
  `severity_weighted_recall=0.615`, `false_positives_per_task=0.313`
  (n=150, 114 violation-bearing tasks). Results:
  `data/eval-results-sft-golden150.jsonl` (gitignored).
- Interpretation: near-perfect JSON-format compliance and strong
  PASS/FLAG calibration (92.7%) — the SFT adapter learned the output
  contract and top-level verdict well. Severity-weighted recall (61.5%)
  is the weak point: spot-checked example (task `i9-777-997742588`)
  showed a correct FLAG verdict with the right severity/CFR citation but
  the wrong violation *type* (`TIMELINESS` predicted vs
  `FIELD_INCOMPLETE` expected) — plausible root-cause misclassification
  rather than random noise. This is exactly the gap RLVR's
  `oracle_reward_func` (exact type/severity match, not just verdict
  match) is designed to close — proceeding to RLVR is the right next
  step rather than a second SFT pass.
- Per-task inference throughput: real, highly variable — observed
  ~90s/task in short bursts up to ~600s/task at points, averaging
  roughly 60-130s/task steady-state on one A100, single-sample sequential
  generation (`max_workers=1`, no batching — GPU util sat at 37-39%
  during generation, meaningful headroom unused; batching multiple
  eval prompts per forward pass is a real speedup opportunity not
  implemented here, flagged for a future eval-throughput pass rather
  than blocking RLVR on it).

## 2026-08-20 — P4 — RLVR research pass + 2 live bugs on first smoke run
- Before any RLVR GPU spend, researched 2026 SOTA (user request: "reach
  the biggest we can" before more GPU work) and corrected `modal_rlvr.py`'s
  original never-live-verified design: `epsilon_high=1.0` was wrong (DAPO
  paper value is 0.28, confirmed against installed `trl/grpo_config.py`);
  adopted GSPO (`importance_sampling_level="sequence"`, the algorithm
  Qwen3 itself trains with — arXiv:2507.18071) over plain GRPO's noisy
  token-level importance ratio. Also researched LLM-as-a-Verifier
  (arXiv:2607.05391, user-supplied) — not applicable to the RLVR reward
  itself (we have a deterministic oracle, strictly safer against reward
  hacking than any LLM-verifier), but a real future upgrade path for
  Stage 4 self-play (best-of-N candidate selection) and Stage 5 judges
  (continuous logprob scoring) — noted in methodology.md, not built yet.
  Added `model/src/habeas_model/rlvr_cli.py` (GCP-path entrypoint, same
  Modal/GCP split `run_sft` already went through) and `cloud/gcp_rlvr.sh`
  (`MODE=smoke|validate|real`, same enum discipline as `gcp_spot.sh`).
  Added 6 verifier-hardening unit tests (fuzzing `reward()` against
  degenerate completions) per the research's "fuzz the verifier before
  training" recommendation.
- **First live RLVR smoke run** (`habeas-rlvr-0820-0244`, A100 preemptible)
  found 2 real bugs neither the literature review nor the code-only round
  caught:
  1. TRL warns at runtime that pairing `importance_sampling_level=
     "sequence"` with `loss_type="dapo"` (TRL's plain default, and this
     file's first guess) sums per-token contributions in a way that
     doesn't reproduce a true per-sequence objective — the library's own
     warning says explicitly to use `loss_type="grpo"` to reproduce
     GSPO's actual paper setup. Fixed: `loss_type="grpo"`.
  2. `oracle_reward_func` assumed `completions: list[str]`, but
     GRPOTrainer wraps each completion as `[{"role": "assistant",
     "content": text}]` whenever the dataset is conversational (ours is —
     `build_rlvr_prompt`'s `"prompt"` column is a list of role/content
     dicts). Crashed with `TypeError: expected string or bytes-like
     object` inside `to_forge_verdict` on the very first reward
     computation. Fixed: unwrap `completion[0]["content"]` when
     `completion` is a list, matching `grpo_trainer.py`'s own
     `is_conversational(inputs[0])` branch.
  Both fixed, tested (new `test_oracle_reward_func_conversational_
  completion_shape`), and pushed before the smoke run was re-attempted —
  same live-iterate-on-real-hardware discipline as every other bug this
  session.
- **3 more bugs found across re-attempts, in order**: (a) with liger
  enabled, crashed inside `compute_liger_loss`'s vision forward with
  "Image features and image tokens do not match, tokens: 5107, features:
  5104" — disabling `use_liger_kernel` (new toggle added) got past it,
  seeming to confirm a liger+multimodal-batching interaction; (b) with
  liger disabled, `per_device_train_batch_size` defaulted to HF's
  `TrainingArguments` default of 8 (never set explicitly) — OOM'd
  ("Tried to allocate 3.79 GiB" on a 39.49GB A100) inside the logprob
  forward pass; set explicit `batch_size=1`/`grad_accum` mirroring what
  fit SFT; (c) `GRPOConfig.__post_init__` requires `generation_batch_size`
  (`batch_size * grad_accum` by default) evenly divisible by
  `num_generations` — `grad_accum=4` with `group_size=8` failed that
  check; bumped default `grad_accum` to 8.
- **Status: BLOCKED, not resolved.** With all of the above fixed
  (`--no-liger-kernel --batch-size 1 --grad-accum 8`), the smoke run got
  further than ever — 3/5 steps completed cleanly — then hit the **same**
  "Image features and image tokens do not match" error again on step 4
  (tokens: 639, features: 638, a different off-by-1 this time vs the
  earlier off-by-3), this time **without liger enabled**. This disproves
  the liger-specific theory from bug (a) above — the real bug is
  independent of liger. Since `render_form()` always renders a fixed
  700×920 canvas (constant across every task, no per-task image-size
  variance), and the failure is intermittent (3 clean steps, then a
  failure) rather than deterministic, this points at TRL's own documented
  multi-image-batch-splitting bug (huggingface/trl#4488: "Missing
  `image_grid_thw` variable in batch input skips the split processing on
  `split_pixel_values_by_grid`") rather than anything in this repo's
  code — each GRPO micro-batch effectively carries `num_generations`
  duplicate copies of one source image, and TRL's own indexing of that
  duplicated-image pixel-value block is apparently still fragile in the
  installed `trl==1.10.0` for some (not all) micro-batch compositions,
  despite that issue reportedly being fixed in trl>=0.25.0.
- **Not attempted this round** (stopping to reassess rather than keep
  guessing at real GPU cost — 5 live attempts, multiple hours of A100
  wall-clock spent on this one blocker): monkey-patching TRL's
  `split_pixel_values_by_grid`/`_get_per_token_logps_and_entropies` per
  the fix described in trl#4488's closed PR #6570; bisecting trl versions
  around 0.25.0 for a possible regression in 1.10.0; filing/checking for
  an upstream trl issue matching this exact intermittent variant.
  Instance `habeas-rlvr-0820-0244` deleted; RLVR is paused, SFT-only
  checkpoint (`checkpoints/sft-final/`, eval numbers above) remains the
  current best artifact.
- **Confirmed DETERMINISTIC, not flaky (habeas-rlvr-0820-1012, same day,
  full retry of the identical config)**: same 8-record smoke set, same
  flags (`--no-liger-kernel --batch-size 1 --grad-accum 8`), same 3 clean
  steps, then the **exact same** error at step 4 — `tokens: 639, features:
  638`, byte-identical to the first attempt's numbers. Rules out a
  race/nondeterminism theory entirely: this is a specific micro-batch
  composition reached deterministically at step 4 of this smoke set that
  reliably triggers TRL's image-token/feature-count bug. A third retry
  would almost certainly reproduce it again — not worth the GPU spend to
  confirm further. Instance deleted. **Decision: pause RLVR here** rather
  than invest in a blind monkeypatch of TRL internals we can't fully see
  (real risk: a wrong patch could silently corrupt logprob computation
  instead of loudly crashing, which is worse than the current failure
  mode). Next real attempt should either (a) get a genuine fix/version
  from upstream trl, or (b) deliberately trade away part of GSPO's
  group-relative design (e.g. `group_size=1`, sacrificing the
  group-normalization RLVR depends on) as a fallback, not a first choice.

## 2026-08-21 — P4 — Teacher distillation attempted: negative result
- **Decision (user)**: pause RLVR (blocked, see above), pursue teacher
  distillation as the next lever instead — SFT was trained on raw oracle
  targets, not verifier-filtered teacher traces per methodology.md's
  actual Stage 2 spec, and the eval gap (weak violation-type recall,
  strong verdict/format) looked like exactly what real reasoning traces
  from a stronger model might fix.
- **Infra built**: `model/src/habeas_model/vertex_provider.py`
  (`VertexProvider`, Gemini via Vertex AI REST API, `gcloud auth
  print-access-token` auth — no new credentials, reuses the existing GCP
  project). Live-found and fixed 2 bugs before any real distillation
  volume: (1) `SYSTEM_PROMPT` never enumerated the actual
  `ViolationType`/`Severity` values, so the teacher invented its own
  labels the verifier-filter silently rejected — fixed by building the
  enum list into the prompt from `habeas_forge.schema` directly; (2)
  `_access_token`'s naive thread-safety and time-based-only refresh
  caused a real stalled run (8 concurrent `gcloud` calls contending on
  its own config-directory lock) and a real 807-call 401 cascade
  (token expired inside the 45-min refresh window) — fixed with a
  `threading.Lock` + retry-on-401-with-forced-refresh + retry-with-
  backoff-on-429.
- **Quota wall**: `gemini-2.5-flash` (the only servable model — all
  `gemini-1.5-*`/`gemini-pro`/`gemini-experimental`/`model-optimizer`
  quota rows are stale metadata, 404 on actual `generateContent` calls)
  has no dedicated per-model quota in this project; it falls under a
  non-adjustable system-default bucket (confirmed both via `gcloud alpha
  services quota update` — `COMMON_QUOTA_CONSUMER_OVERRIDE_TOO_HIGH, max:
  0` — and via the Console UI, user-confirmed "not adjusted"/no
  self-service option). At `max_workers=8` this saturated into a 429
  cascade; even at `max_workers=3` sustained throughput was ~2 records/15
  min (~150+ hours to complete the full 1597-task corpus) — genuinely
  impractical. No self-service fix exists; a real fix would need a
  Google Cloud support ticket (uncertain timeline, possibly needs a paid
  support tier), not pursued further this round.
- **Decision (user)**: stop at whatever was collected (399 verifier-
  filtered records, ~25% of the 1597-task corpus) rather than wait on
  quota, and retrain SFT on that set to see if distillation helps at all
  even at reduced scale.
- **Real retrain**: `habeas-train-0820-1627`, A100 preemptible, same
  `batch_size=2/grad_accum=4` as the original SFT run, 399 records → 100
  steps (vs. the original's 1597 records → 400 steps), survived one
  preemption via checkpoint-resume (resumed from step 6, not from
  scratch). Final `train_loss≈0.077`. Checkpoint:
  `checkpoints/sft-teacher-final/` (local, gitignored).
- **Eval result — NEGATIVE across every metric**, 150-task golden
  subsample (same subsample as the oracle-only baseline, apples-to-apples
  on the eval side):

  | Metric | Oracle-only SFT (1597 recs/400 steps) | Teacher-distilled (399 recs/100 steps) |
  |---|---|---|
  | parse_rate | 0.993 | 0.960 |
  | verdict_accuracy | 0.927 | 0.780 |
  | severity_weighted_recall | 0.615 | 0.398 |
  | false_positives_per_task | 0.313 | 0.740 |

  Reported honestly, not spun: this is a real regression, not a wash.
  **Confounded, not a clean test of "teacher distillation" as a
  technique** — the teacher-distilled run also had ~4x less data and ~4x
  fewer training steps than the baseline, either of which alone could
  plausibly explain a drop this size. A fair test would need a
  teacher-distilled corpus matching the oracle run's full 1597-task
  scale, which the quota wall above made infeasible this round.
  `data/sft-train-teacher.jsonl` (399 records) and the raw eval results
  are the artifacts if this is revisited once quota is available.
- **Net effect on "current best artifact"**: unchanged —
  `checkpoints/sft-final/` (the original oracle-only SFT adapter) remains
  the best-performing checkpoint measured so far.
  `checkpoints/sft-teacher-final/` is kept locally for reference/
  possible future analysis but is not promotable given these numbers.

## 2026-08-22 — P4 — Citation exact-match metric + definitive full-golden eval
- Added `citation_exact_match` to `score_predictions`/`summarize`
  (`forge/src/habeas_forge/score.py`) — README.md's promotion gate names
  a >95% citation-accuracy target that was never actually measured before
  this. Denominator is caught-violation-instance count (a miss or false
  positive has no citation to score); vacuously 1.0 when nothing was
  caught, matching `severity_weighted_recall`'s own no-denominator
  convention. 3 new tests (`forge/tests/test_forge.py`).
- **Full 1000-task golden eval** (not a subsample this time) of
  `checkpoints/sft-final` on `habeas-eval-0821-0943` (A100 preemptible,
  survived zero preemptions this run):

  | Metric | 150-task subsample (2026-08-20) | **Full 1000 (definitive)** |
  |---|---|---|
  | parse_rate | 0.993 | **1.000** |
  | verdict_accuracy | 0.927 | **0.931** |
  | severity_weighted_recall | 0.615 | **0.614** |
  | false_positives_per_task | 0.313 | **0.251** |
  | citation_exact_match | (not measured) | **0.915** |

  The 150-task subsample tracked the full 1000 closely on every
  overlapping metric (largest gap: false_positives_per_task, 0.313 vs
  0.251 — the subsample happened to be a bit noisier there, not
  systematically biased). Confirms `severity_weighted_recall` (61.4%) is
  the real, stable weak point, not a small-sample artifact — this is the
  metric a future RLVR or better-distillation pass needs to move.
  `citation_exact_match` (91.5%) is decent but below README's >95% gate.
  Raw per-task results: `data/eval-results-sft-golden-full.jsonl`
  (gitignored).
- This is now the **definitive, promotion-gate-comparable number** for
  `checkpoints/sft-final` — supersedes the 150-task subsample figure
  everywhere it's cited (docs/BENCHMARK.md, docs/HANDOFF.md updated).

## 2026-08-22 — P4 — Root-cause analysis + rules-reference prompt fix (Phase 0/0b)
- User asked for a deep, SOTA-informed dive into closing the
  severity-weighted-recall gap (61.4% vs the >95% promotion target)
  rather than another blind RLVR/distillation attempt. Free error
  analysis on the already-collected full-golden eval results
  (`data/eval-results-sft-golden-full.jsonl`) found the gap was far more
  tractable than assumed:
  - **189/436 missed instances (43%) were a single, 100%-consistent
    severity-inversion bug**, not a detection failure — the model
    correctly identifies the violation type but substitutes its own
    "common sense" severity for the oracle's fixed per-type constant:
    CATEGORY_MISMATCH/FIELD_INCOMPLETE (oracle=MEDIUM, model always
    guessed HIGH, 79/79 and 32/32) and DATA_INCONSISTENT (oracle=LOW,
    model always guessed MEDIUM, 78/78). Confirmed against `verify.py`'s
    `_SEV` dict: severity here is a static lookup table, not
    context-dependent.
  - **99/436 (23%) were EDITION_WRONG true misses** despite normal
    training representation (168/1597 tasks) — `VALID_EDITIONS =
    {"2023-08-01", "2025-01-20"}` is an external fact never stated in
    `SYSTEM_PROMPT`, purely inferable from training examples.
  - 2026 fine-tuning literature confirms this exact failure mode:
    fine-tuning learns patterns well but doesn't reliably memorize
    precise lookup-table facts; explicit rule-injection into the prompt
    is the documented cheap fix, cheaper than any retrain.
- **Phase 0** (severity table only, injected into `SYSTEM_PROMPT`, tested
  against the already-trained `checkpoints/sft-final` — no retrain) on a
  180-task subsample weighted toward the 4 problem types
  (`data/golden-targeted.jsonl` — 150 tasks containing at least one of
  CATEGORY_MISMATCH/DATA_INCONSISTENT/FIELD_INCOMPLETE/EDITION_WRONG +
  30 other tasks for baseline signal):
  `severity_weighted_recall=0.855` (vs 0.614 full-golden baseline),
  `verdict_accuracy=0.994`, `false_positives_per_task=0.183`. Real, large
  improvement — but `citation_exact_match` dropped to `0.559` (vs 0.915
  full-golden baseline): the newly-caught instances (thanks to the
  severity fix) still had un-stated citations for the model to guess at.
- **Root cause of the citation regression**: every violation type also
  fires from exactly one fixed CFR/M-274 citation (confirmed by reading
  every `_v()` call site in `verify.py` — `REMOTE_EXAM_INVALID`'s two
  branches even share the same citation) — same failure mode as
  severity, just not caught until severity-matching exposed it. Refactored
  `verify.py` to expose `TYPE_CFR: dict[ViolationType, str]` (was inline
  literal strings duplicated across call sites) and added it to
  `SYSTEM_PROMPT` alongside the severity table. Zero behavior change to
  `verify()` itself (forge tests green, same 15/15 before and after).
- **Phase 0b** (severity + citation tables, same checkpoint, same
  180-task subsample):
  `severity_weighted_recall=0.765` (still +15pts over the 0.614
  baseline, though a real dip from Phase 0's 0.855 — the longer prompt
  traded a little severity precision for citation precision, an honest
  tradeoff worth noting, not spun away), `citation_exact_match=1.000`
  (up from 0.559, and above the 0.915 full-golden baseline),
  `verdict_accuracy=0.994`, `false_positives_per_task=0.217`. Raw
  results: `data/eval-results-phase0b-targeted.jsonl`.
- **Decision: proceed to Phase 1** — rebuild the SFT training corpus
  with this prompt baked into every training example's system turn (not
  just used at inference), then retrain from scratch and re-evaluate on
  the full 1000-task golden set. Both Phase 0 and 0b passed the
  go/no-go bar (large gains on the target metric, no metric catastrophically
  broken) even before any retraining — a real, cheap, targeted fix
  instead of the originally-assumed need for RLVR or teacher distillation
  at scale.
