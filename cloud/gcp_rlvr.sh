#!/usr/bin/env bash
# GCP RLVR launch: GRPO/GSPO against the forge oracle reward, base = an
# SFT adapter already produced by cloud/gcp_spot.sh. MODE is REQUIRED —
# same enum discipline as gcp_spot.sh (its header comment documents two
# real near-misses from a boolean-pair default; never repeat that
# pattern). Set MODE=smoke|validate|real:
#   smoke     — 8 prompts, 5 steps. Tooling-only validation (mirrors the
#               SFT smoke-LoRA gate that caught 10 real bugs before any
#               real SFT spend — expect this to find real bugs too).
#   validate  — a small prompt count, bounded steps, real image/prompt
#               scale — confirms real GPU-memory fit (group_size rollouts
#               are a different memory profile than SFT's single forward/
#               backward pass) before committing to a full run.
#   real      — full run against data/rlvr-prompts.jsonl (built from
#               train.jsonl, n=2000), no step bound.
# GPU_TYPE=nvidia-tesla-a100 (default here — L4 was already confirmed
# insufficient for the real SFT config; RLVR's rollout memory footprint
# is at least as demanding, no reason to re-attempt L4). PREEMPTIBLE=true
# required for A100 (only preemptible quota approved as of 2026-08-18,
# see docs/DECISIONS.md).
# BASE_ADAPTER: path to the SFT adapter dir on the instance (uploaded the
# same tar+scp way checkpoints/sft-final was pulled down after SFT
# training — see docs/DECISIONS.md's "Real SFT run COMPLETE" entry).
set -euo pipefail
PROJECT=${GCP_PROJECT:-project-ddef13eb-b20f-47e0-af0}
ZONE=${GCP_ZONE:-us-central1-a}
GPU_TYPE=${GPU_TYPE:-nvidia-tesla-a100}
if [ "$GPU_TYPE" = "nvidia-tesla-a100" ]; then
  MACHINE=${GCP_MACHINE:-a2-highgpu-1g}
else
  MACHINE=${GCP_MACHINE:-g2-standard-12}
fi
MODE=${MODE:?"Set MODE=smoke|validate|real (no default — see header comment)"}
case "$MODE" in
  smoke|validate|real) ;;
  *) echo "MODE must be smoke, validate, or real (got: $MODE)" >&2; exit 1 ;;
esac
PREEMPTIBLE=${PREEMPTIBLE:-true}
BASE_ADAPTER=${BASE_ADAPTER:-/root/checkpoints/sft-final}
GROUP_SIZE=${GROUP_SIZE:-8}
VALIDATE_STEPS=${VALIDATE_STEPS:-5}
HF_TOKEN=${HF_TOKEN:-}
NAME="habeas-rlvr-$(date +%m%d-%H%M)"

if [ "$PREEMPTIBLE" = "true" ]; then
  PREEMPT_FLAGS="--maintenance-policy=TERMINATE --preemptible"
else
  PREEMPT_FLAGS="--maintenance-policy=TERMINATE"
fi

case "$MODE" in
  smoke)
    RLVR_CLI_FLAG="--smoke"
    ;;
  validate)
    RLVR_CLI_FLAG="--max-steps $VALIDATE_STEPS"
    ;;
  real)
    RLVR_CLI_FLAG=""
    ;;
esac
RLVR_CLI_FLAG="$RLVR_CLI_FLAG --group-size $GROUP_SIZE"

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
# Same DLVM image, same driver/dependency fixes as gcp_spot.sh — see that
# script's comments for why each of these is needed (bare ubuntu has no
# GPU driver; torchaudio ABI break; old jinja2; liger-kernel for the fixed
# fp32-lm_head OOM).
pip3 install "jinja2>=3.1.0" transformers peft trl accelerate datasets bitsandbytes liger-kernel click pydantic pillow numpy
pip3 uninstall -y torchaudio || true
[ -d /root/habeas ] || (cd /root && git clone https://github.com/caiotheodoro/habeas.git)
cd /root/habeas
export PYTHONPATH=/root/habeas/model/src:/root/habeas/forge/src
$([ -n "$HF_TOKEN" ] && echo "export HF_TOKEN=$HF_TOKEN")
# train.jsonl must already exist (either pre-uploaded or regenerated the
# same deterministic way gcp_spot.sh does) — RLVR prompts are built from
# it, never from val/golden (methodology.md: RLVR data never mixed into
# SFT, and prompts must never be sourced from the held-out eval sets).
cd forge
[ -f data/train.jsonl ] || {
  python3 -m habeas_forge.cli pilot --seed 7 --n 2000 --out data/pilot.jsonl
  python3 -m habeas_forge.cli split --pilot data/pilot.jsonl --out-train data/train.jsonl --out-val data/val.jsonl
}
cd ..
[ -f data/rlvr-prompts.jsonl ] || python3 -m habeas_model.dataset_builder prompts --tasks-file forge/data/train.jsonl --out data/rlvr-prompts.jsonl
python3 -m habeas_model.rlvr_cli --prompts data/rlvr-prompts.jsonl --base-adapter $BASE_ADAPTER --out /root/checkpoints/rlvr $RLVR_CLI_FLAG 2>&1 | tee -a /root/rlvr.log
EOF

BOOT_DISK_SIZE=${GCP_BOOT_DISK_SIZE:-300GB}
gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=$GPU_TYPE,count=1 \
  $PREEMPT_FLAGS \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo "started $NAME ($GPU_TYPE, preemptible=$PREEMPTIBLE, MODE=$MODE) — gcloud compute ssh $NAME --zone=$ZONE"
echo "NOTE: $BASE_ADAPTER must exist on the instance before rlvr_cli runs — upload the SFT adapter first (tar+scp, same as sft-final was pulled down)."
echo "tail progress:   gcloud compute ssh $NAME --zone=$ZONE --command='tail -f /root/rlvr.log'"
