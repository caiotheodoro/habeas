#!/usr/bin/env bash
# GCP $300 fallback: L4 training. Three modes:
#   SMOKE=true (default)     — cheap tooling-only validation (Stage 0 gate),
#                               8 tasks, 224x224 images, max_length=1024.
#   VALIDATE=true            — real config (max_length=4096, full-res
#                               images) on a small task count, bounded to a
#                               few steps — the way to check real-scale
#                               GPU-memory fit BEFORE committing to a full
#                               run. The smoke config is deliberately too
#                               small to answer that question (verified:
#                               reducing smoke's max_length/image size had
#                               zero effect on its OOM — the true fix there
#                               was use_liger_kernel, see docs/DECISIONS.md;
#                               real-scale headroom is still unverified).
#   both false                — real full run: full corpus, real epochs.
# PREEMPTIBLE=false (default) uses an on-demand instance — spot L4 capacity
# in this project has been unreliable (stockouts across most zones, and one
# spot instance got preempted mid-boot), so on-demand trades a slightly
# higher hourly rate for actually completing a run without a capacity retry
# loop. Set PREEMPTIBLE=true once capacity/reliability allow.
set -euo pipefail
PROJECT=${GCP_PROJECT:-cambio-curitiba-498923}
ZONE=${GCP_ZONE:-us-central1-a}
MACHINE=${GCP_MACHINE:-g2-standard-12}
VALIDATE=${VALIDATE:-false}
# SMOKE defaults to true UNLESS VALIDATE=true was explicitly requested —
# without this, SMOKE's default silently wins the if/elif below and a
# caller who only set VALIDATE=true gets a smoke run instead (this
# actually happened once — see docs/DECISIONS.md).
if [ "$VALIDATE" = "true" ]; then
  SMOKE=${SMOKE:-false}
else
  SMOKE=${SMOKE:-true}
fi
PILOT_N=${PILOT_N:-2000}
VALIDATE_N=${VALIDATE_N:-50}
VALIDATE_STEPS=${VALIDATE_STEPS:-5}
PREEMPTIBLE=${PREEMPTIBLE:-false}
NAME="habeas-train-$(date +%m%d-%H%M)"

# GPU-attached VMs never support live migration, so --maintenance-policy
# TERMINATE is required either way — only --preemptible itself toggles.
if [ "$PREEMPTIBLE" = "true" ]; then
  PREEMPT_FLAGS="--maintenance-policy=TERMINATE --preemptible"
else
  PREEMPT_FLAGS="--maintenance-policy=TERMINATE"
fi

if [ "$SMOKE" = "true" ]; then
  PILOT_N=20
  TRAIN_CLI_FLAG="--smoke"
elif [ "$VALIDATE" = "true" ]; then
  PILOT_N=$VALIDATE_N
  TRAIN_CLI_FLAG="--max-steps $VALIDATE_STEPS"
else
  TRAIN_CLI_FLAG=""
fi

STARTUP_SCRIPT="$(mktemp)"
trap 'rm -f "$STARTUP_SCRIPT"' EXIT
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
# The image family below (see instance-create below) is a Google Deep
# Learning VM: PyTorch + CUDA + the NVIDIA driver are already installed.
# A bare ubuntu-2204-lts image has NO GPU driver at all — an earlier
# attempt trained silently on CPU for 20+ min before transformers'
# TrainingArgs validation caught it ("doesn't support bf16/gpu"); found
# via an actual smoke run, see docs/DECISIONS.md.
# flash-attn/unsloth/vllm intentionally omitted: nothing in the current
# SFT/RLVR code paths imports them (see the cloud/Dockerfile comment).
# torch/torchvision intentionally NOT reinstalled here: the DLVM image's
# preinstalled build is CUDA-linked; a bare \`pip install torch\` can
# silently replace it with a mismatched or CPU-only wheel.
# jinja2>=3.1.0 required by transformers' apply_chat_template; the DLVM
# image ships 3.0.3 (found via an actual smoke run — ImportError
# otherwise, see docs/DECISIONS.md).
# liger-kernel: fused CE loss avoids trl's default fp32 lm_head upcast,
# which OOM'd an L4 with a fixed ~4.74GiB allocation regardless of
# max_length/image size (found via an actual smoke run).
pip3 install "jinja2>=3.1.0" transformers peft trl accelerate datasets bitsandbytes liger-kernel click pydantic pillow numpy
# One of the above transitively pulled torchaudio 2.11.0 while the image's
# torch stayed at 2.9.1 — an ABI break (peft -> transformers -> ... ->
# torchaudio's compiled extension: "undefined symbol"). We don't use audio
# at all, so just remove it rather than fight pip's resolver for a pin
# that would need re-verifying every image update. Found via an actual
# smoke run, see docs/DECISIONS.md.
pip3 uninstall -y torchaudio || true
cd /root && git clone https://github.com/caiotheodoro/habeas.git && cd habeas
export PYTHONPATH=/root/habeas/model/src:/root/habeas/forge/src
# data/ is gitignored: a fresh clone has no pilot/train/val files, so
# regenerate deterministically (seed 7, matches docs/DECISIONS.md P4 Stage
# 1) instead of assuming a prebuilt data artifact exists on the box.
cd forge
python3 -m habeas_forge.cli pilot --seed 7 --n $PILOT_N --out data/pilot.jsonl
python3 -m habeas_forge.cli split --pilot data/pilot.jsonl --out-train data/train.jsonl --out-val data/val.jsonl
cd ..
python3 -m habeas_model.dataset_builder build --tasks-file forge/data/train.jsonl --out data/sft-train.jsonl
python3 -m habeas_model.train_cli --data data/sft-train.jsonl --out /root/checkpoints/sft $TRAIN_CLI_FLAG 2>&1 | tee /root/train.log
EOF

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 \
  $PREEMPT_FLAGS \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"

echo "started $NAME (L4, preemptible=$PREEMPTIBLE, SMOKE=$SMOKE, VALIDATE=$VALIDATE) — gcloud compute ssh $NAME --zone=$ZONE"
echo "tail progress:   gcloud compute ssh $NAME --zone=$ZONE --command='tail -f /root/train.log'"
