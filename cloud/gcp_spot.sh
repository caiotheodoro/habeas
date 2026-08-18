#!/usr/bin/env bash
# GCP $300 fallback: spot L4 training marathon.
set -euo pipefail
PROJECT=${GCP_PROJECT:-cambio-curitiba-498923}
ZONE=${GCP_ZONE:-us-central1-a}
MACHINE=${GCP_MACHINE:-g2-standard-12}
NAME="habeas-train-$(date +%m%d-%H%M)"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE --preemptible \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --metadata=startup-script='#!/bin/bash
set -euo pipefail
apt-get update && apt-get install -y python3-pip
pip3 install torch transformers peft trl accelerate datasets bitsandbytes flash-attn unsloth vllm click pydantic pillow numpy
cd /root && git clone https://github.com/caiotheodoro/habeas.git && cd habeas
export PYTHONPATH=/root/habeas/model/src:/root/habeas/forge/src
# data/ is gitignored: a fresh clone has no pilot/train/val files, so
# regenerate deterministically (seed 7, matches docs/DECISIONS.md P4 Stage
# 1) instead of assuming a prebuilt data artifact exists on the box.
cd forge
python3 -m habeas_forge.cli pilot --seed 7 --n 2000 --out data/pilot.jsonl
python3 -m habeas_forge.cli split --pilot data/pilot.jsonl --out-train data/train.jsonl --out-val data/val.jsonl
cd ..
python3 -m habeas_model.dataset_builder build --tasks-file forge/data/train.jsonl --out data/sft-train.jsonl
python3 -m habeas_model.train_cli --data data/sft-train.jsonl --out /root/checkpoints/sft 2>&1 | tee /root/train.log'

echo "started $NAME (spot L4) — gcloud compute ssh $NAME --zone=$ZONE"
