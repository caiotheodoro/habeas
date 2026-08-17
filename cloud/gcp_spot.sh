#!/usr/bin/env bash
# GCP $300 fallback: spot L4 training marathon.
set -euo pipefail
PROJECT=${GCP_PROJECT:-cambio-curitiba-498923}
ZONE=${GCP_ZONE:-us-central1-a}
MACHINE=${GCP_MACHINE:-g2-standard-12}
NAME="attest-train-$(date +%m%d-%H%M)"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE --preemptible \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --metadata=startup-script='#!/bin/bash
apt-get update && apt-get install -y python3-pip
pip3 install torch transformers peft trl accelerate datasets bitsandbytes flash-attn unsloth vllm
cd /root && git clone https://github.com/caiotheodoro/attest.git && cd attest
python3 -m attest_model.train --data data/train.jsonl 2>&1 | tee /root/train.log'

echo "started $NAME (spot L4) — gcloud compute ssh $NAME --zone=$ZONE"
