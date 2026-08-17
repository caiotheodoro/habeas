FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip python3-venv curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    "torch>=2.6" "transformers>=4.55" "peft>=0.13" "trl>=0.16" \
    "accelerate>=1.0" "datasets>=3.0" "pillow>=11" "huggingface_hub>=0.26" \
    "bitsandbytes>=0.44" "flash-attn" "unsloth>=2025.11" "vllm"

COPY . .

CMD ["python", "-m", "habeas_model.train"]
