FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3-pip python3-venv curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# flash-attn/unsloth/vllm deliberately NOT installed here: nothing in
# train_cli.py/modal_train.py/modal_rlvr.py imports them today (verified
# via grep — they were speculative future-stage deps, e.g. a DeltaNet-
# specific gate or a later inference-serving stage). Compiling flash-attn
# from source alone can take 20-40+ min with no functional benefit to the
# current SFT/RLVR code paths — add them back in a follow-up image once
# something actually imports them, not preemptively.
# torchvision: Qwen3VL's AutoProcessor pulls in a video-capable sub-
# processor that hard-requires it even for image-only use (found via an
# actual smoke run — ImportError otherwise; a sibling repo hit the same
# gap independently).
# liger-kernel: fused CE loss avoids trl's default fp32 lm_head upcast,
# a fixed ~4.74GiB allocation regardless of sequence/image size that OOM'd
# an L4 (found via an actual GCP smoke run — see docs/DECISIONS.md).
RUN pip install --no-cache-dir \
    "torch>=2.6" "torchvision" "transformers>=4.55" "peft>=0.13" "trl>=0.16" \
    "accelerate>=1.0" "datasets>=3.0" "pillow>=11" "huggingface_hub>=0.26" \
    "bitsandbytes>=0.44" "liger-kernel" "click>=8.1" "pydantic>=2.7" "numpy>=1.26"

COPY . .

# habeas_forge and habeas_model are uv "package = false" projects (no build
# backend, PYTHONPATH-based imports — matches the local test/CLI pattern in
# forge/tests/conftest.py and model/tests/conftest.py).
ENV PYTHONPATH=/workspace/model/src:/workspace/forge/src

CMD ["python", "-m", "habeas_model.train_cli"]
