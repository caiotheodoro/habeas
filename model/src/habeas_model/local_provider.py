"""Local HF inference Provider: base Qwen3.8-27B + QLoRA SFT adapter, for
running benchmark_eval.py against data/golden.jsonl on the same class of
GPU that trained the adapter. Reuses train_cli.py's exact model-loading
config (4-bit quantization_config, dtype="bfloat16", same base model id) —
a different quantization/dtype here would make eval numbers not comparable
to what was actually trained.
"""

from __future__ import annotations

import base64
import io


class LocalHFProvider:
    """Provider (see dataset_builder.Provider) backed by a local
    transformers model + PEFT adapter. One process, sequential generate()
    calls — run_eval's ThreadPoolExecutor concurrency doesn't help a
    single GPU and risks overlapping generate() calls under one CUDA
    context; callers should pass max_workers=1 to benchmark_eval.run_eval.
    """

    def __init__(self, adapter_path: str, base_model: str = "Qwen/Qwen3.8-27B",
                max_new_tokens: int = 512):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens
        self._torch = torch
        base = AutoModelForMultimodalLM.from_pretrained(
            base_model, device_map="auto", dtype="bfloat16",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(base_model)

    def complete(self, system: str, user: str, image_b64: str) -> str:
        from PIL import Image

        # Images go inline in the content list, not a top-level `images=`
        # kwarg — apply_chat_template's Jinja template only emits the
        # <|vision_start|><|image_pad|><|vision_end|> tokens for content
        # items carrying an "image"/"image_url" key or type=="image"; a
        # separate `images=` kwarg is not a recognized parameter and gets
        # silently misrouted into processor.__call__'s kwargs instead
        # (found via an actual eval run on GCP — TypeError from
        # Qwen3VLProcessor, see docs/DECISIONS.md).
        user_content: str | list[dict] = user
        if image_b64:
            img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
            user_content = [{"type": "image", "image": img}, {"type": "text", "text": user}]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(gen, skip_special_tokens=True)
