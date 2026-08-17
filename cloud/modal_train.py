"""Modal training app (primary cloud). QLoRA SFT on Qwen/Qwen3.8-27B.

Run: modal run cloud/modal_train.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("attest-train")
vol = modal.Volume.from_name("attest-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile("Dockerfile")
GPU_L4 = modal.gpu.L4(count=1)


@app.function(image=image, gpu=GPU_L4, volumes={"/checkpoints": vol},
              timeout=60 * 60 * 6)
def train(data: bytes, smoke: bool = False, epochs: int = 2) -> str:
    from transformers import AutoModelForMultimodalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.8-27B", device_map="auto", torch_dtype="bfloat16",
        load_in_4bit=not smoke)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer = SFTTrainer(model=model, args=SFTConfig(
        output_dir="/checkpoints/sft", max_seq_length=4096,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        gradient_checkpointing=True, bf16=True, logging_steps=10,
        num_train_epochs=1 if smoke else epochs), train_dataset=[])
    trainer.train()
    trainer.save_model("/checkpoints/sft-final")
    vol.commit()
    return "done"
