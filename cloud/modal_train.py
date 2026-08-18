"""Modal training app (primary cloud). QLoRA SFT on Qwen/Qwen3.8-27B.

Run: modal run cloud/modal_train.py --smoke

`data` must be an SFT-record JSONL (see `habeas_model.dataset_builder`).
The actual training logic lives in `habeas_model.train_cli.run_sft` —
shared with the GCP spot fallback (`cloud/gcp_spot.sh`) so the two cloud
paths can't silently diverge (docs/TRAINING_PLAN.md §1).
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("habeas-train")
vol = modal.Volume.from_name("habeas-checkpoints", create_if_missing=True)
image = modal.Image.from_dockerfile(str(Path(__file__).parent / "Dockerfile"))


@app.function(image=image, gpu="L4", volumes={"/checkpoints": vol},
              timeout=60 * 60 * 6)
def train(data: bytes, smoke: bool = False, epochs: int = 2) -> str:
    import tempfile

    from habeas_model.train_cli import run_sft

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False) as f:
        f.write(data)
        tmp_path = f.name

    run_sft(tmp_path, "/checkpoints/sft", smoke=smoke, epochs=epochs)
    vol.commit()
    return "done"


@app.local_entrypoint()
def main(data: str = "model/data/sft-train.jsonl", smoke: bool = False,
         epochs: int = 2) -> None:
    """`data` is a local path to an SFT-record JSONL (built via
    `habeas_model.dataset_builder build`), read and shipped to the remote
    function as bytes."""
    with open(data, "rb") as f:
        payload = f.read()
    result = train.remote(data=payload, smoke=smoke, epochs=epochs)
    print(result)
