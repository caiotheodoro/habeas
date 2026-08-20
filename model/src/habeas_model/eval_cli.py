"""CLI: run benchmark_eval.run_eval against a local HF+LoRA Provider.

Plain path-based function (mirrors train_cli.py's shape) — used by
cloud/gcp_eval.sh to score the trained SFT adapter against
data/golden.jsonl before committing to an RLVR run against it.
"""

from __future__ import annotations

import json

import click

from .benchmark_eval import run_eval
from .local_provider import LocalHFProvider


def run(adapter_path: str, tasks_path: str, out_path: str,
       base_model: str = "Qwen/Qwen3.8-27B", max_new_tokens: int = 512) -> dict:
    provider = LocalHFProvider(adapter_path, base_model=base_model,
                               max_new_tokens=max_new_tokens)
    # max_workers=1: single GPU — concurrent generate() calls would just
    # serialize on the device anyway and risk overlapping CUDA contexts.
    return run_eval(tasks_path, provider, out_path, max_workers=1)


@click.command()
@click.option("--adapter", "adapter_path", required=True, help="SFT adapter dir")
@click.option("--tasks", "tasks_path", required=True, help="golden/eval task JSONL")
@click.option("--out", "out_path", required=True, help="per-task results JSONL (resumable)")
@click.option("--base-model", default="Qwen/Qwen3.8-27B")
@click.option("--max-new-tokens", default=512)
def main(adapter_path: str, tasks_path: str, out_path: str, base_model: str,
        max_new_tokens: int) -> None:
    summary = run(adapter_path, tasks_path, out_path, base_model=base_model,
                 max_new_tokens=max_new_tokens)
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
