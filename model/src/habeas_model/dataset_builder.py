"""SFT dataset builder: forge Task JSONL -> chat-format training records.

Reads `habeas_forge.schema.Task` records (forge/data/{train,val}.jsonl),
renders each task's document image, and emits one chat-format JSONL record
per task: system prompt (`habeas_model.schema.SYSTEM_PROMPT`), a user turn
describing the presented Form I-9 + documents (with the rendered page as a
base64 PNG for multimodal training), and an assistant turn holding the
oracle-derived ground-truth verdict as JSON matching `VerdictOut`.

Image rendering re-derives a representative page from `task.form` at build
time via `habeas_forge.generate.render_form` — it need not be byte-identical
to the pilot-generation-time render (same `form`, so same content; noise
level is carried over from `task.ocr_noise_level` for the difficulty this
task was drawn at). See docs/DECISIONS.md for the OCR-noise design.
"""

from __future__ import annotations

import base64
import json
import os
import random

import click

from habeas_forge.generate import render_form
from habeas_forge.schema import Task, Violation

from .schema import SYSTEM_PROMPT


def _load_tasks(path: str) -> list[Task]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Task.model_validate_json(line))
    return out


def _violation_out(v: Violation) -> dict:
    return {
        "type": v.type.value,
        "severity": v.severity.value,
        "field": v.field,
        "observed": v.observed,
        "expected": v.expected,
        "cfr": v.cfr,
        "correction": v.correction,
    }


def _user_content(task: Task, image_b64: str) -> str:
    form = task.form
    docs = "; ".join(
        f"List {d.list_type}: {d.doc_type} #{d.number} exp {d.expiration or 'n/a'}"
        for d in form.documents
    )
    lines = [
        f"Form I-9 edition: {form.edition}",
        f"Hire date: {form.hire_date}   Section 2 date: {form.section2_date}",
        f"Section 1 complete: {form.section1_complete}",
        f"Habeasation category: {form.habeasation_category}",
        f"Section 1 name/DOB: {form.name_section1} / {form.dob_section1}",
        f"Section 2 name/DOB: {form.name_section2} / {form.dob_section2}",
        f"Presented documents: {docs or 'none'}",
        f"Reverified: {form.reverified}   Work auth expiration: "
        f"{form.work_auth_expiration or 'n/a'}",
    ]
    if form.remote_examination:
        lines.append(
            f"Remote examination: E-Verify enrolled {form.everify_enrolled}, "
            f"copies retained {form.remote_copies_retained}"
        )
    lines.append(f"[document page image, base64 PNG, {len(image_b64)} chars omitted]")
    return "\n".join(lines)


def _render_rng(task: Task) -> random.Random:
    """Deterministic per-task RNG for image rendering, so the same task
    always renders to the same bytes (CONTRACTS.md §6: identical inputs
    across eval runs/models)."""
    return random.Random(int(task.signature[:16], 16))


def build_record(task: Task) -> dict:
    img = render_form(task.form, ocr_noise_level=task.ocr_noise_level,
                      rng=_render_rng(task))
    image_b64 = base64.b64encode(img).decode("ascii")
    target = {
        "verdict": task.expected.verdict,
        "violations": [_violation_out(v) for v in task.expected.violations],
    }
    return {
        "task_id": task.task_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(task, image_b64)},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ],
        "image_b64": image_b64,
    }


def build_dataset(tasks_path: str, out_path: str) -> int:
    tasks = _load_tasks(tasks_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(build_record(t), separators=(",", ":")) + "\n")
    return len(tasks)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--tasks-file", required=True)
@click.option("--out", required=True)
def build(tasks_file: str, out: str) -> None:
    n = build_dataset(tasks_file, out)
    click.echo(f"wrote {n} SFT records to {out}")


if __name__ == "__main__":
    cli()
