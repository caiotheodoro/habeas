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
from collections import Counter
from typing import Literal, Protocol

import click

from habeas_forge.generate import render_form
from habeas_forge.schema import Task, Violation

from .schema import SYSTEM_PROMPT, to_forge_verdict


class Provider(Protocol):
    """Minimal adapter: raw model completion for one task."""

    def complete(self, system: str, user: str, image_b64: str) -> str:
        ...


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


def build_record(task: Task, target_source: Literal["oracle", "teacher"] = "oracle",
                 teacher: Provider | None = None) -> dict | None:
    """Build one SFT chat-format record.

    target_source="oracle" (default): assistant turn is the ground-truth
    oracle Verdict — today's behavior, unchanged.

    target_source="teacher": assistant turn is a teacher model's own
    response, **verifier-filtered** per docs/methodology.md ("SFT
    cold-start on verifier-filtered traces distilled from a stronger
    model") — kept only if the teacher's parsed Verdict is a true exact
    match: same verdict string, and the *multiset* of (type, severity)
    violations matches the oracle's exactly (not just set-membership —
    `score_predictions`'s caught/total/fp are severity-weighted sums over
    a *set* of (type, severity) pairs, so they alone can't detect a
    teacher trace that drops one instance of a duplicated violation type;
    see docs/DECISIONS.md). Returns None (caller drops the task) on any
    mismatch.
    """
    img = render_form(task.form, ocr_noise_level=task.ocr_noise_level,
                      rng=_render_rng(task))
    image_b64 = base64.b64encode(img).decode("ascii")
    user_content = _user_content(task, image_b64)

    if target_source == "oracle":
        target_verdict = task.expected
    else:
        if teacher is None:
            raise ValueError('teacher provider required when target_source="teacher"')
        raw = teacher.complete(SYSTEM_PROMPT, user_content, image_b64)
        predicted = to_forge_verdict(raw)
        if predicted is None or predicted.verdict != task.expected.verdict:
            return None
        exp_counts = Counter((v.type, v.severity) for v in task.expected.violations)
        pred_counts = Counter((v.type, v.severity) for v in predicted.violations)
        if exp_counts != pred_counts:
            return None
        target_verdict = predicted

    target = {
        "verdict": target_verdict.verdict,
        "violations": [_violation_out(v) for v in target_verdict.violations],
    }
    return {
        "task_id": task.task_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ],
        "image_b64": image_b64,
    }


def _already_built_task_ids(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    ids = set()
    with open(out_path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["task_id"])
    return ids


def build_dataset(tasks_path: str, out_path: str,
                  target_source: Literal["oracle", "teacher"] = "oracle",
                  teacher: Provider | None = None, max_workers: int = 1) -> int:
    """Build an SFT-record JSONL from a task file.

    `target_source="teacher"` with `max_workers > 1`: concurrent + append-
    resumable, same pattern as `benchmark_eval.run_eval` — each teacher
    call is an independent network round-trip (VertexProvider), so a
    ~1600-task corpus at sequential ~3-10s/call would take hours; a
    thread pool is the same fix `run_eval` already applies for exactly
    this reason. Resumable: re-running with the same `out_path` skips
    task_ids already written (real API calls, real cost — don't redo work
    on a restart after a transient failure). `target_source="oracle"`
    stays the original fast sequential path (no network calls, no need
    for either concurrency or resume).
    """
    tasks = _load_tasks(tasks_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if target_source == "oracle" or max_workers <= 1:
        n_written = 0
        mode = "a" if target_source == "teacher" else "w"
        done = _already_built_task_ids(out_path) if mode == "a" else set()
        with open(out_path, mode) as f:
            for t in tasks:
                if t.task_id in done:
                    continue
                record = build_record(t, target_source=target_source, teacher=teacher)
                if record is None:
                    continue  # verifier-filtered out (teacher trace didn't match oracle)
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                n_written += 1
        return n_written

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = _already_built_task_ids(out_path)
    pending = [t for t in tasks if t.task_id not in done]
    n_written = 0
    with open(out_path, "a") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(build_record, t, target_source, teacher): t
                      for t in pending}
            for fut in as_completed(futures):
                record = fut.result()
                if record is None:
                    continue  # verifier-filtered out
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                f.flush()
                n_written += 1
    return n_written


def build_rlvr_prompt(task: Task) -> dict:
    """RLVR prompt record: system+user turns only, no assistant target —
    that's what GRPO generates. Structurally distinct from build_record()'s
    SFT record (no gold trace at all), so it cannot leak into SFT training
    and RLVR prompts can never be pointed at the SFT trace JSONL by
    accident — satisfies methodology.md's "RLVR data never mixed into SFT."
    """
    img = render_form(task.form, ocr_noise_level=task.ocr_noise_level,
                      rng=_render_rng(task))
    image_b64 = base64.b64encode(img).decode("ascii")
    return {
        "task_id": task.task_id,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_content(task, image_b64)},
        ],
        "images": [image_b64],
        "expected_verdict": task.expected.model_dump_json(),
        # Qwen3's chat template defaults enable_thinking to True unless
        # explicitly disabled — trl's apply_chat_template forwards a
        # per-example "chat_template_kwargs" dict straight through to the
        # renderer (see trl/data_utils.py). Without this, GRPO rollouts
        # would burn max_completion_length on CoT prose instead of the
        # trained JSON-only verdict, same as the eval bug found live on
        # GCP (see docs/DECISIONS.md) — fixed here proactively instead of
        # rediscovering it after burning a real RLVR run on it.
        "chat_template_kwargs": {"enable_thinking": False},
    }


def build_rlvr_prompts(tasks_path: str, out_path: str) -> int:
    tasks = _load_tasks(tasks_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(build_rlvr_prompt(t), separators=(",", ":")) + "\n")
    return len(tasks)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--tasks-file", required=True)
@click.option("--out", required=True)
@click.option("--target-source", type=click.Choice(["oracle", "teacher"]), default="oracle")
@click.option("--teacher-project", default=None,
             help="GCP project for Vertex AI (required if --target-source=teacher).")
@click.option("--teacher-location", default="us-central1")
@click.option("--teacher-model", default="gemini-2.5-flash")
@click.option("--max-workers", default=1,
             help="Concurrent teacher calls (ignored for --target-source=oracle).")
def build(tasks_file: str, out: str, target_source: str, teacher_project: str | None,
         teacher_location: str, teacher_model: str, max_workers: int) -> None:
    teacher = None
    if target_source == "teacher":
        if not teacher_project:
            raise click.UsageError("--teacher-project is required when --target-source=teacher")
        from .vertex_provider import VertexProvider
        teacher = VertexProvider(project=teacher_project, location=teacher_location,
                                 model=teacher_model)
    n = build_dataset(tasks_file, out, target_source=target_source, teacher=teacher,
                      max_workers=max_workers)
    click.echo(f"wrote {n} SFT records to {out} (target_source={target_source})")


@cli.command()
@click.option("--tasks-file", required=True)
@click.option("--out", required=True)
def prompts(tasks_file: str, out: str) -> None:
    n = build_rlvr_prompts(tasks_file, out)
    click.echo(f"wrote {n} RLVR prompts to {out}")


if __name__ == "__main__":
    cli()
