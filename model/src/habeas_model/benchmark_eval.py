"""Head-to-head benchmark eval: provider adapter -> forge oracle scoring.

Per CONTRACTS.md §6: same system prompt for every model, non-thinking verdict
output, frontier scored with identical inputs + scoring code, unparseable =
miss, concurrent eval with JSONL checkpointing (resumable).

`Provider` is the minimal adapter surface a caller wires in (local vLLM/MLX,
or a frontier API client) — this module owns prompting, parsing, concurrency,
checkpointing, and scoring; it never calls a specific vendor SDK directly.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from habeas_forge.schema import Severity, Task, Verdict, Violation, ViolationType
from habeas_forge.score import score_predictions, summarize

from .dataset_builder import _user_content
from .schema import SYSTEM_PROMPT, parse


class Provider(Protocol):
    """Minimal adapter: raw model completion for one task."""

    def complete(self, system: str, user: str, image_b64: str) -> str:
        ...


def _to_forge_verdict(raw: str) -> Verdict | None:
    out = parse(raw)
    if out is None:
        return None
    violations: list[Violation] = []
    for v in out.violations:
        try:
            violations.append(Violation(
                type=ViolationType(v.type), severity=Severity(v.severity),
                field=v.field, observed=v.observed, expected=v.expected,
                cfr=v.cfr, correction=v.correction,
            ))
        except ValueError:
            continue  # unrecognized type/severity: dropped, not scored as a match
    verdict = out.verdict if out.verdict in ("PASS", "FLAG") else "FLAG"
    return Verdict(verdict=verdict, violations=violations)


def _predict_one(task: Task, provider: Provider) -> Verdict | None:
    img = None
    try:
        from .dataset_builder import build_record
        img = build_record(task)["image_b64"]
    except Exception:
        img = ""
    user = _user_content(task, img)
    raw = provider.complete(SYSTEM_PROMPT, user, img)
    return _to_forge_verdict(raw)


def _load_tasks(path: str) -> list[Task]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(Task.model_validate_json(line))
    return out


def _load_checkpoint(out_path: str) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done[rec["task_id"]] = rec
    return done


def run_eval(tasks_path: str, provider: Provider, out_path: str,
            max_workers: int = 8) -> dict[str, float]:
    """Concurrent eval with JSONL checkpointing (resumable): re-running with
    the same out_path skips already-scored task_ids.
    """
    tasks = _load_tasks(tasks_path)
    done = _load_checkpoint(out_path)
    pending = [t for t in tasks if t.task_id not in done]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_predict_one, t, provider): t for t in pending}
            for fut in as_completed(futures):
                t = futures[fut]
                predicted = fut.result()
                rec = {
                    "task_id": t.task_id,
                    "predicted": predicted.model_dump(mode="json") if predicted else None,
                }
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
                done[t.task_id] = rec

    results = []
    for t in tasks:
        rec = done[t.task_id]
        predicted = Verdict.model_validate(rec["predicted"]) if rec["predicted"] else None
        results.append(score_predictions(t.expected, predicted))
    return summarize(results)
