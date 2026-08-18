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

from habeas_forge.schema import Task, Verdict
from habeas_forge.score import score_predictions, summarize

from .dataset_builder import Provider, _user_content, build_record
from .schema import SYSTEM_PROMPT, to_forge_verdict

# Back-compat alias: to_forge_verdict moved to habeas_model.schema (shared
# by dataset_builder's teacher-trace filtering too) so both modules use one
# raw-text -> Verdict converter instead of duplicating the parse logic.
_to_forge_verdict = to_forge_verdict


def _predict_one(task: Task, provider: Provider) -> Verdict | None:
    img = None
    try:
        img = build_record(task)["image_b64"]
    except Exception:
        img = ""
    user = _user_content(task, img)
    raw = provider.complete(SYSTEM_PROMPT, user, img)
    return to_forge_verdict(raw)


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
