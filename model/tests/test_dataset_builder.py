"""Tests for habeas_model.dataset_builder."""

import json
import random

from habeas_forge import generate
from habeas_model.dataset_builder import (build_dataset, build_record,
                                          build_rlvr_prompt, build_rlvr_prompts)


def test_build_record_matches_expected_verdict():
    rng = random.Random(1)
    task = generate.task(rng, seed=7, n_violations=1)
    record = build_record(task)
    assert record["task_id"] == task.task_id
    assert len(record["messages"]) == 3
    assert record["messages"][0]["role"] == "system"
    target = json.loads(record["messages"][2]["content"])
    assert target["verdict"] == task.expected.verdict
    assert len(target["violations"]) == len(task.expected.violations)
    assert record["image_b64"]


class _EchoOracleProvider:
    """Teacher stand-in that always answers with the exact oracle verdict —
    should always pass the verifier filter."""

    def __init__(self, verdict_json: str):
        self._raw = verdict_json

    def complete(self, system: str, user: str, image_b64: str) -> str:
        return self._raw


class _WrongProvider:
    """Teacher stand-in that always answers PASS with no violations —
    should be filtered out for any task with expected violations."""

    def complete(self, system: str, user: str, image_b64: str) -> str:
        return '{"verdict": "PASS", "violations": []}'


def test_build_record_teacher_source_exact_match_kept():
    rng = random.Random(6)
    task = generate.task(rng, seed=6, n_violations=1)
    raw = json.dumps({
        "verdict": task.expected.verdict,
        "violations": [
            {"type": v.type.value, "severity": v.severity.value, "field": v.field,
             "observed": v.observed, "expected": v.expected, "cfr": v.cfr,
             "correction": v.correction}
            for v in task.expected.violations
        ],
    })
    record = build_record(task, target_source="teacher", teacher=_EchoOracleProvider(raw))
    assert record is not None
    target = json.loads(record["messages"][2]["content"])
    assert target["verdict"] == task.expected.verdict


def test_build_record_teacher_source_mismatch_filtered():
    rng = random.Random(7)
    task = generate.task(rng, seed=7, n_violations=1)
    if task.expected.verdict == "PASS":
        return  # need a FLAG task for the mismatch to be detectable
    record = build_record(task, target_source="teacher", teacher=_WrongProvider())
    assert record is None


def test_build_record_teacher_source_duplicate_violation_undercount_filtered():
    """A teacher that reports only one instance of a duplicated violation
    type must be filtered out, not incorrectly kept — regression test for
    the multiset-vs-set exact-match bug found in code review."""
    from habeas_forge.schema import FormI9, PresentedDoc, Task as ForgeTask, Verdict, Violation, ViolationType, Severity

    v = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                  field="s1", observed="a", expected="b", cfr="c", correction="d")
    v2 = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                   field="s2", observed="a", expected="b", cfr="c", correction="d")
    expected = Verdict(verdict="FLAG", violations=[v, v2])  # duplicate type+severity

    class _UndercountProvider:
        def complete(self, system, user, image_b64):
            return json.dumps({
                "verdict": "FLAG",
                "violations": [{"type": "DOC_EXPIRED", "severity": "HIGH", "field": "s1",
                                "observed": "a", "expected": "b", "cfr": "c", "correction": "d"}],
            })  # only ONE of the two duplicated violations

    form = FormI9(
        edition="2023-08-01", hire_date="2026-08-03", section2_date="2026-08-05",
        section1_complete=True, habeasation_category="citizen",
        documents=[PresentedDoc(doc_type="passport", list_type="A", number="P1",
                                expiration="2020-01-01")],
        reverified=False, work_auth_expiration=None,
        name_section1="Test Person", name_section2="Test Person",
        dob_section1="1990-01-01", dob_section2="1990-01-01",
    )
    task = ForgeTask(task_id="t-dup", seed=1, form=form,
                     image_form_sha256="0" * 64, expected=expected,
                     signature="a" * 64)
    record = build_record(task, target_source="teacher", teacher=_UndercountProvider())
    assert record is None  # must be filtered, not incorrectly kept


def test_build_record_teacher_source_requires_provider():
    rng = random.Random(8)
    task = generate.task(rng, seed=8, n_violations=1)
    try:
        build_record(task, target_source="teacher")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_rlvr_prompt_has_no_assistant_turn():
    rng = random.Random(9)
    task = generate.task(rng, seed=9, n_violations=1)
    prompt = build_rlvr_prompt(task)
    assert prompt["task_id"] == task.task_id
    roles = [m["role"] for m in prompt["prompt"]]
    assert roles == ["system", "user"]  # no gold assistant trace
    assert json.loads(prompt["expected_verdict"])["verdict"] == task.expected.verdict
    assert len(prompt["images"]) == 1


def test_build_rlvr_prompts_writes_jsonl_distinct_from_sft(tmp_path):
    rng = random.Random(12)
    tasks_path = tmp_path / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for _ in range(3):
            t = generate.task(rng, seed=12, n_violations=1)
            f.write(t.model_dump_json() + "\n")
    out_path = tmp_path / "rlvr-prompts.jsonl"
    n = build_rlvr_prompts(str(tasks_path), str(out_path))
    assert n == 3
    for line in out_path.read_text().strip().splitlines():
        rec = json.loads(line)
        assert "prompt" in rec and "expected_verdict" in rec
        assert "messages" not in rec  # SFT-record shape, not this shape


def test_build_dataset_writes_jsonl(tmp_path):
    rng = random.Random(2)
    tasks_path = tmp_path / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for _ in range(5):
            t = generate.task(rng, seed=2, n_violations=1)
            f.write(t.model_dump_json() + "\n")
    out_path = tmp_path / "sft.jsonl"
    n = build_dataset(str(tasks_path), str(out_path))
    assert n == 5
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        rec = json.loads(line)
        assert "messages" in rec and "task_id" in rec


def test_build_dataset_teacher_concurrent_matches_sequential(tmp_path):
    # max_workers>1 must produce the same set of kept records as the
    # sequential path (only ordering may differ). Restricted to
    # n_violations=0 (PASS, empty violations) tasks so one shared canned
    # answer validly matches every task — keeps this test focused on the
    # concurrency/resumability wiring, not per-task filtering (already
    # covered by test_build_record_teacher_source_* above).
    rng = random.Random(10)
    pass_tasks = []
    while len(pass_tasks) < 4:
        t = generate.task(rng, seed=10, n_violations=0)
        if t.expected.verdict == "PASS" and not t.expected.violations:
            pass_tasks.append(t)
    tasks_path = tmp_path / "pass_tasks.jsonl"
    with open(tasks_path, "w") as f:
        for t in pass_tasks:
            f.write(t.model_dump_json() + "\n")
    provider = _EchoOracleProvider('{"verdict": "PASS", "violations": []}')

    out_seq = tmp_path / "seq.jsonl"
    out_conc = tmp_path / "conc.jsonl"
    n_seq = build_dataset(str(tasks_path), str(out_seq), target_source="teacher",
                          teacher=provider, max_workers=1)
    n_conc = build_dataset(str(tasks_path), str(out_conc), target_source="teacher",
                           teacher=provider, max_workers=4)
    assert n_seq == n_conc == 4
    seq_ids = {json.loads(l)["task_id"] for l in out_seq.read_text().strip().splitlines()}
    conc_ids = {json.loads(l)["task_id"] for l in out_conc.read_text().strip().splitlines()}
    assert seq_ids == conc_ids


def test_build_dataset_teacher_resumable(tmp_path):
    rng = random.Random(11)
    pass_tasks = []
    while len(pass_tasks) < 3:
        t = generate.task(rng, seed=11, n_violations=0)
        if t.expected.verdict == "PASS" and not t.expected.violations:
            pass_tasks.append(t)
    tasks_path = tmp_path / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for t in pass_tasks:
            f.write(t.model_dump_json() + "\n")
    out_path = tmp_path / "sft.jsonl"
    provider = _EchoOracleProvider('{"verdict": "PASS", "violations": []}')

    n1 = build_dataset(str(tasks_path), str(out_path), target_source="teacher",
                       teacher=provider, max_workers=2)
    assert n1 == 3
    # Re-running against the same out_path with the same tasks must write
    # nothing new — every task_id is already present (real-API-call cost
    # matters here; a restart after a transient failure shouldn't redo
    # already-completed work).
    n2 = build_dataset(str(tasks_path), str(out_path), target_source="teacher",
                       teacher=provider, max_workers=2)
    assert n2 == 0
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 3
