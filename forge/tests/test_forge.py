"""Habeas forge tests: oracle, generator gate, scoring, contamination."""

import random

from habeas_forge import contamination, generate
from habeas_forge.schema import ViolationType, Verdict, Violation, Severity
from habeas_forge.score import score_predictions, summarize
from habeas_forge.verify import oracle_gate, verify


def test_pass_packet_fires_nothing():
    rng = random.Random(1)
    t = generate.task(rng, seed=7, n_violations=0)
    assert t.expected.verdict == "PASS"
    assert oracle_gate(t.form, set())


def test_every_violation_type_reachable():
    rng = random.Random(42)
    seen = set()
    for _ in range(600):
        t = generate.task(rng, seed=42, n_violations=1)
        got = {v.type for v in t.expected.violations}
        seen |= got
        assert oracle_gate(t.form, got)
    for vt in ViolationType:
        assert vt in seen, vt


def test_scoring():
    v = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                  field="x", observed="o", expected="e", cfr="", correction="")
    exp = Verdict(verdict="FLAG", violations=[v])
    r = summarize([score_predictions(exp, Verdict(verdict="FLAG", violations=[v])),
                   score_predictions(exp, Verdict(verdict="PASS", violations=[]))])
    assert r["severity_weighted_recall"] == 0.5


def test_deterministic_same_seed():
    rng1, rng2 = random.Random(123), random.Random(123)
    t1 = generate.task(rng1, seed=7, n_violations=2)
    t2 = generate.task(rng2, seed=7, n_violations=2)
    assert t1.signature == t2.signature
    assert verify(t1.form) == verify(t2.form)


def test_remote_exam_gate():
    rng = random.Random(99)
    form = generate.generate_packet(rng, [])
    form.remote_examination = True
    form.everify_enrolled = True
    form.remote_copies_retained = True
    assert oracle_gate(form, set())
    assert verify(form).verdict == "PASS"

    rng2 = random.Random(99)
    seen = set()
    for _ in range(200):
        t = generate.task(rng2, seed=99, n_violations=1)
        seen |= {v.type for v in t.expected.violations}
        assert oracle_gate(t.form, {v.type for v in t.expected.violations})
        if ViolationType.REMOTE_EXAM_INVALID in seen:
            break
    assert ViolationType.REMOTE_EXAM_INVALID in seen


def test_ocr_noise_does_not_affect_oracle():
    rng_clean, rng_noisy = random.Random(55), random.Random(55)
    t_clean = generate.task(rng_clean, seed=55, n_violations=1, difficulty=0.0)
    t_noisy = generate.task(rng_noisy, seed=55, n_violations=1, difficulty=1.0)
    assert t_clean.form == t_noisy.form
    assert t_clean.signature == t_noisy.signature
    assert t_clean.expected == t_noisy.expected
    assert t_noisy.ocr_noise_level > t_clean.ocr_noise_level


def test_ocr_noise_produces_valid_image():
    rng = random.Random(3)
    t = generate.task(rng, seed=3, n_violations=0, difficulty=1.0)
    assert t.ocr_noise_level > 0
    assert len(t.image_form_sha256) == 64


def test_contamination_roc():
    rng = random.Random(11)
    train = [generate.task(rng, seed=7, n_violations=1) for _ in range(20)]
    ev = [generate.task(rng, seed=8, n_violations=1) for _ in range(20)]
    index = contamination.build_train_index(train)
    assert contamination.probe(index, ev)["n_leaked"] == 0
    leaked_index = dict(index)
    leaked_index[ev[0].signature] = "LEAK"
    assert contamination.probe(leaked_index, ev)["n_leaked"] == 1