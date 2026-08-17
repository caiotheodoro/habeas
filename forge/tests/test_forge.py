"""Attest forge tests: oracle, generator gate, scoring, contamination."""

import random

from attest_forge import contamination, generate
from attest_forge.schema import ViolationType, Verdict, Violation, Severity
from attest_forge.score import score_predictions, summarize
from attest_forge.verify import oracle_gate, verify


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


def test_contamination_roc():
    rng = random.Random(11)
    train = [generate.task(rng, seed=7, n_violations=1) for _ in range(20)]
    ev = [generate.task(rng, seed=8, n_violations=1) for _ in range(20)]
    index = contamination.build_train_index(train)
    assert contamination.probe(index, ev)["n_leaked"] == 0
    leaked_index = dict(index)
    leaked_index[ev[0].signature] = "LEAK"
    assert contamination.probe(leaked_index, ev)["n_leaked"] == 1