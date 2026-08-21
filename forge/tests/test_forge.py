"""Habeas forge tests: oracle, generator gate, scoring, contamination."""

import random

from habeas_forge import contamination, generate
from habeas_forge.schema import ViolationType, Verdict, Violation, Severity
from habeas_forge.score import reward, score_predictions, summarize
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


def test_citation_exact_match_correct_cfr():
    v = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                  field="x", observed="o", expected="e",
                  cfr="8 CFR 274a.2(b)(1)(vi)", correction="")
    exp = Verdict(verdict="FLAG", violations=[v])
    predicted = Verdict(verdict="FLAG", violations=[v])
    r = score_predictions(exp, predicted)
    assert r["cfr_correct"] == 1.0 and r["cfr_total"] == 1.0
    assert summarize([r])["citation_exact_match"] == 1.0


def test_citation_exact_match_wrong_cfr_still_counts_as_caught():
    v_exp = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                      field="x", observed="o", expected="e",
                      cfr="8 CFR 274a.2(b)(1)(vi)", correction="")
    v_pred = v_exp.model_copy(update={"cfr": "8 CFR 274a.2(b)(1)(ii)"})
    exp = Verdict(verdict="FLAG", violations=[v_exp])
    predicted = Verdict(verdict="FLAG", violations=[v_pred])
    r = score_predictions(exp, predicted)
    # Type+severity still matched (counts toward severity_weighted_recall),
    # but the citation itself is wrong — must not count as a citation match.
    assert r["caught"] == r["total"]
    assert r["cfr_correct"] == 0.0 and r["cfr_total"] == 1.0
    assert summarize([r])["citation_exact_match"] == 0.0


def test_citation_exact_match_denominator_excludes_missed_and_false_positive_violations():
    v_exp = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                      field="x", observed="o", expected="e", cfr="cfr-a", correction="")
    v_fp = Violation(type=ViolationType.TIMELINESS, severity=Severity.MEDIUM,
                     field="y", observed="o", expected="e", cfr="cfr-b", correction="")
    exp = Verdict(verdict="FLAG", violations=[v_exp])
    # Predicted misses v_exp entirely and adds an unrelated false positive —
    # neither has a "caught" instance, so the citation denominator is 0.
    predicted = Verdict(verdict="FLAG", violations=[v_fp])
    r = score_predictions(exp, predicted)
    assert r["cfr_total"] == 0.0 and r["cfr_correct"] == 0.0
    assert summarize([r])["citation_exact_match"] == 1.0  # no denom -> vacuous 1.0, not 0


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


def test_scoring_verdict_consistency():
    v = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                  field="x", observed="o", expected="e", cfr="", correction="")
    exp_flag = Verdict(verdict="FLAG", violations=[v])
    exp_pass = Verdict(verdict="PASS", violations=[])

    # instance-level recall is perfect (violation matched) but the model's
    # own top-level verdict string contradicts its violation list
    self_contradictory = score_predictions(exp_flag, Verdict(verdict="PASS", violations=[v]))
    assert self_contradictory["caught"] == self_contradictory["total"]
    assert self_contradictory["verdict_correct"] == 0.0

    correct = score_predictions(exp_flag, Verdict(verdict="FLAG", violations=[v]))
    assert correct["verdict_correct"] == 1.0

    # FLAG with no violations listed on an expected-clean packet
    hallucinated_flag = score_predictions(exp_pass, Verdict(verdict="FLAG", violations=[]))
    assert hallucinated_flag["verdict_correct"] == 0.0

    r = summarize([self_contradictory, correct])
    assert r["verdict_accuracy"] == 0.5


def test_reward_general_case():
    v = Violation(type=ViolationType.DOC_EXPIRED, severity=Severity.HIGH,
                  field="x", observed="o", expected="e", cfr="", correction="")
    exp = Verdict(verdict="FLAG", violations=[v])
    perfect = reward(exp, Verdict(verdict="FLAG", violations=[v]))
    assert perfect == 1.0 + 0.2  # recall=1, no fp, verdict_bonus

    missed = reward(exp, Verdict(verdict="PASS", violations=[]))
    assert missed < perfect

    false_positive = reward(exp, Verdict(verdict="FLAG", violations=[
        v, Violation(type=ViolationType.TIMELINESS, severity=Severity.MEDIUM,
                    field="y", observed="o", expected="e", cfr="", correction=""),
    ]))
    assert false_positive == 1.0 - 0.3 + 0.2  # recall=1, one fp, verdict correct


def test_reward_clean_pass_case():
    exp = Verdict(verdict="PASS", violations=[])
    assert reward(exp, Verdict(verdict="PASS", violations=[])) == 1.0
    assert reward(exp, Verdict(verdict="FLAG", violations=[])) == -1.0


def test_reward_unparseable():
    exp = Verdict(verdict="FLAG", violations=[])
    assert reward(exp, None) == -1.0


def test_contamination_roc():
    rng = random.Random(11)
    train = [generate.task(rng, seed=7, n_violations=1) for _ in range(20)]
    ev = [generate.task(rng, seed=8, n_violations=1) for _ in range(20)]
    index = contamination.build_train_index(train)
    assert contamination.probe(index, ev)["n_leaked"] == 0
    leaked_index = dict(index)
    leaked_index[ev[0].signature] = "LEAK"
    assert contamination.probe(leaked_index, ev)["n_leaked"] == 1