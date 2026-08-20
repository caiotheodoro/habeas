"""Model-output contract + parser for Habeas (camelCase)."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from habeas_forge.schema import Severity, Verdict, Violation, ViolationType


class ViolationOut(BaseModel):
    type: str
    severity: str
    field: str = ""
    observed: str = ""
    expected: str = ""
    cfr: str = ""
    correction: str = ""


class VerdictOut(BaseModel):
    verdict: str
    violations: list[ViolationOut] = Field(default_factory=list)


# SFT training targets and RLVR expected_verdict are always built from
# habeas_forge's own ViolationType/Severity enum values (see
# dataset_builder.build_record), so the trained model implicitly learns
# this closed vocabulary from its own training data even without it being
# spelled out here. An external teacher model (no exposure to that
# training data — e.g. dataset_builder's target_source="teacher" path via
# vertex_provider.VertexProvider) has no such implicit signal: a live test
# against Gemini 2.5 Flash returned invented labels ("Data Error"/"High"
# instead of "EDITION_WRONG"/"MEDIUM") without this enumeration, which
# to_forge_verdict's unrecognized-type-dropping would silently discard,
# making the verifier-filter reject nearly every teacher trace. Built from
# the enums directly (not a hand-copied list) so it can't drift out of
# sync with habeas_forge.schema.
_TYPE_LIST = "|".join(t.value for t in ViolationType)
_SEVERITY_LIST = "|".join(s.value for s in Severity)
SYSTEM_PROMPT = (
    "You are an I-9 compliance auditor. Given the Form I-9 and presented "
    "documents, emit JSON {\"verdict\": \"PASS\"|\"FLAG\", \"violations\": "
    "[{type, severity, field, observed, expected, cfr, correction}]} per "
    "M-274 Handbook and 8 CFR 274a.2. "
    f"type must be exactly one of: {_TYPE_LIST}. "
    f"severity must be exactly one of: {_SEVERITY_LIST}. "
    "Emit only the JSON."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse(raw: str) -> VerdictOut | None:
    if not raw:
        return None
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return VerdictOut.model_validate(json.loads(m.group(0)))
    except Exception:
        return None


def to_forge_verdict(raw: str) -> Verdict | None:
    """Raw model text -> a forge Verdict, or None if unparseable.

    The canonical model-output-text -> forge-Verdict converter, shared by
    benchmark_eval (scoring), dataset_builder (teacher-trace filtering),
    and habeas_forge.score.reward (RLVR) via habeas_model.rlvr_reward —
    single source of truth for "how do we turn raw model text into a
    scoreable Verdict." Unrecognized violation type/severity strings are
    dropped individually rather than failing the whole parse.
    """
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
