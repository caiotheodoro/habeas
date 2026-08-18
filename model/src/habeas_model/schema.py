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


SYSTEM_PROMPT = (
    "You are an I-9 compliance auditor. Given the Form I-9 and presented "
    "documents, emit JSON {\"verdict\": \"PASS\"|\"FLAG\", \"violations\": "
    "[{type, severity, field, observed, expected, cfr, correction}]} per "
    "M-274 Handbook and 8 CFR 274a.2. Emit only the JSON."
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
