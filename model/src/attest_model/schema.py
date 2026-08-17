"""Model-output contract + parser for Attest (camelCase)."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


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
