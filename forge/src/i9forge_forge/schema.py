"""I9forge canonical model. Mirrors CONTRACTS.md §1–§2.

A task is a FormI9 packet (Sections 1-3 + Supplement A/B) + presented
documents. The oracle applies M-274 / 8 CFR 274a.2 rules; the generator
builds a valid packet then injects N violations and runs the self-check gate.
All data is synthetic — no real PII ever.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.HIGH: 1.0,
    Severity.MEDIUM: 0.6,
    Severity.LOW: 0.3,
}


class ViolationType(str, Enum):
    COMBINATION_INVALID = "COMBINATION_INVALID"
    DOC_INVALID = "DOC_INVALID"
    DOC_EXPIRED = "DOC_EXPIRED"
    REVERIFICATION = "REVERIFICATION"
    TIMELINESS = "TIMELINESS"
    FIELD_INCOMPLETE = "FIELD_INCOMPLETE"
    EDITION_WRONG = "EDITION_WRONG"
    CATEGORY_MISMATCH = "CATEGORY_MISMATCH"
    DATA_INCONSISTENT = "DATA_INCONSISTENT"


VIOLATION_WEIGHTS: dict[ViolationType, float] = {
    ViolationType.COMBINATION_INVALID: 1.0,
    ViolationType.DOC_INVALID: 1.0,
    ViolationType.DOC_EXPIRED: 1.0,
    ViolationType.REVERIFICATION: 1.0,
    ViolationType.TIMELINESS: 0.6,
    ViolationType.FIELD_INCOMPLETE: 0.6,
    ViolationType.EDITION_WRONG: 0.6,
    ViolationType.CATEGORY_MISMATCH: 0.6,
    ViolationType.DATA_INCONSISTENT: 0.3,
}


class PresentedDoc(BaseModel):
    doc_type: str          # passport | green_card | ead | driver_license | ...
    list_type: str         # "A" | "B" | "C" (as presented in Section 2)
    number: str
    expiration: str | None  # ISO date or None (no expiration)


class FormI9(BaseModel):
    edition: str                 # "2023-08-01" | "2025-01-20"
    hire_date: str               # ISO
    section2_date: str           # ISO (when Section 2 was completed)
    section1_complete: bool
    attestation_category: str    # citizen | noncitizen_national | lpr | authorized
    documents: list[PresentedDoc] = Field(default_factory=list)
    reverified: bool
    work_auth_expiration: str | None
    name_section1: str
    name_section2: str
    dob_section1: str
    dob_section2: str


class Violation(BaseModel):
    type: ViolationType
    severity: Severity
    field: str
    observed: str
    expected: str
    cfr: str
    correction: str


class Verdict(BaseModel):
    verdict: str  # "PASS" | "FLAG"
    violations: list[Violation] = Field(default_factory=list)


class Task(BaseModel):
    task_id: str
    seed: int
    form: FormI9
    image_form_sha256: str
    expected: Verdict
    signature: str