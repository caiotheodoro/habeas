"""Attest verifier-as-oracle: M-274 Handbook + 8 CFR 274a.2 rules engine.

Sources: USCIS M-274 Handbook for Employers, 8 CFR 274a.2, eCFR (verify exact
citations in P1). Deterministic rules over the synthetic packet per
CONTRACTS.md §3.
"""

from __future__ import annotations

import datetime as dt

from .schema import FormI9, PresentedDoc, Severity, Verdict, Violation, ViolationType

VALID_EDITIONS = {"2023-08-01", "2025-01-20"}
VALID_CATEGORIES = {"citizen", "noncitizen_national", "lpr", "authorized"}

# doc_type -> acceptable List
DOC_LIST: dict[str, str] = {
    "passport": "A",
    "green_card": "A",
    "ead": "A",
    "foreign_passport_i94": "A",
    "parole": "A",
    "driver_license": "B",
    "state_id": "B",
    "military_id": "B",
    "ssn_card": "C",
    "birth_certificate": "C",
}

_SEV = {
    ViolationType.COMBINATION_INVALID: Severity.HIGH,
    ViolationType.DOC_INVALID: Severity.HIGH,
    ViolationType.DOC_EXPIRED: Severity.HIGH,
    ViolationType.REVERIFICATION: Severity.HIGH,
    ViolationType.TIMELINESS: Severity.MEDIUM,
    ViolationType.FIELD_INCOMPLETE: Severity.MEDIUM,
    ViolationType.EDITION_WRONG: Severity.MEDIUM,
    ViolationType.CATEGORY_MISMATCH: Severity.MEDIUM,
    ViolationType.DATA_INCONSISTENT: Severity.LOW,
}


def _v(t: ViolationType, field: str, observed: str, expected: str,
       cfr: str, correction: str) -> Violation:
    return Violation(type=t, severity=_SEV[t], field=field, observed=observed,
                     expected=expected, cfr=cfr, correction=correction)


def _add_business_days(d: dt.date, n: int) -> dt.date:
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def _biz_days_between(a: dt.date, b: dt.date) -> int:
    days = 0
    cur = a
    while cur < b:
        cur += dt.timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def verify(form: FormI9) -> Verdict:
    out: list[Violation] = []

    if form.edition not in VALID_EDITIONS:
        out.append(_v(ViolationType.EDITION_WRONG, "SECTION1.EDITION",
                      f"edition {form.edition}", "2023-08-01 or 2025-01-20",
                      "M-274 Ch.1 / 8 CFR 274a.2(a)",
                      "use the current I-9 edition"))
    if not form.section1_complete:
        out.append(_v(ViolationType.FIELD_INCOMPLETE, "SECTION1",
                      "required fields missing", "all Section 1 fields completed",
                      "8 CFR 274a.2(b)(1)(i)", "complete Section 1"))
    if form.name_section1 != form.name_section2 or form.dob_section1 != form.dob_section2:
        out.append(_v(ViolationType.DATA_INCONSISTENT, "SECTION1/2",
                      "name or DOB differs across sections",
                      "identical name and DOB in Sections 1 and 2",
                      "M-274 Ch.4", "reconcile the data-entry"))

    has_a = any(d.list_type == "A" for d in form.documents)
    has_b = any(d.list_type == "B" for d in form.documents)
    has_c = any(d.list_type == "C" for d in form.documents)
    if not (has_a or (has_b and has_c)):
        out.append(_v(ViolationType.COMBINATION_INVALID, "SECTION2.DOCUMENTS",
                      "no List A and no (List B + List C)",
                      "one List A OR one List B + one List C",
                      "8 CFR 274a.2(b)(1)(v)", "obtain an acceptable combination"))

    s2 = dt.date.fromisoformat(form.section2_date)
    for d in form.documents:
        valid_list = DOC_LIST.get(d.doc_type)
        if valid_list is None or d.list_type != valid_list:
            out.append(_v(ViolationType.DOC_INVALID, f"SECTION2.{d.doc_type}",
                          f"{d.doc_type} presented as List {d.list_type}",
                          f"List {valid_list or 'unknown'}",
                          "M-274 Ch.4 / 8 CFR 274a.2(b)(1)(v)",
                          "reclassify or replace the document"))
            continue
        if d.list_type in ("A", "B") and d.expiration:
            if dt.date.fromisoformat(d.expiration) < s2:
                out.append(_v(ViolationType.DOC_EXPIRED, f"SECTION2.{d.doc_type}",
                              f"expired {d.expiration}", "unexpired as of Section 2",
                              "8 CFR 274a.2(b)(1)(v)(A)", "obtain an unexpired document"))

    hire = dt.date.fromisoformat(form.hire_date)
    deadline = _add_business_days(hire, 3)
    if s2 > deadline:
        out.append(_v(ViolationType.TIMELINESS, "SECTION2.TIMELINESS",
                      f"Section 2 on {s2}, deadline {deadline}",
                      "Section 2 within 3 business days of hire",
                      "8 CFR 274a.2(b)(1)(ii)", "document the late completion"))

    if form.work_auth_expiration:
        if dt.date.fromisoformat(form.work_auth_expiration) < s2 and not form.reverified:
            out.append(_v(ViolationType.REVERIFICATION, "SECTION3",
                          "work authorization expired without reverification",
                          "Supplement B reverification recorded",
                          "8 CFR 274a.2(b)(1)(vii)", "record Supplement B reverification"))

    if form.attestation_category not in VALID_CATEGORIES:
        out.append(_v(ViolationType.CATEGORY_MISMATCH, "SECTION1.ATTESTATION",
                      f"invalid category '{form.attestation_category}'",
                      "citizen | noncitizen_national | lpr | authorized",
                      "8 CFR 274a.2(b)(1)(i)(A)", "correct the attestation"))

    return Verdict(verdict="PASS" if not out else "FLAG", violations=out)


def types_of(verdict: Verdict) -> set[ViolationType]:
    return {v.type for v in verdict.violations}


def oracle_gate(form: FormI9, injected: set[ViolationType]) -> bool:
    return types_of(verify(form)) == injected