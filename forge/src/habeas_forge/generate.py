"""Seeded I-9 packet generator with violation injection + oracle gate.

Builds a valid packet, injects N violations, renders the form to an image,
and emits a Task with a contamination signature. Synthetic only. See
CONTRACTS.md §3–§5.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .schema import FormI9, PresentedDoc, Task, ViolationType
from .verify import DOC_LIST, _add_business_days, oracle_gate, verify

ALL = list(ViolationType)
FIRST = ["Ana", "Liam", "Sofia", "Noah", "Maya", "Elias", "Zoe", "Omar"]


def _valid_packet(rng: random.Random) -> FormI9:
    name = f"{rng.choice(FIRST)} {rng.choice(FIRST)}son"
    dob = "1992-04-17"
    hire = dt.date(2026, 8, 3)
    s2 = _add_business_days(hire, 2)
    edition = rng.choice(["2023-08-01", "2025-01-20"])
    use_list_a = rng.random() < 0.7
    if use_list_a:
        docs = [PresentedDoc(doc_type="passport", list_type="A",
                             number=f"P{rng.randint(10**8, 10**9)}",
                             expiration="2031-01-01")]
    else:
        docs = [PresentedDoc(doc_type="driver_license", list_type="B",
                             number=f"D{rng.randint(10**7, 10**8)}",
                             expiration="2029-05-01"),
                PresentedDoc(doc_type="ssn_card", list_type="C",
                             number=f"{rng.randint(10**8, 10**9)}", expiration=None)]
    category = "citizen" if rng.random() < 0.6 else "authorized"
    work_exp = None if category == "citizen" else "2028-12-31"
    remote = rng.random() < 0.15
    return FormI9(
        edition=edition, hire_date=hire.isoformat(), section2_date=s2.isoformat(),
        section1_complete=True, habeasation_category=category, documents=docs,
        reverified=work_exp is not None and dt.date.fromisoformat(work_exp) < s2,
        work_auth_expiration=work_exp,
        name_section1=name, name_section2=name,
        dob_section1=dob, dob_section2=dob,
        remote_examination=remote,
        everify_enrolled=remote,
        remote_copies_retained=True,
    )


def generate_packet(rng: random.Random,
                    injections: list[ViolationType] | None = None,
                    difficulty: float = 0.5) -> FormI9:
    if injections is None:
        injections = []
    form = _valid_packet(rng)
    inj = set(injections)
    s2 = dt.date.fromisoformat(form.section2_date)

    if ViolationType.EDITION_WRONG in inj:
        form.edition = "2021-10-21"
    if ViolationType.FIELD_INCOMPLETE in inj:
        form.section1_complete = False
    if ViolationType.DATA_INCONSISTENT in inj:
        form.name_section2 = form.name_section1 + " Jr."
    if ViolationType.CATEGORY_MISMATCH in inj:
        form.habeasation_category = "undocumented"
    if ViolationType.COMBINATION_INVALID in inj:
        form.documents = [d for d in form.documents if d.list_type == "B"]
        if not form.documents:
            form.documents = [PresentedDoc(
                doc_type="driver_license", list_type="B",
                number=f"D{rng.randint(10**7, 10**8)}", expiration="2029-05-01")]
    if ViolationType.DOC_INVALID in inj and form.documents:
        form.documents[0].list_type = "B" if form.documents[0].list_type != "B" else "A"
    if ViolationType.DOC_EXPIRED in inj:
        for d in form.documents:
            if d.list_type in ("A", "B") and d.expiration:
                d.expiration = "2025-01-01"
    if ViolationType.TIMELINESS in inj:
        form.section2_date = (_add_business_days(dt.date.fromisoformat(form.hire_date), 6)).isoformat()
    if ViolationType.REVERIFICATION in inj:
        form.work_auth_expiration = "2025-06-01"
        form.reverified = False
    if ViolationType.REMOTE_EXAM_INVALID in inj:
        form.remote_examination = True
        if rng.random() < 0.5:
            form.everify_enrolled = False
            form.remote_copies_retained = True
        else:
            form.everify_enrolled = True
            form.remote_copies_retained = False

    return form


def task(rng: random.Random, seed: int, n_violations: int = 1,
         difficulty: float = 0.5, max_tries: int = 12) -> Task:
    for _ in range(max_tries):
        inj = [ViolationType(rng.choice(ALL)) for _ in range(n_violations)]
        form = generate_packet(rng, inj, difficulty)
        if oracle_gate(form, set(inj)):
            noise = max(0.0, min(1.0, difficulty)) * 0.6
            img = render_form(form, ocr_noise_level=noise, rng=rng)
            return Task(
                task_id=f"i9-{seed}-{rng.randint(0, 10**9):09d}",
                seed=seed, form=form,
                image_form_sha256=hashlib.sha256(img).hexdigest(),
                expected=verify(form),
                ocr_noise_level=noise,
                signature=signature(form),
            )
    raise RuntimeError(f"oracle gate failed n_violations={n_violations}")


def signature(form: FormI9) -> str:
    return hashlib.sha256(
        json.dumps(form.model_dump(mode="json"), sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()


def render_form(form: FormI9, ocr_noise_level: float = 0.0,
                rng: random.Random | None = None) -> bytes:
    img = Image.new("RGB", (700, 920), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    y = 20
    d.text((20, y), "FORM I-9  Employment Eligibility Verification", fill="black", font=font); y += 24
    d.text((20, y), f"Edition: {form.edition}", fill="black", font=small); y += 20
    d.text((20, y), f"Section 1 name: {form.name_section1}", fill="black", font=small); y += 16
    d.text((20, y), f"DOB: {form.dob_section1}   Habeasation: {form.habeasation_category}", fill="black", font=small); y += 16
    d.text((20, y), f"Complete: {form.section1_complete}", fill="black", font=small); y += 22
    d.text((20, y), "SECTION 2", fill="black", font=font); y += 20
    d.text((20, y), f"Hire date: {form.hire_date}   Section 2 date: {form.section2_date}", fill="black", font=small); y += 16
    for doc in form.documents:
        d.text((24, y), f"List {doc.list_type}: {doc.doc_type}  #{doc.number}  exp {doc.expiration or 'n/a'}", fill="black", font=small); y += 16
    y += 6
    d.text((20, y), f"Section 3 name: {form.name_section2}   DOB: {form.dob_section2}", fill="black", font=small); y += 16
    d.text((20, y), f"Reverified: {form.reverified}   Work auth exp: {form.work_auth_expiration or 'n/a'}", fill="black", font=small); y += 16
    if form.remote_examination:
        d.text((20, y), f"Remote exam: E-Verify enrolled {form.everify_enrolled}   "
                        f"copies retained {form.remote_copies_retained}",
               fill="black", font=small); y += 16
    if ocr_noise_level > 0:
        img = _apply_ocr_noise(img, ocr_noise_level, rng or random.Random())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _apply_ocr_noise(img: Image.Image, level: float, rng: random.Random) -> Image.Image:
    """Rendering-layer-only perturbation. Never touches FormI9 fields, so the
    oracle gate (which traces the ground-truth form, not pixels) is unaffected.
    """
    level = max(0.0, min(1.0, level))
    arr = np.array(img).astype(np.float32)
    arr += np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, 18 * level, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if level > 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=level * 1.2))
    if level > 0.5:
        img = img.rotate(rng.uniform(-1.5, 1.5) * level, fillcolor="white", expand=False)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(35, int(95 - 60 * level)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")