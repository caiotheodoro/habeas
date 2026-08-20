"""Vertex AI (Gemini) Provider — the fast teacher-completion backend for
`dataset_builder.build_record(target_source="teacher", ...)`.

Chosen over a frontier API key (none configured in this environment) and
over `opencode` (the only already-authenticated live-model path, but
agentic/slow — 15-40 min/call observed, infeasible at the hundreds-of-calls
volume teacher distillation needs). This project already has a working
GCP account/project from the SFT training work this session, and Vertex
AI's `aiplatform.googleapis.com` API is already enabled there — reusing
that avoids any new credential setup.

Auth: shells out to `gcloud auth print-access-token` rather than needing
a separate `gcloud auth application-default login` (which requires an
interactive browser flow this environment can't do) — the same `gcloud`
CLI already used all session for Compute Engine is sufficient. Access
tokens are short-lived (~1hr); refreshed automatically here.
"""

from __future__ import annotations

import base64
import subprocess
import time

import requests


class VertexProvider:
    """Provider (see dataset_builder.Provider) backed by Vertex AI's
    Gemini generateContent REST API. One process, sequential HTTP calls
    (teacher distillation runs at task-corpus scale, not training-loop
    scale — no need for GPU-aware batching here).
    """

    def __init__(self, project: str, location: str = "us-central1",
                model: str = "gemini-2.5-flash", timeout: float = 60.0):
        self.project = project
        self.location = location
        self.model = model
        self.timeout = timeout
        self._token: str | None = None
        self._token_fetched_at: float = 0.0

    def _access_token(self) -> str:
        # Refresh every 45 min — tokens are valid ~1hr, this leaves margin
        # without shelling out to gcloud on every single call.
        if self._token is None or (time.time() - self._token_fetched_at) > 45 * 60:
            out = subprocess.run(["gcloud", "auth", "print-access-token"],
                                 capture_output=True, text=True, check=True)
            self._token = out.stdout.strip()
            self._token_fetched_at = time.time()
        return self._token

    def complete(self, system: str, user: str, image_b64: str) -> str:
        url = (f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
              f"{self.project}/locations/{self.location}/publishers/google/"
              f"models/{self.model}:generateContent")
        parts: list[dict] = [{"text": user}]
        if image_b64:
            parts.insert(0, {"inline_data": {"mime_type": "image/png",
                                             "data": image_b64}})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0},
        }
        resp = requests.post(
            url, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self._access_token()}",
                    "Content-Type": "application/json"},
            json=body)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts_out = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts_out)
