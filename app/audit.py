"""Audit trail: who/what/when for every run, with personal data kept OUT of the log."""
import hashlib
import json
from datetime import datetime, timezone


class AuditLog:
    def __init__(self, path: str | None = None):
        self.entries: list = []
        self.path = path

    def record(self, text: str, result) -> dict:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": result.run_id,
            "doc_id": result.doc_id,
            "doc_sha256": hashlib.sha256(text.encode()).hexdigest()[:12],  # traceable without storing content
            "category": result.category,
            "status": result.status,
            "action": result.action,
            "flags": result.flags,
            "reasons": result.reasons,
            "fields_present": sorted(k for k, v in (result.fields or {}).items() if v),  # names only, never values
            "guardrails": result.guardrails,
            "model": result.model,
        }
        self.entries.append(entry)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return entry
