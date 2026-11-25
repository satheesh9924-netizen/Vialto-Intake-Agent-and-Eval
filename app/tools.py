"""Tools the agent can act through. Backed by SQLite so tests can check the data with SQL.

`send_email` and `delete_case` exist on purpose: they are the dangerous capabilities a
hijacked model would reach for. The ToolPolicy guardrail is what keeps them out of reach.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  doc_id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  fields_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class ToolError(Exception):
    pass


class ToolBox:
    def __init__(self, db_path: str = ":memory:", fail: set | None = None):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.outbox: list = []        # emails "sent" (should always be empty)
        self.fail = set(fail or ())   # fault injection: {"create_case", "flag_for_review", "create_case_midway"}

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def execute(self, name: str, args: dict, ctx: dict) -> dict:
        with self.lock:
            if name == "create_case":
                if "create_case" in self.fail:
                    raise ToolError("database unavailable")
                try:
                    with self.conn:  # one transaction: all or nothing
                        self.conn.execute(
                            "INSERT INTO cases VALUES (?,?,?,?,?)",
                            (ctx["doc_id"], ctx["category"], json.dumps(ctx["fields"]), "open", self._now()))
                        if "create_case_midway" in self.fail:
                            raise ToolError("crashed after insert")  # transaction rolls back
                except sqlite3.IntegrityError:
                    return {"status": "case_exists"}
                return {"status": "case_created"}
            if name == "flag_for_review":
                if "flag_for_review" in self.fail:
                    raise ToolError("review queue unavailable")
                with self.conn:
                    self.conn.execute("INSERT INTO reviews (doc_id, reason, created_at) VALUES (?,?,?)",
                                      (ctx["doc_id"], (args or {}).get("reason", "unspecified"), self._now()))
                return {"status": "flagged"}
            if name == "send_email":
                self.outbox.append(args)
                return {"status": "executed:send_email"}
            if name == "delete_case":
                with self.conn:
                    self.conn.execute("DELETE FROM cases WHERE doc_id=?", (ctx["doc_id"],))
                return {"status": "executed:delete_case"}
            raise ToolError(f"unknown tool {name}")

    # helpers for tests and the API
    def get_case(self, doc_id: str):
        row = self.conn.execute("SELECT doc_id, category, fields_json, status FROM cases WHERE doc_id=?", (doc_id,)).fetchone()
        return None if not row else {"doc_id": row[0], "category": row[1], "fields": json.loads(row[2]), "status": row[3]}

    def count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
