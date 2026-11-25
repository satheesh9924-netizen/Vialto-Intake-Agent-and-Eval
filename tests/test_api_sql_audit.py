"""API-level tests (httpx via FastAPI TestClient), SQL data-integrity checks, and the audit trail."""
import json

import pytest
from fastapi.testclient import TestClient

from app.agent import DocumentIntakeAgent
from app.api import create_app
from app.llm import MockLLM
from app.schemas import REQUIRED_FIELDS
from app.tools import ToolBox
from conftest import ATTACKS, AS_OF, GOLDEN, doc


@pytest.fixture
def client():
    return TestClient(create_app())


# ---- API contract -----------------------------------------------------------------------------
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_intake_contract(client):
    d = doc("passport-1")
    r = client.post("/intake", json={"doc_id": d["id"], "text": d["text"]})
    assert r.status_code == 200
    body = r.json()
    assert {"doc_id", "run_id", "status", "category", "confidence", "fields", "flags", "reasons", "summary"} <= body.keys()
    assert body["status"] == "case_created" and body["fields"] == d["expected_fields"]
    assert "trace" not in body and "raw_fields" not in body            # internals are not exposed


def test_case_can_be_read_back(client):
    d = doc("visa-1")
    client.post("/intake", json={"doc_id": d["id"], "text": d["text"]})
    r = client.get(f"/cases/{d['id']}")
    assert r.status_code == 200 and r.json()["category"] == "visa_approval"


def test_unknown_case_is_404(client):
    assert client.get("/cases/nope").status_code == 404


@pytest.mark.parametrize("payload", [
    {}, {"doc_id": "x"}, {"text": "hello"}, {"doc_id": "", "text": "hello"}, {"doc_id": "x", "text": ""},
    {"doc_id": "x", "text": "a" * 100_001}, {"doc_id": 123, "text": "hello"}, {"doc_id": "x", "text": None},
])
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/intake", json=payload).status_code == 422


def test_duplicate_post_does_not_create_two_cases(client):
    d = doc("visa-1")
    first = client.post("/intake", json={"doc_id": d["id"], "text": d["text"]}).json()
    second = client.post("/intake", json={"doc_id": d["id"], "text": d["text"]}).json()
    assert (first["status"], second["status"]) == ("case_created", "case_exists")
    assert client.app.state.tools.count("cases") == 1


def test_attack_through_the_api_cannot_send_email(client):
    a = next(x for x in ATTACKS if x["id"] == "inj-04")
    r = client.post("/intake", json={"doc_id": "api-attack", "text": a["text"]})
    assert r.status_code == 200 and r.json()["status"] == "flagged"
    assert client.app.state.tools.outbox == []


# ---- SQL integrity over a full run --------------------------------------------------------------
@pytest.fixture(scope="module")
def populated():
    tools = ToolBox()
    agent = DocumentIntakeAgent(MockLLM("v1"), toolbox=tools, as_of=AS_OF)
    for d in GOLDEN:
        agent.process(d["id"], d["text"])
    return tools


def rows(tools, sql):
    return tools.conn.execute(sql).fetchall()


def test_sql_every_document_ends_up_in_exactly_one_place(populated):
    cases = {r[0] for r in rows(populated, "SELECT doc_id FROM cases")}
    reviews = {r[0] for r in rows(populated, "SELECT doc_id FROM reviews")}
    assert cases.isdisjoint(reviews)
    assert cases | reviews == {d["id"] for d in GOLDEN}


def test_sql_no_duplicate_review_rows(populated):
    assert rows(populated, "SELECT doc_id, COUNT(*) c FROM reviews GROUP BY doc_id HAVING c > 1") == []


def test_sql_stored_cases_are_complete_and_correct(populated):
    """Data integrity against ground truth: whatever was stored must be exactly right and have every required field."""
    for doc_id, category, fields_json in rows(populated, "SELECT doc_id, category, fields_json FROM cases"):
        stored, truth = json.loads(fields_json), next(d for d in GOLDEN if d["id"] == doc_id)
        assert category == truth["expected_category"], doc_id
        assert all(stored.get(f) is not None for f in REQUIRED_FIELDS[category]), doc_id
        assert stored == truth["expected_fields"], doc_id


def test_sql_unrecognised_documents_never_stored_as_cases(populated):
    assert rows(populated, "SELECT COUNT(*) FROM cases WHERE category IN ('unknown','seed')")[0][0] == 0


def test_sql_every_review_has_a_reason_and_timestamp(populated):
    assert rows(populated, "SELECT COUNT(*) FROM reviews WHERE reason IS NULL OR reason='' OR created_at IS NULL")[0][0] == 0


# ---- audit trail ------------------------------------------------------------------------------
def test_audit_log_is_traceable_and_contains_no_personal_data():
    agent = DocumentIntakeAgent(MockLLM("v1"), as_of=AS_OF)
    results = [agent.process(d["id"], d["text"]) for d in GOLDEN]
    assert [e["run_id"] for e in agent.audit.entries] == [r.run_id for r in results]   # one entry per run, traceable
    blob = json.dumps(agent.audit.entries)
    for d in GOLDEN:
        for field, value in d["expected_fields"].items():
            if value and field in ("full_name", "passport_number", "receipt_number", "gross_pay", "total_income"):
                assert str(value) not in blob, f"{field} leaked into the audit log"
    assert all(len(e["doc_sha256"]) == 12 for e in agent.audit.entries)
