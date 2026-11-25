"""End-to-end agent behaviour: positive, negative, edge and failure scenarios."""
import json

from app.llm import FaultInjectingLLM, MockLLM
from conftest import doc


class StubLLM:
    """Returns whatever we script, to test how the agent copes with odd model output."""
    def __init__(self, classify, extract=None):
        self.classify, self.extract = classify, extract
        self.profile = "stub"

    def generate(self, system, user):
        task = user.split("\n", 1)[0].split(":", 1)[1].strip()
        if task == "classify":
            return json.dumps(self.classify)
        if task == "extract":
            return json.dumps(self.extract)
        return json.dumps({"summary": "ok"})


# ---- positive -----------------------------------------------------------------------------
def test_positive_clean_passport_creates_case(make_agent):
    agent, tools = make_agent()
    d = doc("passport-1")
    r = agent.process(d["id"], d["text"])
    assert r.status == "case_created" and r.action == "create_case"
    stored = tools.get_case(d["id"])
    assert stored["category"] == "passport" and stored["fields"] == d["expected_fields"]
    assert r.summary and "Ana Silva" in r.summary


# ---- negative -----------------------------------------------------------------------------
def test_negative_non_document_is_flagged_not_processed(make_agent):
    agent, tools = make_agent()
    r = agent.process("u1", doc("unknown-2")["text"])
    assert r.status == "flagged" and tools.count("cases") == 0


# ---- edge ---------------------------------------------------------------------------------
def test_edge_passport_expiring_soon_goes_to_review(make_agent):
    agent, tools = make_agent()
    d = doc("passport-6")
    r = agent.process(d["id"], d["text"])
    assert r.status == "flagged" and "passport_expiring_soon" in r.reasons and tools.count("cases") == 0


def test_edge_expiry_rule_depends_on_evaluation_date(make_agent):
    from datetime import date
    agent, _ = make_agent()
    agent.as_of = date(2025, 1, 1)  # a year earlier: the same passport is comfortably valid
    d = doc("passport-6")
    assert agent.process(d["id"], d["text"]).status == "case_created"


def test_edge_visa_mentioned_in_passing_is_still_an_employment_letter(make_agent):
    agent, _ = make_agent()
    d = doc("employment-6")
    r = agent.process(d["id"], d["text"])
    assert r.category == "employment_letter" and r.status == "case_created"


def test_edge_windows_line_endings(make_agent):
    agent, _ = make_agent()
    d = doc("passport-1")
    r = agent.process("crlf", d["text"].replace("\n", "\r\n"))
    assert r.raw_fields == d["expected_fields"]


def test_edge_extra_whitespace_and_blank_lines(make_agent):
    agent, _ = make_agent()
    d = doc("visa-1")
    noisy = "\n\n".join(f"   {line}   " for line in d["text"].splitlines())
    assert agent.process("ws", noisy).raw_fields == d["expected_fields"]


def test_edge_empty_and_whitespace_input_rejected(make_agent):
    agent, tools = make_agent()
    for text in ["", "   \n\t  "]:
        r = agent.process("e", text)
        assert r.status == "rejected" and r.reasons == ["empty_input"]
    assert tools.count("cases") == 0 and tools.count("reviews") == 0


def test_edge_oversized_input_rejected_without_calling_model(make_agent):
    llm = MockLLM("v1")
    agent, _ = make_agent(llm)
    r = agent.process("big", "x" * 25_000)
    assert r.status == "rejected" and llm.calls == 0


def test_edge_duplicate_submission_is_idempotent(make_agent):
    agent, tools = make_agent()
    d = doc("visa-1")
    first = agent.process(d["id"], d["text"])
    second = agent.process(d["id"], d["text"])
    assert (first.status, second.status) == ("case_created", "case_exists")
    assert tools.count("cases") == 1


# ---- failure ------------------------------------------------------------------------------
def test_failure_model_timeout_degrades_gracefully(make_agent):
    agent, tools = make_agent(FaultInjectingLLM(MockLLM("v1"), "timeout"))
    r = agent.process("t", doc("passport-1")["text"])
    assert r.status == "flagged" and "model_unavailable" in r.reasons and tools.count("cases") == 0


def test_failure_transient_timeout_is_retried(make_agent):
    agent, _ = make_agent(FaultInjectingLLM(MockLLM("v1"), "timeout", times=1))
    assert agent.process("t", doc("passport-1")["text"]).status == "case_created"


def test_failure_model_returns_garbage(make_agent):
    agent, tools = make_agent(FaultInjectingLLM(MockLLM("v1"), "garbage"))
    r = agent.process("g", doc("passport-1")["text"])
    assert r.status == "flagged" and "model_unavailable" in r.reasons and tools.count("cases") == 0


def test_failure_model_returns_invalid_category(make_agent):
    agent, _ = make_agent(StubLLM({"category": "banana", "confidence": 0.99}))
    assert agent.process("b", "anything").reasons == ["model_unavailable"]


def test_failure_database_down_falls_back_to_review_queue(make_agent):
    agent, tools = make_agent(fail={"create_case"})
    r = agent.process("d", doc("passport-1")["text"])
    assert r.status == "flagged" and "tool_failure" in r.reasons
    assert tools.count("cases") == 0 and tools.count("reviews") == 1


def test_failure_crash_mid_write_leaves_no_partial_data(make_agent):
    """Transaction atomicity: the insert happened, then the tool crashed. Nothing may remain."""
    agent, tools = make_agent(fail={"create_case_midway"})
    r = agent.process("m", doc("passport-1")["text"])
    assert tools.count("cases") == 0 and r.status == "flagged"


def test_failure_everything_down_returns_error_without_raising(make_agent):
    agent, _ = make_agent(fail={"create_case", "flag_for_review"})
    r = agent.process("x", doc("passport-1")["text"])
    assert r.status == "error"


# ---- guardrails vs a misbehaving model -----------------------------------------------------
def test_overconfident_model_is_overruled_by_policy(make_agent):
    """The degraded model proposes create_case even with fields missing. Policy must downgrade it."""
    d = doc("passport-4")  # expiry date missing
    guarded, tools_g = make_agent(MockLLM("v2_degraded"))
    r = guarded.process(d["id"], d["text"])
    assert r.status == "flagged" and tools_g.count("cases") == 0
    assert r.fields["expiry_date"] is None and any(f.startswith("ungrounded_field_removed") for f in r.flags)


def test_without_guardrails_the_same_model_creates_a_bad_case(make_agent):
    """Ablation: switch the guardrails off and the invented value goes straight into the database."""
    d = doc("passport-4")
    unguarded, tools_u = make_agent(MockLLM("v2_degraded"), guardrails=False)
    r = unguarded.process(d["id"], d["text"])
    assert r.status == "case_created"
    assert tools_u.get_case(d["id"])["fields"]["expiry_date"] == "2030-01-01"  # made up
