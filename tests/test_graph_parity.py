"""LangGraph agent == plain pipeline agent. Same steps, so every outcome must match exactly."""
import json

import pytest

from app.agent import DocumentIntakeAgent
from app.graph_agent import GraphAgent
from app.llm import FaultInjectingLLM, MockLLM
from app.tools import ToolBox
from conftest import ATTACKS, AS_OF, GOLDEN


def outcome(agent_cls, llm, doc_id, text, guardrails, fail=None):
    a = agent_cls(llm, toolbox=ToolBox(fail=fail), guardrails=guardrails, as_of=AS_OF)
    d = a.process(doc_id, text).to_dict()
    d.pop("run_id")                      # random per run
    return json.dumps(d, sort_keys=True)


@pytest.mark.parametrize("profile", ["v1", "v2_degraded"])
@pytest.mark.parametrize("guardrails", [True, False])
def test_identical_on_every_golden_document(profile, guardrails):
    for d in GOLDEN:
        assert outcome(DocumentIntakeAgent, MockLLM(profile), d["id"], d["text"], guardrails) == \
               outcome(GraphAgent, MockLLM(profile), d["id"], d["text"], guardrails), d["id"]


@pytest.mark.parametrize("guardrails", [True, False])
def test_identical_on_every_attack(guardrails):
    for a in ATTACKS:
        assert outcome(DocumentIntakeAgent, MockLLM("v1"), a["id"], a["text"], guardrails) == \
               outcome(GraphAgent, MockLLM("v1"), a["id"], a["text"], guardrails), a["id"]


@pytest.mark.parametrize("fault", ["timeout", "garbage"])
def test_identical_when_the_model_fails(fault):
    text = GOLDEN[0]["text"]
    assert outcome(DocumentIntakeAgent, FaultInjectingLLM(MockLLM("v1"), fault), "f", text, True) == \
           outcome(GraphAgent, FaultInjectingLLM(MockLLM("v1"), fault), "f", text, True)


@pytest.mark.parametrize("fail", [{"create_case"}, {"create_case_midway"}, {"create_case", "flag_for_review"}])
def test_identical_when_tools_fail(fail):
    text = GOLDEN[0]["text"]
    assert outcome(DocumentIntakeAgent, MockLLM("v1"), "t", text, True, fail) == \
           outcome(GraphAgent, MockLLM("v1"), "t", text, True, fail)


def test_graph_shape_matches_the_steps():
    mermaid = GraphAgent(MockLLM("v1")).mermaid()
    for step in DocumentIntakeAgent.STEPS:
        assert step in mermaid
    assert "input_guard -.-> __end__" in mermaid     # early exit (rejected input) is part of the graph
