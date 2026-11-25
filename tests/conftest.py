import json
import pathlib
from datetime import date

import pytest

from app.agent import DocumentIntakeAgent
from app.llm import MockLLM
from app.tools import ToolBox
from evalkit import runner

ROOT = pathlib.Path(__file__).resolve().parents[1]
AS_OF = runner.AS_OF


def load(rel):
    return [json.loads(l) for l in (ROOT / rel).read_text().splitlines() if l.strip()]


GOLDEN = load("data/golden/documents.jsonl")
ATTACKS = load("data/adversarial/attacks.jsonl")
BY_ID = {d["id"]: d for d in GOLDEN}


@pytest.fixture(scope="session")
def golden():
    return GOLDEN


@pytest.fixture(scope="session")
def v1_records():
    """The whole golden set through the healthy model, once."""
    return runner.run_golden(MockLLM("v1"), GOLDEN)


@pytest.fixture(scope="session")
def v1_metrics(v1_records):
    return runner.compute_metrics(v1_records)[0]


@pytest.fixture(params=["pipeline", "langgraph"])
def make_agent(request):
    """Every scenario test runs against BOTH the plain pipeline and the LangGraph agent."""
    from app.graph_agent import agent_class
    cls = agent_class(request.param)

    def _make(llm=None, guardrails=True, fail=None, **kw):
        tools = ToolBox(fail=fail)
        agent = cls(llm or MockLLM("v1"), toolbox=tools, guardrails=guardrails, as_of=AS_OF, **kw)
        return agent, tools
    return _make


def doc(doc_id):
    return BY_ID[doc_id]
