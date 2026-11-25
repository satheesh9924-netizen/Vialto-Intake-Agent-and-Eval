"""The same quality checks expressed with the tools named in the job description: DeepEval and RAGAS.

Everything here is deterministic and needs no API key. LLM-judged metrics from these libraries
(GEval, Faithfulness...) need a judge model and are intentionally not part of the offline suite.
"""
import asyncio
import json
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import pytest
from deepeval import assert_test
from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.test_case import LLMTestCase, ToolCall
from ragas.metrics.collections import ExactMatch, NonLLMStringSimilarity, StringPresence

from app.agent import DocumentIntakeAgent
from app.llm import MockLLM
from app.tools import ToolBox
from evalkit import runner
from evalkit.deepeval_suite import GroundingMetric, ToolPermissionMetric, to_test_case
from conftest import ATTACKS, AS_OF, GOLDEN, doc

CLEAN = [d for d in GOLDEN if d["tag"] in {"clean", "date_variant", "format_variant"} and d["expected_category"] != "unknown"]


# ---------------------------------------------------------------- DeepEval
@pytest.mark.parametrize("d", CLEAN, ids=lambda d: d["id"])
def test_deepeval_extraction_equals_ground_truth(v1_records, d):
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    assert_test(to_test_case(d, r), [ExactMatchMetric(threshold=1.0)])


@pytest.mark.parametrize("d", [d for d in GOLDEN if d["expected_category"] != "unknown"], ids=lambda d: d["id"])
def test_deepeval_no_hallucination_on_healthy_model(v1_records, d):
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    assert_test(to_test_case(d, r), [GroundingMetric()])


def test_deepeval_grounding_metric_catches_a_hallucinating_model():
    """Test the tester: the metric must FAIL when a model invents values."""
    d = doc("passport-4")  # expiry date missing in the document
    r = DocumentIntakeAgent(MockLLM("v2_degraded"), toolbox=ToolBox(), guardrails=False, as_of=AS_OF).process(d["id"], d["text"])
    m = GroundingMetric()
    m.measure(to_test_case(d, r))
    assert m.score < 1.0 and not m.is_successful() and "expiry_date" in m.reason


def test_deepeval_date_format_pattern():
    d = doc("passport-3")  # source date is "05-Sep-2029"; the output must be ISO
    r = DocumentIntakeAgent(MockLLM("v1"), as_of=AS_OF).process(d["id"], d["text"])
    tc = LLMTestCase(input=d["text"], actual_output=r.raw_fields["expiry_date"])
    assert_test(tc, [PatternMatchMetric(pattern=r"^\d{4}-\d{2}-\d{2}$")])


def test_deepeval_date_format_pattern_catches_unnormalised_dates():
    d = doc("passport-3")
    r = DocumentIntakeAgent(MockLLM("v2_degraded"), guardrails=False, as_of=AS_OF).process(d["id"], d["text"])
    m = PatternMatchMetric(pattern=r"^\d{4}-\d{2}-\d{2}$")
    m.measure(LLMTestCase(input="", actual_output=r.raw_fields["expiry_date"]))
    assert not m.is_successful()


@pytest.mark.parametrize("attack", [a for a in ATTACKS if "exfil" in a["goals"] or "delete" in a["goals"]], ids=lambda a: a["id"])
def test_deepeval_agent_only_uses_allowed_tools(attack):
    """Agent safety. With guardrails the executed tool is always allowed; without them the hijacked tool runs."""
    on = DocumentIntakeAgent(MockLLM("v1"), toolbox=ToolBox(), guardrails=True, as_of=AS_OF).process(attack["id"], attack["text"])
    assert_test(to_test_case({"text": attack["text"], "expected_fields": {}}, on), [ToolPermissionMetric()])
    off = DocumentIntakeAgent(MockLLM("v1"), toolbox=ToolBox(), guardrails=False, as_of=AS_OF).process(attack["id"], attack["text"])
    m = ToolPermissionMetric()
    m.measure(to_test_case({"text": attack["text"], "expected_fields": {}}, off))
    assert m.score == 0.0, "the attack should have used a disallowed tool when guardrails are off"


# ---------------------------------------------------------------- RAGAS (non-LLM metrics)
def run(coro):
    return asyncio.run(coro)


def summary_of(d):
    """The summary the healthy model produces for the ground-truth fields: our reference answer."""
    return json.loads(MockLLM("v1").generate("", f"TASK: summarize\nCATEGORY: {d['expected_category']}\n"
                                              f"FIELDS: {json.dumps(d['expected_fields'])}\n<<<DOC\nx\nDOC>>>"))["summary"]


@pytest.mark.parametrize("d", CLEAN, ids=lambda d: d["id"])
def test_ragas_summary_mentions_every_key_value(v1_records, d):
    """RAGAS StringPresence: is each ground-truth value present in the summary (completeness)?"""
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    for field, value in d["expected_fields"].items():
        assert run(StringPresence().ascore(reference=str(value), response=r.summary)).value == 1.0, field


def test_ragas_summary_similarity_to_reference_separates_healthy_from_degraded():
    healthy = runner.run_golden(MockLLM("v1"), CLEAN)
    degraded = runner.run_golden(MockLLM("v2_degraded"), CLEAN, guardrails=False)  # raw degraded output

    def mean_sim(records):
        sims = [run(NonLLMStringSimilarity().ascore(reference=summary_of(d), response=r.summary)).value for d, r in records if r.summary]
        return sum(sims) / len(sims)

    assert mean_sim(healthy) == pytest.approx(1.0)
    assert mean_sim(degraded) < 0.95


def test_ragas_exact_match_on_structured_output(v1_records):
    d = doc("visa-1")
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    got = run(ExactMatch().ascore(reference=json.dumps(d["expected_fields"], sort_keys=True),
                                  response=json.dumps(r.raw_fields, sort_keys=True)))
    assert got.value == 1.0
