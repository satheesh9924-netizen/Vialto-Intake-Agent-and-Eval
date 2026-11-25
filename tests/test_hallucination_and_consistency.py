"""Hallucination (grounding), consistency (run it again), and the judge on free text."""
import pytest

from app.grounding import is_grounded
from app.llm import MockLLM
from evalkit import runner
from conftest import GOLDEN


# ---- grounding unit tests ---------------------------------------------------------------------
@pytest.mark.parametrize("field,value,text,expected", [
    ("passport_number", "P1234567", "Passport No: P1234567", True),
    ("passport_number", "P1234568", "Passport No: P1234567", False),          # one digit off = invented
    ("expiry_date", "2030-03-12", "Date of expiry: 12 March 2030", True),     # normalised date, present in a different format
    ("expiry_date", "2030-03-13", "Date of expiry: 12 March 2030", False),
    ("gross_pay", "USD 8500.00", "Gross Pay: USD 8,500.00", True),            # commas normalised
    ("gross_pay", "USD 9500.00", "Gross Pay: USD 8,500.00", False),
    ("gross_pay", "EUR 8500.00", "Gross Pay: USD 8,500.00", False),           # right number, wrong currency
    ("full_name", "ana  silva", "Name: Ana Silva", True),                     # case/whitespace tolerant
    ("full_name", None, "anything", True),                                    # null is never a hallucination
])
def test_grounding_rules(field, value, text, expected):
    assert is_grounded(field, value, text) is expected


def test_healthy_model_does_not_hallucinate(v1_metrics):
    assert v1_metrics["hallucination_rate"]["value"] == 0


def test_hallucination_metric_detects_a_degraded_model():
    """Test the tester: the metric must actually fire when a model invents values."""
    records = runner.run_golden(MockLLM("v2_degraded"), GOLDEN)
    assert runner.compute_metrics(records)[0]["hallucination_rate"]["value"] > 0.05


def test_guardrail_removes_hallucinated_values_at_runtime():
    records = runner.run_golden(MockLLM("v2_degraded"), GOLDEN)
    assert runner.compute_metrics(records)[0]["bad_case_rate"]["value"] == 0  # none reached the database


# ---- consistency ------------------------------------------------------------------------------
def test_healthy_model_is_consistent():
    assert runner.run_consistency(MockLLM("v1"), GOLDEN)["value"] == 1.0


def test_consistency_metric_detects_a_flaky_model():
    """Test the tester: a model that sometimes flips its answer must score below the 95% floor."""
    assert runner.run_consistency(MockLLM("flaky"), GOLDEN)["value"] < 0.95


# ---- judge on free text ---------------------------------------------------------------------
def test_judge_accepts_faithful_summaries(v1_records):
    from evalkit.judge import HeuristicJudge
    out = runner.run_judge(v1_records, HeuristicJudge())
    assert out["value"] >= 0.90 and out["n"] > 20


def test_judge_rejects_summaries_from_degraded_model():
    from evalkit.judge import HeuristicJudge
    out = runner.run_judge(runner.run_golden(MockLLM("v2_degraded"), GOLDEN), HeuristicJudge())
    assert out["value"] < 0.5  # every v2 summary contains an invented claim
