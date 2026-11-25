"""AI-judge calibration, drift detection / release gating, and the metric maths."""
import math

import pytest

from app.llm import MockLLM
from evalkit import drift, judge, metrics, runner
from conftest import BY_ID, GOLDEN, load


# ---- judge calibration ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calibration():
    return judge.calibrate(judge.HeuristicJudge(), load("data/judge/calibration.jsonl"), BY_ID)


def test_judge_is_trustworthy_enough(calibration):
    assert calibration["kappa"] >= 0.6 and calibration["trusted"]


def test_judge_catches_invented_claims_wrong_values_leaks_and_gaps(calibration):
    missed = {d["id"] for d in calibration["disagreements"]}
    assert not any(m.startswith(("cal-invent", "cal-wrong", "cal-incomplete", "cal-leak")) for m in missed)


def test_judge_known_blind_spot_unsupported_inference(calibration):
    """Test the tester, and document its limit: a rule-based judge is fooled by an unsupported inference that
    reuses words from the document. This is why a real pipeline adds an LLM judge for that class of defect."""
    assert {d["id"] for d in calibration["disagreements"]} == {"cal-inference-1", "cal-inference-2"}


# ---- drift: decision logic on hand-made snapshots ----------------------------------------------
def snap(**m):
    return {"metrics": {k: {"value": v[0], "n": v[1]} for k, v in m.items()}}


def verdict(findings, metric):
    return next(f["verdict"] for f in findings if f["metric"] == metric)


def verdicts(findings, metric):
    return [f["verdict"] for f in findings if f["metric"] == metric]


def test_drift_small_change_is_ok():
    f = drift.compare(snap(classification_accuracy=(0.97, 100)), snap(classification_accuracy=(0.95, 100)))
    assert verdict(f, "classification_accuracy") == "ok"


def test_drift_large_significant_drop_is_regression():
    f = drift.compare(snap(classification_accuracy=(0.97, 200)), snap(classification_accuracy=(0.85, 200)))
    assert verdict(f, "classification_accuracy") == "regress"
    assert drift.gate(f)["passed"] is False


def test_drift_large_drop_on_tiny_sample_is_only_a_warning():
    """6 samples cannot prove a change is real: warn, do not block (and ask for more data)."""
    f = drift.compare(snap(flag_rate=(0.10, 6)), snap(flag_rate=(0.40, 6)))
    assert verdict(f, "flag_rate") == "warn" and drift.gate(f)["passed"] is True
    f = drift.compare(snap(flag_rate=(0.10, 200)), snap(flag_rate=(0.40, 200)))   # same change, enough data
    assert verdict(f, "flag_rate") == "regress"


def test_drift_direction_matters_for_lower_is_better_metrics():
    f = drift.compare(snap(hallucination_rate=(0.0, 100)), snap(hallucination_rate=(0.10, 100)))
    assert verdict(f, "hallucination_rate") == "regress"
    f = drift.compare(snap(hallucination_rate=(0.10, 100)), snap(hallucination_rate=(0.0, 100)))
    assert verdict(f, "hallucination_rate") == "ok"  # improvement


def test_floor_breach_blocks_even_without_drift():
    f = drift.compare(snap(consistency=(0.90, 75)), snap(consistency=(0.90, 75)))
    assert verdicts(f, "consistency") == ["ok", "floor_breach"]   # no drift from baseline, but below the business minimum
    assert not drift.gate(f)["passed"]


# ---- drift: the real release gate on real runs -------------------------------------------------
@pytest.fixture(scope="module")
def baseline():
    b = runner.load_baseline()
    assert b, "run `python run_evals.py --update-baseline` first"
    return b


def test_gate_passes_for_the_healthy_model(baseline):
    res = runner.full_eval("v1")
    _, g = runner.evaluate_release(res, baseline)
    assert g["passed"], g["blocking"]


def test_gate_blocks_the_degraded_model(baseline):
    res = runner.full_eval("v2_degraded")
    findings, g = runner.evaluate_release(res, baseline)
    assert not g["passed"]
    assert {"hallucination_rate", "classification_accuracy", "field_f1"} <= set(g["blocking"])


def test_gate_blocks_the_flaky_model(baseline):
    res = runner.full_eval("flaky")
    _, g = runner.evaluate_release(res, baseline)
    assert not g["passed"] and "consistency" in g["blocking"]


# ---- metric maths, checked by hand --------------------------------------------------------------
def test_classification_report_hand_example():
    r = metrics.classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert r["accuracy"] == 0.75
    assert r["per_class"]["a"]["precision"] == 1.0 and r["per_class"]["a"]["recall"] == 0.5
    assert math.isclose(r["per_class"]["b"]["precision"], 2 / 3) and r["per_class"]["b"]["recall"] == 1.0


def test_field_prf_hand_example():
    # 3 expected values: 1 right, 1 wrong, 1 missing; plus 1 value where nothing was expected
    r = metrics.field_prf([({"x": "1", "y": "2", "z": "3", "w": None}, {"x": "1", "y": "9", "z": None, "w": "7"})])
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 2)


def test_two_proportion_p_value_behaviour():
    assert metrics.two_proportion_p(0.5, 100, 0.5, 100) == 1.0
    assert metrics.two_proportion_p(0.9, 100, 0.5, 100) < 0.001
    assert metrics.two_proportion_p(1.0, 5, 0.8, 5) > 0.05  # small samples: not convincing


def test_wilson_interval_contains_estimate_and_narrows_with_more_data():
    lo1, hi1 = metrics.wilson_interval(9, 10)
    lo2, hi2 = metrics.wilson_interval(90, 100)
    assert lo1 < 0.9 < hi1 and (hi2 - lo2) < (hi1 - lo1)


def test_cohens_kappa_known_values():
    assert metrics.cohens_kappa(list("aabb"), list("aabb")) == 1.0
    assert math.isclose(metrics.cohens_kappa(list("aabb"), list("abab")), 0.0, abs_tol=1e-9)
