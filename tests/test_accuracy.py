"""Classification + extraction accuracy against the golden dataset (ground truth)."""
import pytest

from app.schemas import CATEGORIES
from conftest import GOLDEN, doc

CLEAN_TAGS = {"clean", "date_variant", "format_variant"}


# ---- classification ---------------------------------------------------------------------------
def test_classification_accuracy_meets_floor(v1_metrics):
    assert v1_metrics["classification_accuracy"]["value"] >= 0.90


@pytest.mark.parametrize("category", [c for c in CATEGORIES])
def test_per_category_recall(v1_records, category):
    """Overall accuracy can hide one category failing completely, so check each separately."""
    rows = [(d, r) for d, r in v1_records if d["expected_category"] == category]
    recall = sum(r.category == category for _, r in rows) / len(rows)
    assert recall >= 0.80, f"{category} recall {recall:.2f}"


def test_no_silent_errors(v1_metrics):
    """A wrong category must never silently become a case: it has to be routed to a human."""
    assert v1_metrics["silent_error_rate"]["value"] == 0


def test_ambiguous_document_is_routed_to_human(v1_records):
    d, r = next((d, r) for d, r in v1_records if d["id"] == "employment-5")
    assert r.status == "flagged" and "low_confidence" in r.reasons
    assert r.confidence < 0.70


@pytest.mark.parametrize("d", [d for d in GOLDEN if d["expected_category"] == "unknown"], ids=lambda d: d["id"])
def test_unrecognised_documents_never_create_cases(make_agent, d):
    agent, tools = make_agent()
    r = agent.process(d["id"], d["text"])
    assert r.status == "flagged" and tools.count("cases") == 0 and tools.count("reviews") == 1


# ---- extraction -------------------------------------------------------------------------------
@pytest.mark.parametrize("d", [d for d in GOLDEN if d["tag"] in CLEAN_TAGS and d["expected_category"] != "unknown"],
                         ids=lambda d: d["id"])
def test_clean_documents_extract_exactly(v1_records, d):
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    assert r.raw_fields == d["expected_fields"]
    assert r.status == "case_created"


def test_field_f1_meets_floor(v1_metrics):
    assert v1_metrics["field_f1"]["value"] >= 0.90


def test_all_values_in_required_format(v1_metrics):
    assert v1_metrics["format_validity_rate"]["value"] == 1.0


@pytest.mark.parametrize("d", [d for d in GOLDEN if d["tag"] == "missing_field"], ids=lambda d: d["id"])
def test_missing_field_is_null_never_invented(v1_records, d):
    """NEGATIVE test: when the document lacks a field the answer must be null, and the case goes to review."""
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    missing = [f for f, v in d["expected_fields"].items() if v is None]
    assert missing and all(r.raw_fields[f] is None for f in missing)
    assert r.status == "flagged" and "missing_fields" in r.reasons


@pytest.mark.parametrize("d", [d for d in GOLDEN if d["tag"] == "noisy_ocr"], ids=lambda d: d["id"])
def test_noisy_ocr_fails_safe(v1_records, d):
    """KNOWN LIMITATION: OCR damage to a label makes a field unreadable for this model. The safe behaviour
    is to return null and send the document to a human, not to guess."""
    r = next(r for dd, r in v1_records if dd["id"] == d["id"])
    wrong = [f for f, v in d["expected_fields"].items() if r.raw_fields.get(f) not in (v, None)]
    assert wrong == [], "model returned a WRONG value instead of null"
    assert r.status == "flagged"
