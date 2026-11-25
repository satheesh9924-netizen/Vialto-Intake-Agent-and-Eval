"""Document-AI testing with REAL OCR (Tesseract): image quality -> OCR -> agent -> case.

Slow (about a minute): marked `slow`. Skip with  pytest -m "not slow".
"""
import shutil

import pytest

from app.llm import MockLLM
from evalkit import ocr_eval, runner
from conftest import GOLDEN

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")]

QUALITIES = ["clean", "blur", "noise", "rotated", "low_res", "bad_scan"]


@pytest.fixture(scope="module")
def table():
    rows = ocr_eval.run_all(GOLDEN, lambda: MockLLM("v1"), runner.AS_OF, QUALITIES)
    return {r["quality"]: r for r in rows}


def test_clean_scan_is_read_correctly(table):
    c = table["clean"]
    assert c["ocr_similarity"] >= 0.99 and c["classification_accuracy"] >= 0.90 and c["field_f1"] >= 0.90


def test_clean_scan_stores_no_wrong_data(table):
    """Regression guard for the O/0 and USD/USO slips the first OCR run found (fixed by closed-list validators)."""
    assert table["clean"]["bad_case_rate"] == 0


def test_unreadable_scan_fails_safe(table):
    """When OCR is hopeless the system must send everything to a human and store nothing wrong."""
    b = table["bad_scan"]
    assert b["ocr_similarity"] < 0.5 and b["flag_rate"] >= 0.95 and b["bad_case_rate"] == 0


@pytest.mark.parametrize("quality", ["blur", "noise", "rotated", "low_res"])
def test_degraded_scans_stay_within_error_budget(table, quality):
    assert table[quality]["bad_case_rate"] <= 0.15 and table[quality]["classification_accuracy"] >= 0.90


def test_rotation_hurts_extraction_most(table):
    assert table["rotated"]["field_f1"] < table["clean"]["field_f1"] - 0.10


@pytest.mark.known_residual
@pytest.mark.xfail(strict=True, reason="KNOWN RESIDUAL: OCR swaps look-alike characters inside free-form IDs (Z/2 in a passport number). "
                                       "The value is grounded in the OCR text, so only a check digit or OCR confidence score can catch it.")
@pytest.mark.parametrize("quality", ["blur", "noise", "rotated", "low_res"])
def test_no_wrong_data_under_degraded_scans(table, quality):
    assert table[quality]["bad_case_rate"] == 0
