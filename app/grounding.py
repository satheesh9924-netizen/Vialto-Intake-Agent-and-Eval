"""Grounding check: is a value the model returned actually supported by the source document?

This is the core hallucination test. If the model returns a passport number that
does not appear in the document, it made it up. Used both at runtime (as a
guardrail) and in the evaluation metrics.
"""
import re

from .schemas import DATE_FIELDS, MONEY_FIELDS, parse_date

_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b",
    r"\b\d{1,2}-[A-Za-z]{3}-\d{4}\b",
    r"\b\d{1,2} [A-Za-z]{3} \d{4}\b",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def iso_dates_in(text: str) -> set:
    found = set()
    for pat in _DATE_PATTERNS:
        for m in re.finditer(pat, text):
            iso = parse_date(m.group(0))
            if iso:
                found.add(iso)
    return found


def numbers_in(text: str) -> set:
    nums = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            nums.add(round(float(m.group(0).replace(",", "")), 2))
        except ValueError:
            pass
    return nums


def value_in_text(value, text: str, field: str = "") -> bool:
    """True if `value` is supported by `text` (after sensible normalisation)."""
    if value is None:
        return True
    v = str(value)
    if field in DATE_FIELDS or re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        # normalised ISO date present in doc, or the raw value is literally in the doc
        return v in iso_dates_in(text) or _norm(v) in _norm(text)
    m = re.fullmatch(r"([A-Z]{3}) (\d+(?:\.\d+)?)", v)
    if field in MONEY_FIELDS or m:
        if not m:
            return _norm(v) in _norm(text)
        cur, amt = m.group(1), round(float(m.group(2)), 2)
        return cur in text and amt in numbers_in(text)
    return _norm(v) in _norm(text)


def is_grounded(field: str, value, text: str) -> bool:
    return value_in_text(value, text, field)
