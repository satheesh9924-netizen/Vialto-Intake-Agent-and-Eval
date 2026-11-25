"""Shared definitions: document categories, required fields, validation formats."""
import re
from datetime import date, datetime

CATEGORIES = ["passport", "visa_approval", "employment_letter", "payslip", "tax_return", "unknown"]

REQUIRED_FIELDS = {
    "passport": ["full_name", "passport_number", "nationality", "expiry_date"],
    "visa_approval": ["full_name", "visa_type", "receipt_number", "valid_until"],
    "employment_letter": ["full_name", "employer", "job_title", "start_date"],
    "payslip": ["full_name", "employer", "gross_pay", "period"],
    "tax_return": ["full_name", "tax_year", "total_income"],
    "unknown": [],
}

DATE_FIELDS = {"expiry_date", "valid_until", "start_date"}
MONEY_FIELDS = {"gross_pay", "total_income"}

# Tools the agent is allowed to call. Anything else a model proposes is blocked.
ALLOWED_TOOLS = {"create_case", "flag_for_review"}

FORMAT_RULES = {
    "passport_number": re.compile(r"^[A-Z0-9]{6,9}$"),
    "receipt_number": re.compile(r"^[A-Z]{3}\d{10}$"),
    "gross_pay": re.compile(r"^[A-Z]{3} \d+\.\d{2}$"),
    "total_income": re.compile(r"^[A-Z]{3} \d+\.\d{2}$"),
    "tax_year": re.compile(r"^(19|20)\d{2}$"),
}

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# Closed lists: a value outside them is almost always an OCR slip (O/0, B/8, D/O...) and must go to a human.
VISA_TYPES = {"H-1B", "H-4", "L-1A", "L-1B", "L-2", "O-1", "E-2", "E-3", "F-1", "J-1", "B-1", "B-2", "TN", "EB-1", "EB-2", "EB-3"}
CURRENCIES = {"USD", "EUR", "GBP", "INR", "CAD", "AUD", "SGD", "JPY", "CHF", "AED", "CNY", "SEK", "DKK", "NOK"}

DATE_FORMATS = ["%Y-%m-%d", "%d %B %Y", "%d-%b-%Y", "%d %b %Y"]


def parse_date(raw: str):
    """Return an ISO date string (YYYY-MM-DD) or None if the text is not a known date format."""
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def is_valid_format(field: str, value) -> bool:
    """True if a (non-null) value has the expected format for its field."""
    if value is None:
        return True
    if field in DATE_FIELDS:
        try:
            datetime.strptime(str(value), "%Y-%m-%d")
            return True
        except ValueError:
            return False
    if field == "visa_type" and str(value) not in VISA_TYPES:
        return False
    if field in MONEY_FIELDS and str(value)[:3] not in CURRENCIES:
        return False
    rule = FORMAT_RULES.get(field)
    return bool(rule.match(str(value))) if rule else bool(str(value).strip())


def days_until(iso_date: str, as_of: date) -> int:
    return (datetime.strptime(iso_date, "%Y-%m-%d").date() - as_of).days
