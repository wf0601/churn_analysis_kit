"""Parsers for turning a messy export into typed columns.

None of this is specific to one dataset. Mixed decimal separators, currency
symbols, US/EU date order, `Y`/`N` booleans and censored counts like `10+` show up
in every legacy CRM extract, so the parsers live here and the dataset-specific
adapters in `tools/` supply only the column mapping and the alias tables.

Two rules run through all of it. Ambiguity is reported, never guessed at silently —
`1,250` is 1250 under one convention and 1.25 under the other, and the caller gets
told which reading was taken. And nothing is inferred from a single value when the
column as a whole can settle it, which is how date order is resolved.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "missing", "unknown",
    "-", "--", "?", "not available", "not_available",
}

CURRENCY = re.compile(r"[€$£]|\b(?:EUR|USD|GBP)\b", re.IGNORECASE)

DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%m-%d-%Y", "%m/%d/%Y",
    "%B %Y", "%b %Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%Y-%m", "%Y",
]

BOOL_TRUE = {"1", "y", "yes", "true", "t", "ja", "j"}
BOOL_FALSE = {"0", "n", "no", "false", "f", "nee"}

_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")


def is_blank(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().lower() in NULL_TOKENS


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #
def parse_number(value, ambiguous: list | None = None) -> float:
    """Parse a number written in any common convention.

    Mixed separators are resolved by position: whichever of '.' or ',' appears last
    is the decimal separator, because no convention puts a thousands separator after
    the decimal point. A lone separator followed by exactly three digits is genuinely
    ambiguous; those values are appended to `ambiguous` (if given) and read as
    thousands, so the caller can report the choice rather than bury it.
    """
    if is_blank(value):
        return np.nan
    text = CURRENCY.sub("", str(value)).strip()
    text = text.replace(" ", "").replace(" ", "")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = re.sub(r"[^0-9,.\-+]", "", text)
    if not text or text in {"-", "+"}:
        return np.nan

    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        decimal = "." if text.rfind(".") > text.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        text = text.replace(thousands, "").replace(decimal, ".")
    elif has_comma or has_dot:
        sep = "," if has_comma else "."
        tail = text.split(sep)[-1]
        if text.count(sep) > 1:                     # 1.234.567 can only be thousands
            text = text.replace(sep, "")
        elif len(tail) == 3 and len(text.replace(sep, "")) > 3:
            if ambiguous is not None:
                ambiguous.append(str(value))
            text = text.replace(sep, "")            # read as thousands
        else:
            text = text.replace(sep, ".")
    try:
        result = float(text)
    except ValueError:
        return np.nan
    return -result if negative else result


def parse_numeric_series(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Vectorised `parse_number`. Returns the values and the ambiguous originals."""
    ambiguous: list[str] = []
    return series.map(lambda v: parse_number(v, ambiguous)), ambiguous


def parse_bool(value) -> float:
    """Booleans as 1.0/0.0/NaN, covering the usual encodings."""
    if is_blank(value):
        return np.nan
    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return 1.0
    if text in BOOL_FALSE:
        return 0.0
    number = parse_number(text)
    if not np.isnan(number):
        return 1.0 if number > 0 else 0.0
    return np.nan


def parse_capped_count(value) -> tuple[float, float]:
    """`10+` -> (10, capped=1); `5` -> (5, 0); blank -> (nan, nan).

    The flag matters: a censored count is a lower bound, and a model told it is an
    exact value will read the ceiling as a plateau in the response.
    """
    if is_blank(value):
        return np.nan, np.nan
    text = str(value).strip()
    if text.endswith("+"):
        return parse_number(text[:-1]), 1.0
    return parse_number(text), 0.0


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #
def parse_one_date(value, dayfirst: bool) -> pd.Timestamp:
    if is_blank(value):
        return pd.NaT
    text = str(value).strip()
    order = ["%d/%m/%Y", "%m/%d/%Y"] if dayfirst else ["%m/%d/%Y", "%d/%m/%Y"]
    formats = [f for f in DATE_FORMATS if f not in order]
    for fmt in [*formats[:4], *order, *formats[4:]]:
        try:
            return pd.Timestamp(pd.to_datetime(text, format=fmt))
        except (ValueError, TypeError):
            continue
    try:  # last resort: let pandas guess, consistently with the inferred order
        return pd.Timestamp(pd.to_datetime(text, dayfirst=dayfirst))
    except Exception:  # noqa: BLE001
        return pd.NaT


def infer_dayfirst(series: pd.Series) -> tuple[bool, str]:
    """Decide DD/MM vs MM/DD from the column, not from one value.

    A value whose first component exceeds 12 can only be a day; one whose second
    component exceeds 12 can only be a month. If both appear the column is mixed —
    which no parser can resolve, so it is reported rather than papered over.
    """
    day_first = month_first = 0
    for value in series.dropna().astype(str).head(5000):
        match = _DMY.match(value.strip())
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12:
            day_first += 1
        elif second > 12:
            month_first += 1
    if day_first and month_first:
        return True, (
            f"MIXED date order: {day_first} value(s) are unambiguously day-first and "
            f"{month_first} are unambiguously month-first. Ambiguous values in this "
            f"column cannot be resolved; day-first assumed."
        )
    if month_first:
        return False, "month-first (US) order inferred"
    if day_first:
        return True, "day-first (EU) order inferred"
    return True, "no unambiguous values; day-first (EU) assumed"


def parse_date_series(series: pd.Series) -> tuple[pd.Series, str]:
    """Parse a whole date column, inferring its order first. Returns (values, note)."""
    dayfirst, note = infer_dayfirst(series)
    parsed = series.map(lambda v: parse_one_date(v, dayfirst))
    return parsed.dt.normalize(), note


def split_dates(value) -> list[str]:
    """Split a multi-date cell — `a;b`, `a|b`, or a JSON list — into parts."""
    if is_blank(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            return [str(v) for v in json.loads(text.replace("'", '"'))]
        except (json.JSONDecodeError, TypeError):
            pass
    return [part for part in re.split(r"[;|]", text.strip("[]")) if part.strip()]


# --------------------------------------------------------------------------- #
# categories and semi-structured text
# --------------------------------------------------------------------------- #
def normalise_category(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """Collapse spelling variants to canonical values.

    Anything not in `aliases` is lowercased and punctuation-collapsed, so `Direct
    Mail` and `direct-mail` meet at `direct_mail` even without an entry.
    """
    def _one(value):
        if is_blank(value):
            return np.nan
        text = str(value).strip().lower()
        if text in aliases:
            return aliases[text]
        collapsed = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        return aliases.get(collapsed, collapsed)

    return series.map(_one)


def extract_keyed_number(value, keys: tuple[str, ...]) -> float:
    """Pull a number out of `{"email_opens": "5"}`, `opens=3`, or a bare `5`."""
    if is_blank(value):
        return np.nan
    text = str(value).strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text.replace("'", '"'))
            for key in keys:
                if key in payload:
                    return parse_number(payload[key])
        except (json.JSONDecodeError, TypeError):
            pass
    for key in keys:
        match = re.search(rf"{re.escape(key)}\D{{0,3}}(\d+)", text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return parse_number(text)


def postcode_prefix(value) -> str | float:
    """Coarsen a postcode to its leading letters or first two digits.

    Raw postcodes are near-unique, which makes them an identifier rather than a
    feature — the model memorises households instead of learning geography.
    """
    if is_blank(value):
        return np.nan
    text = re.sub(r"\s+", "", str(value).strip().upper())
    letters = re.match(r"^([A-Z]{1,3})", text)
    if letters:
        return letters.group(1)
    digits = re.match(r"^(\d{2})", text)
    return digits.group(1) if digits else np.nan
