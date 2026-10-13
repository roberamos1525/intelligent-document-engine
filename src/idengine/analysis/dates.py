"""Date mention extraction and normalisation to ISO-8601."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4,
    "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}  # fmt: skip

_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
_YEAR = r"(?P<year>(?:19|20)\d{2})"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("iso", re.compile(rf"\b{_YEAR}-(?P<month>\d{{1,2}})-(?P<day>\d{{1,2}})\b")),
    (
        "month_day_year",
        re.compile(
            rf"\b(?P<mname>{_MONTH_ALT})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+{_YEAR}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "day_month_year",
        re.compile(
            rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mname>{_MONTH_ALT})\.?,?\s+{_YEAR}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "numeric",
        re.compile(rf"\b(?P<a>\d{{1,2}})[/.](?P<b>\d{{1,2}})[/.]{_YEAR}\b"),
    ),
    (
        "month_year",
        re.compile(rf"\b(?P<mname>{_MONTH_ALT})\.?\s+{_YEAR}\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True, slots=True)
class DateMention:
    text: str
    iso: str
    start: int
    end: int
    precision: str  # "day" or "month"


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _resolve(kind: str, m: re.Match[str], day_first: bool) -> tuple[date, str] | None:
    year = int(m.group("year"))
    if kind == "iso":
        d = _safe_date(year, int(m.group("month")), int(m.group("day")))
        return (d, "day") if d else None
    if kind in ("month_day_year", "day_month_year"):
        month = MONTHS[m.group("mname").lower()]
        d = _safe_date(year, month, int(m.group("day")))
        return (d, "day") if d else None
    if kind == "numeric":
        a, b = int(m.group("a")), int(m.group("b"))
        if a > 12:  # unambiguous day-first
            day_first = True
        elif b > 12:
            day_first = False
        month, day = (b, a) if day_first else (a, b)
        d = _safe_date(year, month, day)
        return (d, "day") if d else None
    month = MONTHS[m.group("mname").lower()]
    if month == 5 and m.group("mname").lower() == "may" and not m.group("mname")[0].isupper():
        return None  # the modal verb "may" followed by a year is not a date
    d = _safe_date(year, month, 1)
    return (d, "month") if d else None


def find_dates(text: str, *, day_first: bool = False, limit: int = 200) -> list[DateMention]:
    """Find date mentions; overlapping matches keep the most specific pattern."""
    mentions: list[DateMention] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < end and start < m.end() for start, end in taken):
                continue
            resolved = _resolve(kind, m, day_first)
            if resolved is None:
                continue
            value, precision = resolved
            iso = value.isoformat() if precision == "day" else value.isoformat()[:7]
            mentions.append(DateMention(m.group(0), iso, m.start(), m.end(), precision))
            taken.append((m.start(), m.end()))
            if len(mentions) >= limit:
                break
    mentions.sort(key=lambda d: d.start)
    return mentions


def parse_date(value: str) -> str | None:
    """Normalise a single free-form date string (e.g. from metadata) to ISO."""
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"(?:19|20)\d{2}-\d{2}-\d{2}(?:[T ].*)?", value):
        d = _safe_date(int(value[:4]), int(value[5:7]), int(value[8:10]))
        return d.isoformat() if d else None
    found = find_dates(value, limit=1)
    return found[0].iso if found else None
