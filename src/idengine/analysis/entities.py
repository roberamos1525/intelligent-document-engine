"""Rule-based named entity extraction (regular expressions plus gazetteers).

Recognised labels: EMAIL, URL, PHONE, MONEY, PERCENT, DATE, ORG, PERSON and
LOCATION. Organisation and person detection are heuristic: organisations are
capitalised phrases ending in a legal/institutional suffix (or listed in the
gazetteer), people are capitalised names preceded by an honorific or a
signature cue such as "Signed by".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from idengine.analysis.dates import find_dates

ORG_SUFFIXES = (
    "Inc", "Inc.", "Incorporated", "Corp", "Corp.", "Corporation", "LLC", "L.L.C.", "Ltd",
    "Ltd.", "Limited", "PLC", "GmbH", "AG", "S.A.", "SA", "B.V.", "N.V.", "Co.", "Company",
    "Group", "Holdings", "Partners", "Bank", "University", "Institute", "Foundation",
    "Association", "Agency", "Authority", "Council", "Ministry", "Hospital", "Clinic",
    "Laboratories", "Labs", "Systems", "Technologies", "Solutions", "Consulting",
)  # fmt: skip

DEFAULT_ORG_GAZETTEER = frozenset(
    {
        "European Union", "United Nations", "World Health Organization", "World Bank",
        "International Monetary Fund", "Federal Reserve", "European Central Bank",
        "Internal Revenue Service", "Securities and Exchange Commission", "NATO", "OECD",
    }
)  # fmt: skip

DEFAULT_LOCATION_GAZETTEER = frozenset(
    {
        "United States", "United Kingdom", "Canada", "Mexico", "Brazil", "Argentina", "Germany",
        "France", "Spain", "Italy", "Portugal", "Netherlands", "Belgium", "Switzerland",
        "Austria", "Ireland", "Sweden", "Norway", "Denmark", "Poland", "Japan", "China",
        "India", "Australia", "Singapore", "South Africa", "New York", "London", "Paris",
        "Berlin", "Madrid", "Rome", "Lisbon", "Amsterdam", "Tokyo", "Toronto", "Chicago",
        "San Francisco", "Los Angeles", "Boston", "Seattle", "Austin", "Dublin", "Zurich",
        "Munich", "Barcelona", "Sydney", "California", "Texas", "Florida", "Delaware",
        "Washington", "Oregon", "Illinois", "Massachusetts", "Nevada", "Ontario", "Bavaria",
    }
)  # fmt: skip

_HONORIFIC_WORDS = frozenset(
    {"Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Sir", "Madam", "Judge", "Hon"}
)
_HONORIFICS = r"(?:" + "|".join(sorted(_HONORIFIC_WORDS)) + r")\.?"
_NAME = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
_CAP_WORD = r"(?:[A-Z][A-Za-z0-9&'\-]*|of|and|for|the|&)"

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"')\]]+", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<![\w/.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?![\w/-])"
)
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
_SCALE = r"(?:\s?(?:thousand|million|billion|trillion|[kKmMbB]n?)\b)?"
_MONEY_RE = re.compile(
    rf"(?P<sym>[$€£¥])\s?(?P<amount>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?){_SCALE}"
    rf"|\b(?P<code>USD|EUR|GBP|JPY|CHF|CAD|AUD)\s?(?P<amount2>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?){_SCALE}"
    rf"|\b(?P<amount3>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?){_SCALE}\s(?P<word>dollars|euros|pounds)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)
_ORG_RE = re.compile(
    rf"\b([A-Z][A-Za-z0-9&'\-]*\s+(?:{_CAP_WORD}\s+){{0,4}}"
    rf"(?:{'|'.join(re.escape(s) for s in sorted(ORG_SUFFIXES, key=len, reverse=True))}))"
    r"(?![A-Za-z])"
)
_OF_ORG_RE = re.compile(
    r"\b(?:Department|Ministry|Office|Bureau|University|Bank|Institute|College|Board)"
    r" of (?:the )?[A-Z][a-z]+(?: [A-Z][a-z]+){0,3}"
)
_PERSON_HONORIFIC_RE = re.compile(rf"\b{_HONORIFICS}\s+((?:{_NAME}\s+){{0,2}}{_NAME})")
_PERSON_CUE_RE = re.compile(
    r"\b(?:[Ss]igned|[Pp]repared|[Aa]uthored|[Ww]ritten|[Aa]pproved|[Rr]eviewed|[Ll]ed|[Mm]anaged"
    r"|[Ss]ponsored) by"
    rf"\s+((?:{_NAME}\s+)?{_NAME}(?:\s+{_NAME})?)\b"
    rf"|\b(?:[Aa]ttn:?|[Cc]ontact:?|By)\s+((?:{_NAME}\s+)?{_NAME}(?:\s+{_NAME})?)\b"
)
# "Chief Financial Officer, Maria Gonzalez" / "CEO Jane Doe"
_PERSON_TITLE_RE = re.compile(
    r"\b(?:Chief [A-Z][a-z]+ Officer|CEO|CFO|CTO|COO|President|Director|Chair(?:man|woman|person)?)"
    rf",?\s+((?:{_NAME}\s+)?{_NAME})\b"
)
_STATE_RE = re.compile(
    r"\b(?:State|Commonwealth|Province|Republic) of [A-Z][a-z]+(?: [A-Z][a-z]+)?"
)


@dataclass(frozen=True, slots=True)
class Entity:
    label: str
    text: str
    normalized: str
    start: int
    end: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "text": self.text,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
        }


def _normalize_money(m: re.Match[str]) -> str:
    raw = m.group("amount") or m.group("amount2") or m.group("amount3") or "0"
    amount = float(raw.replace(",", ""))
    scale_text = m.group(0).lower()
    for word, factor in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if word in scale_text:
            amount *= factor
            break
    else:
        tail = re.search(r"\d\s?([kmb])n?$", scale_text)
        if tail:
            amount *= {"k": 1e3, "m": 1e6, "b": 1e9}[tail.group(1)]
    if m.group("sym"):
        currency = _CURRENCY_SYMBOLS[m.group("sym")]
    elif m.group("code"):
        currency = m.group("code").upper()
    else:
        currency = {"dollars": "USD", "euros": "EUR", "pounds": "GBP"}[m.group("word").lower()]
    return f"{currency} {amount:.2f}"


class EntityExtractor:
    """Configurable rule-based entity extractor."""

    def __init__(
        self,
        *,
        organizations: Iterable[str] = DEFAULT_ORG_GAZETTEER,
        locations: Iterable[str] = DEFAULT_LOCATION_GAZETTEER,
        day_first: bool = False,
    ) -> None:
        self.day_first = day_first
        self._gazetteers = {
            "ORG": self._compile(organizations),
            "LOCATION": self._compile(locations),
        }

    @staticmethod
    def _compile(names: Iterable[str]) -> re.Pattern[str] | None:
        ordered = sorted({n for n in names if n}, key=len, reverse=True)
        if not ordered:
            return None
        return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in ordered) + r")\b")

    def extract(self, text: str, *, limit: int = 500) -> list[Entity]:
        found: list[Entity] = []
        occupied: list[tuple[int, int]] = []

        def add(label: str, start: int, end: int, normalized: str | None = None) -> None:
            if any(start < e and s < end for s, e in occupied):
                return
            surface = text[start:end]
            found.append(Entity(label, surface, normalized or surface, start, end))
            occupied.append((start, end))

        for m in _EMAIL_RE.finditer(text):
            add("EMAIL", m.start(), m.end(), m.group(0).lower())
        for m in _URL_RE.finditer(text):
            url = m.group(0).rstrip(".,;:")
            add("URL", m.start(), m.start() + len(url), url)
        for d in find_dates(text, day_first=self.day_first):
            add("DATE", d.start, d.end, d.iso)
        for m in _MONEY_RE.finditer(text):
            add("MONEY", m.start(), m.end(), _normalize_money(m))
        for m in _PERCENT_RE.finditer(text):
            number = re.match(r"\d+(?:\.\d+)?", m.group(0))
            add("PERCENT", m.start(), m.end(), f"{number.group(0) if number else ''}%")
        for m in _PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 7 <= len(digits) <= 15:
                add(
                    "PHONE",
                    m.start(),
                    m.end(),
                    ("+" if m.group(0).startswith("+") else "") + digits,
                )
        for m in _OF_ORG_RE.finditer(text):
            add("ORG", m.start(), m.end())
        for m in _STATE_RE.finditer(text):
            add("LOCATION", m.start(), m.end())
        for label in ("ORG", "LOCATION"):
            pattern = self._gazetteers[label]
            if pattern is not None:
                for m in pattern.finditer(text):
                    add(label, m.start(), m.end())
        for m in _ORG_RE.finditer(text):
            start, end = m.span(1)
            first, _, rest = text[start:end].partition(" ")
            if first in ("The", "A", "An", "This", "Each", "Our", "Your"):
                start += len(first) + 1
                if " " not in rest.strip():
                    continue  # just "The Company": a defined term, not a name
            add("ORG", start, end, " ".join(text[start:end].split()))
        for pattern in (_PERSON_HONORIFIC_RE, _PERSON_TITLE_RE, _PERSON_CUE_RE):
            for m in pattern.finditer(text):
                group = next(g for g in range(1, (m.lastindex or 1) + 1) if m.group(g))
                if m.group(group).split()[0].rstrip(".") not in _HONORIFIC_WORDS:
                    add("PERSON", m.start(group), m.end(group))
        found.sort(key=lambda e: e.start)
        return found[:limit]


def summarize_entities(entities: list[Entity], *, per_label: int = 20) -> dict[str, list[str]]:
    """Unique normalised values per label, in order of first appearance."""
    grouped: dict[str, list[str]] = {}
    for entity in entities:
        values = grouped.setdefault(entity.label, [])
        if entity.normalized not in values and len(values) < per_label:
            values.append(entity.normalized)
    return grouped
