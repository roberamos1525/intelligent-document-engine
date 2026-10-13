"""Metadata filter language shared by the vector and BM25 indexes.

Filters are JSON objects in a MongoDB-like syntax::

    {"category": "finance"}                                   # equality
    {"category": {"$in": ["finance", "legal"]}}
    {"document_date": {"$gte": "2024-01-01", "$lt": "2025-01-01"}}
    {"tags": "urgent"}                                        # list membership
    {"$or": [{"language": "en"}, {"language": "de"}]}
    {"$not": {"doc_type": "html"}}

A filter is validated and compiled once into a Python predicate.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from idengine.errors import ValidationError

Predicate = Callable[[Mapping[str, Any]], bool]

_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_DEPTH = 8
_MAX_LIST = 1000


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    try:
        if op == "$gt":
            return bool(actual > expected)
        if op == "$gte":
            return bool(actual >= expected)
        if op == "$lt":
            return bool(actual < expected)
        return bool(actual <= expected)
    except TypeError:
        return False  # incomparable types (e.g. str vs int) never match


def _equals(actual: Any, expected: Any) -> bool:
    if isinstance(actual, list | tuple | set):
        return expected in actual
    return bool(actual == expected)


def _scalar(value: Any, where: str) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValidationError(f"filter value at {where} must be a scalar")


def _field_predicate(field: str, condition: Any, where: str) -> Predicate:
    if not isinstance(condition, Mapping):
        expected = _scalar(condition, where)
        return lambda meta: _equals(meta.get(field), expected)

    checks: list[Predicate] = []
    for op, operand in condition.items():
        path = f"{where}.{op}"
        if op == "$eq":
            expected = _scalar(operand, path)
            checks.append(lambda meta, e=expected: _equals(meta.get(field), e))
        elif op == "$ne":
            expected = _scalar(operand, path)
            checks.append(lambda meta, e=expected: not _equals(meta.get(field), e))
        elif op in ("$in", "$nin"):
            if not isinstance(operand, list) or len(operand) > _MAX_LIST:
                raise ValidationError(f"{path} expects a list of at most {_MAX_LIST} values")
            options = [_scalar(v, path) for v in operand]
            negate = op == "$nin"
            checks.append(
                lambda meta, o=options, n=negate: (
                    n != any(_equals(meta.get(field), value) for value in o)
                )
            )
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            expected = _scalar(operand, path)
            if expected is None or isinstance(expected, bool):
                raise ValidationError(f"{path} expects a number or string")
            checks.append(lambda meta, o=op, e=expected: _compare(o, meta.get(field), e))
        elif op == "$exists":
            if not isinstance(operand, bool):
                raise ValidationError(f"{path} expects true or false")
            checks.append(
                lambda meta, want=operand: (meta.get(field) not in (None, "", [])) is want
            )
        elif op == "$contains":
            needle = _scalar(operand, path)
            if not isinstance(needle, str):
                raise ValidationError(f"{path} expects a string")
            lowered = needle.lower()
            checks.append(lambda meta, n=lowered: _contains(meta.get(field), n))
        else:
            raise ValidationError(f"unknown filter operator '{op}' at {where}")
    return lambda meta: all(check(meta) for check in checks)


def _contains(actual: Any, needle: str) -> bool:
    if isinstance(actual, str):
        return needle in actual.lower()
    if isinstance(actual, list | tuple):
        return any(isinstance(v, str) and needle in v.lower() for v in actual)
    return False


def _compile(spec: Any, where: str, depth: int) -> Predicate:
    if depth > _MAX_DEPTH:
        raise ValidationError("filter is nested too deeply")
    if not isinstance(spec, Mapping):
        raise ValidationError(f"filter at {where} must be an object")
    predicates: list[Predicate] = []
    for key, value in spec.items():
        path = f"{where}.{key}" if where else key
        if key in ("$and", "$or"):
            if not isinstance(value, list) or not value:
                raise ValidationError(f"{path} expects a non-empty list")
            parts = [_compile(item, f"{path}[{i}]", depth + 1) for i, item in enumerate(value)]
            if key == "$and":
                predicates.append(lambda meta, p=parts: all(part(meta) for part in p))
            else:
                predicates.append(lambda meta, p=parts: any(part(meta) for part in p))
        elif key == "$not":
            inner = _compile(value, path, depth + 1)
            predicates.append(lambda meta, i=inner: not i(meta))
        elif key.startswith("$"):
            raise ValidationError(f"unknown filter operator '{key}'")
        elif not _FIELD_RE.match(key):
            raise ValidationError(f"invalid filter field name '{key}'")
        else:
            predicates.append(_field_predicate(key, value, path))
    return lambda meta: all(p(meta) for p in predicates)


def compile_filter(spec: Mapping[str, Any] | None) -> Predicate | None:
    """Validate and compile a filter specification; ``None``/empty means no filter."""
    if not spec:
        return None
    return _compile(spec, "", 0)
