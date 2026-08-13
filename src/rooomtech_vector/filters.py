from __future__ import annotations

from typing import Any


def _lookup(metadata: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _compare(exists: bool, actual: Any, op: str, expected: Any) -> bool:
    if op == "$exists":
        return exists is bool(expected)
    if not exists:
        return op in {"$ne", "$nin"}
    if op == "$eq":
        return actual == expected
    if op == "$ne":
        return actual != expected
    if op == "$in":
        return actual in expected
    if op == "$nin":
        return actual not in expected
    if op == "$gt":
        return actual > expected
    if op == "$gte":
        return actual >= expected
    if op == "$lt":
        return actual < expected
    if op == "$lte":
        return actual <= expected
    if op == "$contains":
        if isinstance(actual, (list, tuple, set, str)):
            return expected in actual
        return False
    if op == "$not_contains":
        if isinstance(actual, (list, tuple, set, str)):
            return expected not in actual
        return True
    raise ValueError(f"unsupported filter operator: {op}")


def matches_filter(metadata: dict[str, Any], expression: dict[str, Any] | None) -> bool:
    """Evaluate a JSON metadata filter with nested metadata paths."""
    if not expression:
        return True

    if "$and" in expression:
        clauses = expression["$and"]
        if not isinstance(clauses, list):
            raise ValueError("$and must contain a list")
        return all(matches_filter(metadata, clause) for clause in clauses)

    if "$or" in expression:
        clauses = expression["$or"]
        if not isinstance(clauses, list):
            raise ValueError("$or must contain a list")
        return any(matches_filter(metadata, clause) for clause in clauses)

    if "$not" in expression:
        return not matches_filter(metadata, expression["$not"])

    for field, condition in expression.items():
        exists, actual = _lookup(metadata, field)
        if isinstance(condition, dict):
            for op, expected in condition.items():
                if not _compare(exists, actual, op, expected):
                    return False
        elif not _compare(exists, actual, "$eq", condition):
            return False
    return True
