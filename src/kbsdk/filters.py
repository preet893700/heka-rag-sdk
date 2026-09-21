"""The SDK's metadata filter dialect, plus an in-Python evaluator.

    {"region": "EU"}                              equality (list-valued metadata: membership)
    {"grade": {"$in": ["L3", "L4"]}}              $eq $ne $gt $gte $lt $lte $in $nin
    {"$and": [f1, f2]}, {"$or": [f1, f2]}         boolean combinations

Vector-store adapters translate this dialect to their native one; stores without native filtering
(like the built-in local store) evaluate it with `matches`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kbsdk.errors import ConfigError
from kbsdk.types import Filter

_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
_BOOLEAN = {"$and", "$or"}


def _is_operator_dict(cond: Any) -> bool:
    return isinstance(cond, dict) and bool(cond) and all(str(k).startswith("$") for k in cond)


def validate_filter(flt: Filter | None) -> None:
    """Raise `ConfigError` if the filter uses unknown operators or malformed operands."""
    if not flt:
        return
    if not isinstance(flt, dict):
        raise ConfigError(f"A filter must be a mapping, got {type(flt).__name__}")
    for key, cond in flt.items():
        if key in _BOOLEAN:
            if not isinstance(cond, list) or not cond:
                raise ConfigError(f"{key} needs a non-empty list of filters")
            for sub in cond:
                validate_filter(sub)
        elif str(key).startswith("$"):
            raise ConfigError(f"Unknown filter operator {key!r}")
        elif _is_operator_dict(cond):
            for op, operand in cond.items():
                if op not in _OPERATORS:
                    raise ConfigError(f"Unknown filter operator {op!r} on field {key!r}")
                if op in {"$in", "$nin"} and not isinstance(operand, list | tuple | set):
                    raise ConfigError(f"{op} on field {key!r} needs a list")


def combine(*filters: Filter | None) -> Filter | None:
    """AND together any number of optional filters."""
    present = [f for f in filters if f]
    if not present:
        return None
    return present[0] if len(present) == 1 else {"$and": present}


def _equal(value: Any, operand: Any) -> bool:
    if isinstance(value, list | tuple | set):
        return operand in value
    return bool(value == operand)


def _member(value: Any, operand: Any) -> bool:
    if isinstance(value, list | tuple | set):
        return any(item in operand for item in value)
    return value in operand


def _compare(op: str, value: Any, operand: Any) -> bool:
    if value is None:
        return False
    try:
        if op == "$gt":
            return bool(value > operand)
        if op == "$gte":
            return bool(value >= operand)
        if op == "$lt":
            return bool(value < operand)
        return bool(value <= operand)
    except TypeError:
        return False


def _apply(op: str, value: Any, operand: Any) -> bool:
    if op == "$eq":
        return _equal(value, operand)
    if op == "$ne":
        return not _equal(value, operand)
    if op == "$in":
        return _member(value, operand)
    if op == "$nin":
        return not _member(value, operand)
    return _compare(op, value, operand)


def matches(flt: Filter | None, metadata: Mapping[str, Any]) -> bool:
    """True if `metadata` satisfies `flt`. An empty or missing filter matches everything."""
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches(sub, metadata) for sub in cond):
                return False
        elif key == "$or":
            if not any(matches(sub, metadata) for sub in cond):
                return False
        else:
            value = metadata.get(key)
            if _is_operator_dict(cond):
                if not all(_apply(op, value, operand) for op, operand in cond.items()):
                    return False
            elif not _equal(value, cond):
                return False
    return True
