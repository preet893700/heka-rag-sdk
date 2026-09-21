import pytest

from heka.rag import ConfigError
from heka.rag.filters import combine, matches, validate_filter

META = {"region": "EU", "grade": "L3", "roles": ["hr", "all"], "year": 2025, "note": None}


@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        (None, True),
        ({}, True),
        ({"region": "EU"}, True),
        ({"region": "US"}, False),
        ({"roles": "hr"}, True),  # list-valued metadata: equality means membership
        ({"roles": "finance"}, False),
        ({"grade": {"$in": ["L3", "L4"]}}, True),
        ({"grade": {"$nin": ["L3"]}}, False),
        ({"roles": {"$in": ["finance", "hr"]}}, True),
        ({"year": {"$gte": 2025}}, True),
        ({"year": {"$gt": 2025}}, False),
        ({"year": {"$gte": 2020, "$lt": 2030}}, True),
        ({"region": {"$ne": "US"}}, True),
        ({"missing": {"$ne": "x"}}, True),
        ({"missing": {"$gt": 1}}, False),
        ({"note": {"$gt": 1}}, False),
        ({"year": {"$gt": "text"}}, False),  # incomparable types never match
        ({"$and": [{"region": "EU"}, {"grade": "L3"}]}, True),
        ({"$and": [{"region": "EU"}, {"grade": "L9"}]}, False),
        ({"$or": [{"region": "US"}, {"grade": "L3"}]}, True),
        ({"$or": [{"region": "US"}, {"grade": "L9"}]}, False),
        ({"region": "EU", "grade": "L9"}, False),  # top-level keys are ANDed
    ],
)
def test_matches(flt, expected):
    assert matches(flt, META) is expected


def test_validate_accepts_good_filters():
    validate_filter({"$and": [{"a": 1}, {"b": {"$in": [1, 2]}}]})
    validate_filter(None)


@pytest.mark.parametrize(
    "bad",
    [{"$nor": []}, {"a": {"$regex": "x"}}, {"$and": []}, {"$or": "x"}, {"a": {"$in": 5}}],
)
def test_validate_rejects_bad_filters(bad):
    with pytest.raises(ConfigError):
        validate_filter(bad)


def test_combine():
    assert combine(None, None) is None
    assert combine({"a": 1}, None) == {"a": 1}
    assert combine({"a": 1}, {"b": 2}) == {"$and": [{"a": 1}, {"b": 2}]}
