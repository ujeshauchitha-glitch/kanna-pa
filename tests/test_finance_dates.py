from __future__ import annotations

import pytest

from finance.dates import normalize_date


@pytest.mark.parametrize("raw,expected", [
    ("2024-03-15", "2024-03-15"),
    ("15/03/2024", "2024-03-15"),
    ("03/15/2024", "2024-03-15"),
    ("15-03-2024", "2024-03-15"),
    ("March 15, 2024", "2024-03-15"),
    ("Mar 15, 2024", "2024-03-15"),
    ("15 March 2024", "2024-03-15"),
    ("15 Mar 2024", "2024-03-15"),
    ("  2024-03-15  ", "2024-03-15"),
])
def test_normalize_date_formats(raw, expected):
    assert normalize_date(raw) == expected


def test_normalize_date_unrecognized_raises():
    with pytest.raises(ValueError):
        normalize_date("not a date at all")
