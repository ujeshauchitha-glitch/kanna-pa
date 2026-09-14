from __future__ import annotations

from decimal import Decimal

import pytest

from core.errors import CurrencyMismatch, InvalidMoneyAmount
from finance.money import Money


def test_parse_rupee_symbol():
    money = Money.parse("₹340", "USD")
    assert money.amount_minor == 34000
    assert money.currency == "INR"


def test_parse_plain_number_uses_default_currency():
    money = Money.parse("340.50", "USD")
    assert money.amount_minor == 34050
    assert money.currency == "USD"


def test_parse_thousands_separator():
    money = Money.parse("1,234.56", "USD")
    assert money.amount_minor == 123456


def test_parse_currency_code_prefix():
    money = Money.parse("INR 1234", "USD")
    assert money.currency == "INR"
    assert money.amount_minor == 123400


def test_parse_within_sentence():
    money = Money.parse("I spent ₹340 on lunch", "USD")
    assert money.amount_minor == 34000
    assert money.currency == "INR"


def test_parse_no_amount_raises():
    with pytest.raises(InvalidMoneyAmount):
        Money.parse("no numbers here", "USD")


def test_rounding_half_up():
    money = Money.from_decimal(Decimal("10.005"), "USD")
    assert money.amount_minor == 1001  # 10.005 rounds up to 10.01 (2 decimal places)


def test_yen_has_zero_decimal_exponent():
    money = Money.from_decimal(Decimal("100"), "JPY")
    assert money.amount_minor == 100


def test_addition_same_currency():
    a = Money(1000, "INR")
    b = Money(500, "INR")
    assert (a + b).amount_minor == 1500


def test_addition_different_currency_raises():
    a = Money(1000, "INR")
    b = Money(500, "USD")
    with pytest.raises(CurrencyMismatch):
        a + b


def test_comparison_operators():
    a = Money(1000, "INR")
    b = Money(2000, "INR")
    assert a < b
    assert b > a
    assert a <= Money(1000, "INR")


def test_str_formatting():
    assert str(Money(34000, "INR")) == "340.00 INR"


def test_zero():
    assert Money.zero("USD").amount_minor == 0
