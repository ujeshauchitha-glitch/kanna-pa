"""Exact money arithmetic.

Amounts are stored as integers in the currency's smallest unit (e.g.
paise for INR, cents for USD) — never as floats. Parsing from natural
language / CSV goes through `Decimal` and is quantized once, at the
boundary; every arithmetic operation afterwards is plain integer math on
`amount_minor`. This is what makes the finance subsystem's totals
deterministic and exact regardless of who or what asked for them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from core.errors import CurrencyMismatch, InvalidMoneyAmount

# Minor-unit exponent per ISO-4217 currency code. Extend as needed —
# unlisted currencies default to 2 (the common case).
_EXPONENTS: dict[str, int] = {
    "INR": 2, "USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "KWD": 3, "BHD": 3,
}

_SYMBOL_TO_CURRENCY = {
    "₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
}

_AMOUNT_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def exponent_for(currency: str) -> int:
    return _EXPONENTS.get(currency.upper(), 2)


@dataclass(frozen=True)
class Money:
    amount_minor: int
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "currency", self.currency.upper())

    @classmethod
    def from_decimal(cls, amount: Decimal, currency: str) -> "Money":
        exp = exponent_for(currency)
        quantum = Decimal(1).scaleb(-exp)
        quantized = amount.quantize(quantum, rounding=ROUND_HALF_UP)
        return cls(amount_minor=int(quantized.scaleb(exp)), currency=currency.upper())

    @classmethod
    def parse(cls, text: str, default_currency: str) -> "Money":
        """Parse an amount from natural language, e.g. '₹340', '340.50', 'INR 1,234', '$12'."""
        stripped = text.strip()

        currency = default_currency
        for symbol, code in _SYMBOL_TO_CURRENCY.items():
            if symbol in stripped:
                currency = code
                stripped = stripped.replace(symbol, "")
                break
        else:
            code_match = re.match(r"^([A-Za-z]{3})\b", stripped.strip())
            if code_match and code_match.group(1).upper() in _EXPONENTS:
                currency = code_match.group(1).upper()
                stripped = stripped[code_match.end():]

        amount_match = _AMOUNT_RE.search(stripped)
        if not amount_match:
            raise InvalidMoneyAmount(f"could not find a numeric amount in: {text!r}")

        raw = amount_match.group(0).replace(",", "")
        try:
            decimal_amount = Decimal(raw)
        except InvalidOperation as exc:
            raise InvalidMoneyAmount(f"could not parse amount: {raw!r}") from exc

        return cls.from_decimal(decimal_amount, currency)

    def to_decimal(self) -> Decimal:
        exp = exponent_for(self.currency)
        return Decimal(self.amount_minor).scaleb(-exp)

    def __str__(self) -> str:
        exp = exponent_for(self.currency)
        return f"{self.to_decimal():.{exp}f} {self.currency}"

    def _check_currency(self, other: "Money") -> None:
        if self.currency != other.currency:
            raise CurrencyMismatch(f"cannot combine {self.currency} with {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check_currency(other)
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check_currency(other)
        return Money(self.amount_minor - other.amount_minor, self.currency)

    def __lt__(self, other: "Money") -> bool:
        self._check_currency(other)
        return self.amount_minor < other.amount_minor

    def __le__(self, other: "Money") -> bool:
        self._check_currency(other)
        return self.amount_minor <= other.amount_minor

    def __gt__(self, other: "Money") -> bool:
        self._check_currency(other)
        return self.amount_minor > other.amount_minor

    def __ge__(self, other: "Money") -> bool:
        self._check_currency(other)
        return self.amount_minor >= other.amount_minor

    @classmethod
    def zero(cls, currency: str) -> "Money":
        return cls(0, currency.upper())
