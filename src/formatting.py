"""Small presentation helpers for prices in Indian rupees (INR).

Indian digit grouping is not the same as the Western style: 420000 is written
"4,20,000" (lakh/crore grouping). These helpers keep the demo and the
Streamlit app consistent.
"""

from __future__ import annotations


def format_inr(value: float) -> str:
    """Format a number with Indian digit grouping, e.g. 420000 -> 4,20,000."""
    value = int(round(value))
    negative = value < 0
    digits = str(abs(value))
    if len(digits) <= 3:
        grouped = digits
    else:
        last3 = digits[-3:]
        rest = digits[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{'-' if negative else ''}{grouped}"


def format_rupees(value: float) -> str:
    """Format with the 'Rs' prefix used in the UI."""
    return f"Rs {format_inr(value)}"


def format_lakh(value: float) -> str:
    """Human-friendly lakh/crore form, e.g. 420000 -> 'Rs 4.2 lakh'."""
    if value >= 1_00_00_000:            # 1 crore
        return f"Rs {value / 1_00_00_000:.2f} crore"
    if value >= 1_00_000:               # 1 lakh
        return f"Rs {value / 1_00_000:.2f} lakh"
    return format_rupees(value)
