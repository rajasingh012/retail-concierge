"""Structured brief extraction and parsing for the shopping agent."""
from __future__ import annotations

import re

MAX_CLARIFICATIONS = 2

# Hardcoded approximate rates for hackathon use (not a live FX API).
_CURRENCY_TO_USD: dict[str, float] = {
    "$": 1.0, "usd": 1.0, "dollar": 1.0, "dollars": 1.0,
    "€": 1.08, "eur": 1.08, "euro": 1.08, "euros": 1.08,
    "£": 1.26, "gbp": 1.26, "pound": 1.26, "pounds": 1.26,
    "₹": 0.012, "inr": 0.012, "rupee": 0.012, "rupees": 0.012,
    "¥": 0.0065, "jpy": 0.0065, "yen": 0.0065,
}

_DIMENSION_UNIT_TO_CM: dict[str, float] = {
    "cm": 1.0, "centimeter": 1.0, "centimeters": 1.0,
    "mm": 0.1, "millimeter": 0.1, "millimeters": 0.1,
    "m": 100.0, "meter": 100.0, "meters": 100.0,
    "in": 2.54, "inch": 2.54, "inches": 2.54,
    '"': 2.54,
    "ft": 30.48, "foot": 30.48, "feet": 30.48,
    "'": 30.48,
}

_CURRENCY_RE = re.compile(
    r"(?P<amount>\d[\d,.]*)\s*(?P<unit>" + "|".join(re.escape(k) for k in _CURRENCY_TO_USD) + r")"
    r"|(?P<unit2>" + "|".join(re.escape(k) for k in _CURRENCY_TO_USD) + r")\s*(?P<amount2>\d[\d,.]*)",
    re.IGNORECASE,
)

_DIMENSION_RE = re.compile(
    r"(?P<value>\d[\d.]*)\s*(?P<unit>" + "|".join(re.escape(k) for k in _DIMENSION_UNIT_TO_CM) + r")\b",
    re.IGNORECASE,
)

_QUANTITY_RE = re.compile(
    r"\b(?:a\s+)?(?P<qty>\d+)\s*(?:x\s*)?(?=[\w-]+(?:s\b|es\b))"
    r"|\b(?P<words>pair|set|pack|dozen|bundle)\s*(?:of\s+)?",
    re.IGNORECASE,
)

_BUDGET_TRIGGERS = re.compile(
    r"\b(under|up to|less than|around|about|budget|max|spend|price|cost)\b",
    re.IGNORECASE,
)


def _extract_budget_usd(text: str) -> tuple[float, str]:
    """Parse a budget mention from free text to USD.

    Returns (amount_usd, note). 0.0 means no parseable budget.
    """
    has_budget_trigger = bool(_BUDGET_TRIGGERS.search(text))
    matches = _CURRENCY_RE.findall(text)
    if not matches:
        # No currency pattern found. Try bare numbers with a budget trigger.
        if has_budget_trigger:
            numbers = re.findall(r"\b\d{2,}\b", text)
            if numbers:
                return 0.0, f"Non-numeric budget mentioned ({numbers[0]}) — clarify"
        return 0.0, ""

    best_amount = 0.0
    best_unit = ""
    for match in matches:
        amount_str = match[0] or match[3]
        unit = (match[1] or match[2]).lower()
        amount = float(amount_str.replace(",", ""))
        factor = _CURRENCY_TO_USD.get(unit, 0.0)
        if factor > 0 and amount * factor > best_amount:
            best_amount = amount * factor
            best_unit = unit

    if best_amount == 0.0:
        return 0.0, "Unrecognized currency — cannot convert"

    unit_label = best_unit if best_unit in ("$", "€", "£", "¥", "₹") else best_unit.upper()
    note = f"Converted from {unit_label}{best_amount / _CURRENCY_TO_USD.get(best_unit, 1.0):.0f}"
    return round(best_amount, 2), note


def _extract_dimensions(text: str) -> tuple[dict[str, float], str]:
    """Parse dimension mentions from free text to cm.

    Returns {height, width, length} (each 0.0 = unset) and a note string.
    """
    dims: dict[str, float] = {"height": 0.0, "width": 0.0, "length": 0.0}
    notes: list[str] = []
    matches = _DIMENSION_RE.findall(text.lower())
    values: list[float] = []
    for val_str, unit in matches:
        val = float(val_str)
        factor = _DIMENSION_UNIT_TO_CM.get(unit, 1.0)
        values.append(val * factor)

    if not values:
        return dims, ""

    # Single dimension — ambiguous (could be height, width, or diagonal).
    # Use as max_dimension_cm via height (search_catalog treats any-dim-fit).
    if len(values) == 1:
        # Heuristic: if the original text mentions "wide" or "width", set width.
        if re.search(r"\b(wide|width)\b", text, re.IGNORECASE):
            dims["width"] = round(values[0], 2)
        elif re.search(r"\b(depth|deep|length)\b", text, re.IGNORECASE):
            dims["length"] = round(values[0], 2)
        else:
            # Default: set height (used as ceiling by search_catalog).
            dims["height"] = round(values[0], 2)
        notes.append(f"Single dimension parsed: {values[0]:.1f} cm")
    elif len(values) == 2:
        # Likely width × depth or height × width.
        dims["width"] = round(max(values), 2)
        dims["length"] = round(min(values), 2)
        notes.append(f"Two dimensions parsed: W={dims['width']:.1f} × L={dims['length']:.1f} cm")
    else:
        dims["height"] = round(max(values), 2)
        others = sorted(values, reverse=True)
        dims["width"] = round(others[1], 2)
        dims["length"] = round(others[2], 2)
        notes.append(f"Three dimensions parsed: H={dims['height']:.1f} × W={dims['width']:.1f} × L={dims['length']:.1f} cm")

    return dims, "; ".join(notes)


def _extract_quantity(text: str) -> int:
    """Extract a requested quantity from free text."""
    match = _QUANTITY_RE.search(text.lower())
    if not match:
        return 1
    if match.group("qty"):
        return max(1, int(match.group("qty")))
    word = match.group("words")
    if word == "pair":
        return 2
    if word == "dozen":
        return 12
    if word in ("set", "pack", "bundle"):
        return 1  # ambiguous; don't guess
    return 1


def _make_budget_note(
    raw_text: str, budget_usd: float, budget_source: str
) -> list[str]:
    """Build the budget-related assumption line."""
    if budget_usd > 0:
        return [
            f"User specified ~${budget_usd:.0f} budget ({budget_source}). "
            "The ABO catalog has no price column — budget cannot be enforced against listings."
        ]
    if "under" in raw_text.lower() or "budget" in raw_text.lower():
        return ["User mentioned a budget constraint without a numeric value — treated as unset"]
    return []


EXTRACT_BRIEF_TOOL = "extract_brief"
