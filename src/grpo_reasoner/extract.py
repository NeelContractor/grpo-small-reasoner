"""Answer extraction and numeric matching for GSM8K-style completions."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_NUMBER_RE = re.compile(r"[-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_GSM8K_MARKER = "####"


def _parse_value(token: str) -> Fraction | None:
    """Parse an integer, decimal, or simple fraction into an exact value."""
    cleaned = token.strip().replace(",", "").replace("\u2212", "-").replace(" ", "")
    cleaned = cleaned.removeprefix("+")
    if not cleaned:
        return None
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _canonical_number(token: str) -> str | None:
    """Return a stable string form of a numeric token."""
    value = _parse_value(token)
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    cleaned = token.strip().replace(",", "").replace("\u2212", "-").removeprefix("+")
    if "/" in cleaned:
        return f"{value.numerator}/{value.denominator}"
    try:
        decimal = Decimal(cleaned)
    except InvalidOperation:
        return f"{value.numerator}/{value.denominator}"
    return format(decimal.normalize(), "f")


def _last_number(text: str) -> str | None:
    for match in reversed(list(_NUMBER_RE.finditer(text))):
        canonical = _canonical_number(match.group())
        if canonical is not None:
            return canonical
    return None


def extract_answer(text: str | None) -> str | None:
    """Extract the final numeric answer from a model completion.

    Priority order:
    1. Contents of the last ``<answer>...</answer>`` block.
    2. Text after the last GSM8K ``####`` marker.
    3. The last non-empty line, falling back to the last number in the text.
    """
    if not text:
        return None

    tag_matches = list(_ANSWER_TAG_RE.finditer(text))
    if tag_matches:
        return _last_number(tag_matches[-1].group(1))

    marker_index = text.rfind(_GSM8K_MARKER)
    if marker_index != -1:
        return _last_number(text[marker_index + len(_GSM8K_MARKER) :])

    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        answer = _last_number(lines[-1])
        if answer is not None:
            return answer

    return _last_number(text)


def numbers_in_text(text: str | None) -> list[str]:
    """Return every canonical numeric token found in text, in order."""
    if not text:
        return []
    numbers: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        canonical = _canonical_number(match.group())
        if canonical is not None:
            numbers.append(canonical)
    return numbers


def text_mentions_answer(text: str | None, answer: str | None) -> bool:
    """Return whether any numeric token in text matches the answer."""
    return any(answers_match(number, answer) for number in numbers_in_text(text))


def extract_gold_answer(reference: str | None) -> str | None:
    """Extract the reference answer from a GSM8K ``answer`` field."""
    if not reference:
        return None
    marker_index = reference.rfind(_GSM8K_MARKER)
    if marker_index != -1:
        return _last_number(reference[marker_index + len(_GSM8K_MARKER) :])
    return _last_number(reference)


def answers_match(prediction: str | None, gold: str | None) -> bool:
    """Compare two answers by exact numeric value."""
    if prediction is None or gold is None:
        return False
    prediction_value = _parse_value(prediction)
    if prediction_value is None:
        extracted = extract_answer(prediction)
        prediction_value = _parse_value(extracted) if extracted is not None else None
    gold_value = _parse_value(gold)
    if gold_value is None:
        extracted_gold = extract_gold_answer(gold)
        gold_value = _parse_value(extracted_gold) if extracted_gold is not None else None
    return prediction_value is not None and prediction_value == gold_value
