"""Automated, server-side-only proxy for "does this annotated value match
the visible document": checks whether a field's value is consistent with
OCR text found near its bounding box.

This is NOT a substitute for human visual review -- see
`docs/company-benchmark-diagnosis-report.md` sections 0 and 2.2 for why it
exists: pulling real page images off the server to eyeball them was (rightly)
blocked by this project's private-data sandboxing, so this proxy gets a
much larger-n quality signal without moving any private image or raw text
off the server. It can't tell if OCR picked up the *wrong* cell that
coincidentally looks similar to the right one, and it inherits OCR's own
error rate -- treat its mismatch rate as a floor, not a final number.

Every function here takes already-loaded tokens/values and returns only a
match/mismatch/etc. outcome label -- never the underlying private content.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from financial_vlm.integrations.evolution_ai_datasets_adapter import BoundingBox, OCRToken

_NUMERIC_RE = re.compile(r"[-+]?[\d,]*\.?\d+")
# "x"/"X" is the standard multiple notation ("8.5x EBITDA") -- distinct from
# the currency-magnitude letters but stripped the same way (confirmed real
# case, 2026-08-12: valuation_multiple values like "8.5x" were falling into
# annotation_unparseable).
_MAGNITUDE_SUFFIX_RE = re.compile(r"\s*(bn|mn|tn|mm|kn|[bmktx])\s*$", re.IGNORECASE)
# Confirmed real case, 2026-08-12: EV/sales annotated in EUR/JPY ("€
# 6,276,908", "¥ 4.3") were falling into annotation_unparseable -- only "$"
# was stripped. Not exhaustive (no attempt at full ISO 4217 coverage), just
# the symbols actually observed in this dataset so far.
_CURRENCY_SYMBOLS = "$€¥£"
# OCR sometimes splits a comma-grouped number into separate word tokens
# ("1," and "852"), which `_NUMERIC_RE` then reads as two unrelated numbers
# (1 and 852) instead of one (1852) -- confirmed real case, 2026-08-12,
# where the annotation, bbox, and OCR all agreed on "1,852" but the check
# still reported a mismatch. Rejoining a comma immediately followed by
# whitespace-then-digits before running `_NUMERIC_RE` fixes this without
# touching genuinely separate numbers (which won't have a bare comma-space
# between them in OCR reading order as often).
_COMMA_SPLIT_RE = re.compile(r",\s+(?=\d)")
# Confirmed real case, 2026-08-12: two "mismatch" outcomes turned out to be
# OCR reading garbage near the field ("Qr723" for "Q1", "BO" for "83.0") at
# confidence 26 and 46 respectively, vs. ~99 for tokens that read correctly
# elsewhere in this dataset. Below this threshold, a "mismatch" says more
# about OCR quality than about the annotation -- reported as a separate
# outcome so it isn't silently treated as equal evidence to a
# high-confidence mismatch.
_LOW_OCR_CONFIDENCE_THRESHOLD = 60
# 10/100 catch a dropped decimal point (a common OCR failure mode -- "22.4"
# read back as "224"), not just genuine currency-unit conventions
# (thousands/millions). See docs/company-benchmark-diagnosis-report.md's
# calibration note: a value flagged "mismatch" that's actually a ratio-10
# OCR misread is not an annotation defect. Tolerance is much tighter for
# 10/100 than for 1000/1e6: a true decimal-point drop produces an *exact*
# power-of-ten ratio, whereas two genuinely different numbers can land near
# a 10x ratio by coincidence (100 vs. 999 is ~0.1001, not ~0.1) -- a loose
# tolerance here would misclassify a real mismatch as a formatting quirk.
_SCALE_FACTOR_TOLERANCES: dict[int, float] = {
    10: 0.0005,
    100: 0.0005,
    1000: 0.02,
    1_000_000: 0.02,
}


def normalize_numeric(value: str) -> float | None:
    """Strip currency-symbol/percent/thousands-separator formatting and a
    trailing magnitude or multiple suffix (`$1.2B`, `3.4Mn`, `8.5x`, ...) --
    OCR text near the same bbox reads the same suffixed form, so
    `_NUMERIC_RE` picks up the bare number there too; stripping it here
    (rather than converting it to an absolute value) keeps both sides
    comparable on the same basis."""

    for symbol in _CURRENCY_SYMBOLS:
        value = value.replace(symbol, "")
    value = value.replace(",", "").replace("%", "").strip()
    value = value.replace("(", "-").replace(")", "")
    value = _MAGNITUDE_SUFFIX_RE.sub("", value).strip()
    try:
        return round(float(value), 2)
    except ValueError:
        return None


def _close_relative(a: float, b: float, relative_tolerance: float) -> bool:
    return abs(a - b) <= max(abs(b), 1e-9) * relative_tolerance


def _is_scale_or_sign_variant(target: float, ocr_number: float) -> bool:
    """True if `target` is `ocr_number` up to a sign flip or a 10x/100x/
    1000x/1e6x scale factor (either direction) -- a formatting/OCR
    difference, not a genuine value disagreement."""

    if ocr_number == 0:
        return False
    if _close_relative(target, -ocr_number, 0.02):
        return True
    ratio = target / ocr_number
    return any(
        _close_relative(ratio, factor, tolerance) or _close_relative(ratio, 1 / factor, tolerance)
        for factor, tolerance in _SCALE_FACTOR_TOLERANCES.items()
    )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def tokens_near_bbox(tokens: Sequence[OCRToken], field_bbox: BoundingBox, margin: int = 8) -> list[OCRToken]:
    """OCR tokens whose center falls inside `field_bbox` expanded by
    `margin` pixels on each side, left-to-right reading order."""

    top, left, bottom, right = field_bbox
    top -= margin
    left -= margin
    bottom += margin
    right += margin
    near = []
    for token in tokens:
        center_y = (token.top + token.bottom) / 2
        center_x = (token.left + token.right) / 2
        if top <= center_y <= bottom and left <= center_x <= right:
            near.append(token)
    near.sort(key=lambda t: t.left)
    return near


def _is_low_confidence(tokens: Sequence[OCRToken], threshold: int = _LOW_OCR_CONFIDENCE_THRESHOLD) -> bool:
    confidences = [t.confidence for t in tokens if t.confidence is not None]
    if not confidences:
        return False
    return (sum(confidences) / len(confidences)) < threshold


def check_value_against_ocr(field_value: str, data_type: str, near_tokens: Sequence[OCRToken]) -> str:
    """One of: "no_ocr_nearby", "annotation_unparseable", "match",
    "match_scale_or_sign", "match_word_set", "mismatch",
    "mismatch_low_ocr_confidence" (a "mismatch" where the nearby OCR tokens'
    average confidence is below `_LOW_OCR_CONFIDENCE_THRESHOLD` -- likely an
    OCR failure, not an annotation error; see that constant's docstring)."""

    concat = " ".join(token.text for token in near_tokens)
    if not concat.strip():
        return "no_ocr_nearby"

    if data_type in ("monetary", "numerical"):
        target = normalize_numeric(field_value)
        if target is None:
            return "annotation_unparseable"
        concat_for_numbers = _COMMA_SPLIT_RE.sub(",", concat)
        ocr_numbers = [
            n for n in (normalize_numeric(m.group(0)) for m in _NUMERIC_RE.finditer(concat_for_numbers)) if n is not None
        ]
        if any(abs(n - target) < 0.5 for n in ocr_numbers):
            return "match"
        if any(_is_scale_or_sign_variant(target, n) for n in ocr_numbers):
            return "match_scale_or_sign"
        return "mismatch_low_ocr_confidence" if _is_low_confidence(near_tokens) else "mismatch"

    target = normalize_text(field_value)
    concat_norm = normalize_text(concat)
    if target and (target in concat_norm or concat_norm in target):
        return "match"
    target_words = set(target.split())
    concat_words = set(concat_norm.split())
    if target_words and target_words.issubset(concat_words):
        return "match_word_set"
    return "mismatch_low_ocr_confidence" if _is_low_confidence(near_tokens) else "mismatch"


def summarize_outcomes(outcomes: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """`outcomes` is a list of (field_name, outcome) pairs -- aggregates
    counts overall and per field. Never touches the underlying values."""

    overall: Counter[str] = Counter()
    by_field: dict[str, Counter[str]] = defaultdict(Counter)
    for field_name, outcome in outcomes:
        overall[outcome] += 1
        by_field[field_name][outcome] += 1

    total = len(outcomes)
    return {
        "sample_n": total,
        "outcome_counts": dict(overall.most_common()),
        "outcome_rates": {k: v / total for k, v in overall.items()} if total else {},
        "outcome_by_field": {name: dict(counts) for name, counts in by_field.items()},
    }
