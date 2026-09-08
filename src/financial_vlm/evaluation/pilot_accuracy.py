"""Accuracy helpers for the SynFinTabs pilot."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Any, Iterable, Mapping, Sequence


ANSWER_PREFIX_RE = re.compile(r"^\s*(?:answer|value)\s*[:：]\s*", re.IGNORECASE)
NUMBER_RE = re.compile(r"\(?-?[$£€]?\s*\d(?:[\d,.\s]*\d)?%?\)?")
DOT_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:\.\d{3})+$")
COMMA_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")
VARIANT_ALIASES = {
    "clean": "clean",
    "target_value": "target_value",
    "target_value_replacement": "target_value",
    "target-value replacement": "target_value",
    "header_swap": "header_address_swap",
    "header_address_swap": "header_address_swap",
    "header/address swap": "header_address_swap",
    "column-header swap": "header_address_swap",
    "irrelevant_cell": "irrelevant_value",
    "irrelevant_value": "irrelevant_value",
    "irrelevant_value_replacement": "irrelevant_value",
    "irrelevant-cell replacement": "irrelevant_value",
    "irrelevant-value replacement": "irrelevant_value",
}


@dataclass(frozen=True)
class PacketCell:
    cell_id: str
    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class AccuracySummary:
    total: int
    exact_correct: int
    numeric_correct: int

    @property
    def exact_accuracy(self) -> float:
        return self.exact_correct / self.total if self.total else 0.0

    @property
    def numeric_accuracy(self) -> float:
        return self.numeric_correct / self.total if self.total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "exact_correct": self.exact_correct,
            "exact_accuracy": self.exact_accuracy,
            "numeric_correct": self.numeric_correct,
            "numeric_accuracy": self.numeric_accuracy,
        }


def normalize_text_answer(value: str) -> str:
    normalized = ANSWER_PREFIX_RE.sub("", value.strip())
    normalized = normalized.strip().strip("\"'`")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def first_number(value: str) -> str | None:
    match = NUMBER_RE.search(value)
    if match is None:
        return None
    return match.group(0)


def normalize_numeric_answer(value: str) -> str | None:
    candidate = first_number(value)
    if candidate is None:
        return None

    stripped = candidate.strip()
    negative = stripped.startswith("(") and stripped.endswith(")")
    stripped = stripped.strip("()")
    stripped = stripped.replace("$", "").replace("£", "").replace("€", "")
    stripped = stripped.replace("%", "").strip()
    stripped = re.sub(r"\s+", "", stripped)
    stripped = normalize_group_separators(stripped)

    try:
        number = Decimal(stripped)
    except InvalidOperation:
        return None

    if negative:
        number = -number
    return format(number.normalize(), "f")


def normalize_group_separators(value: str) -> str:
    """Canonicalize common financial thousands separators before Decimal parsing."""

    if "," not in value and "." not in value:
        return value

    sign = ""
    body = value
    if body.startswith("-"):
        sign, body = "-", body[1:]

    signed_body = f"{sign}{body}"
    if COMMA_THOUSANDS_RE.fullmatch(signed_body) or DOT_THOUSANDS_RE.fullmatch(signed_body):
        return signed_body.replace(",", "").replace(".", "")

    if "," in body and "." in body:
        last_comma = body.rfind(",")
        last_dot = body.rfind(".")
        if last_dot > last_comma:
            decimal_sep = "." if len(body) - last_dot - 1 != 3 else None
        else:
            decimal_sep = "," if len(body) - last_comma - 1 != 3 else None

        if decimal_sep is None:
            return sign + body.replace(",", "").replace(".", "")
        thousands_sep = "," if decimal_sep == "." else "."
        return sign + body.replace(thousands_sep, "").replace(decimal_sep, ".")

    if "," in body:
        return sign + body.replace(",", "")
    return sign + body


def is_exact_correct(prediction: str, target: str) -> bool:
    return normalize_text_answer(prediction).casefold() == normalize_text_answer(target).casefold()


def is_numeric_correct(prediction: str, target: str) -> bool:
    pred_number = normalize_numeric_answer(prediction)
    target_number = normalize_numeric_answer(target)
    return pred_number is not None and pred_number == target_number


def canonical_variant(value: str) -> str:
    key = value.strip().casefold().replace(" ", "_")
    return VARIANT_ALIASES.get(key, VARIANT_ALIASES.get(value.strip().casefold(), value))


def summarize_accuracy(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    all_records = list(records)
    for record in all_records:
        buckets[(str(record["setting"]), canonical_variant(str(record["variant"])))].append(record)

    def build_summary(items: list[Mapping[str, Any]]) -> AccuracySummary:
        return AccuracySummary(
            total=len(items),
            exact_correct=sum(1 for item in items if record_exact_correct(item)),
            numeric_correct=sum(1 for item in items if record_numeric_correct(item)),
        )

    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in all_records:
        by_setting[str(record["setting"])].append(record)

    by_setting_variant = {
        f"{setting}/{variant}": build_summary(items).to_json()
        for (setting, variant), items in sorted(buckets.items())
    }

    return {
        "overall": build_summary(all_records).to_json(),
        "by_setting": {
            setting: build_summary(items).to_json()
            for setting, items in sorted(by_setting.items())
        },
        "by_setting_variant": by_setting_variant,
        "counterfactual_test_by_setting": summarize_counterfactual_test(by_setting_variant),
    }


def geometric_mean(values: Sequence[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return math.prod(value for value in values if value is not None) ** (1.0 / len(values))


def metric_rate(
    by_setting_variant: Mapping[str, Mapping[str, Any]],
    setting: str,
    variant: str,
    correctness_key: str,
) -> float | None:
    summary = by_setting_variant.get(f"{setting}/{variant}")
    if summary is None or int(summary.get("total", 0)) == 0:
        return None
    return float(summary[correctness_key])


def metric_rate_any(
    by_setting_variant: Mapping[str, Mapping[str, Any]],
    setting: str,
    variants: Sequence[str],
    correctness_key: str,
) -> float | None:
    for variant in variants:
        rate = metric_rate(by_setting_variant, setting, variant, correctness_key)
        if rate is not None:
            return rate
    return None


def summarize_counterfactual_test(
    by_setting_variant: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    settings = sorted({key.split("/", 1)[0] for key in by_setting_variant})
    output: dict[str, Any] = {}

    for setting in settings:
        exact_tvfr = metric_rate(by_setting_variant, setting, "target_value", "exact_accuracy")
        exact_hfr = metric_rate_any(
            by_setting_variant,
            setting,
            ("header_swap", "header_address_swap"),
            "exact_accuracy",
        )
        exact_isr = metric_rate_any(
            by_setting_variant,
            setting,
            ("irrelevant_value", "irrelevant_cell"),
            "exact_accuracy",
        )
        numeric_tvfr = metric_rate(by_setting_variant, setting, "target_value", "numeric_accuracy")
        numeric_hfr = metric_rate_any(
            by_setting_variant,
            setting,
            ("header_swap", "header_address_swap"),
            "numeric_accuracy",
        )
        numeric_isr = metric_rate_any(
            by_setting_variant,
            setting,
            ("irrelevant_value", "irrelevant_cell"),
            "numeric_accuracy",
        )

        output[setting] = {
            "clean_accuracy_exact": metric_rate(by_setting_variant, setting, "clean", "exact_accuracy"),
            "clean_accuracy_numeric": metric_rate(by_setting_variant, setting, "clean", "numeric_accuracy"),
            "target_value_following_rate_exact": exact_tvfr,
            "target_value_following_rate_numeric": numeric_tvfr,
            "header_following_rate_exact": exact_hfr,
            "header_following_rate_numeric": numeric_hfr,
            "irrelevant_stability_rate_exact": exact_isr,
            "irrelevant_stability_rate_numeric": numeric_isr,
            "counterfactual_grounding_score_exact": geometric_mean((exact_tvfr, exact_hfr, exact_isr)),
            "counterfactual_grounding_score_numeric": geometric_mean((numeric_tvfr, numeric_hfr, numeric_isr)),
            "definition": {
                "clean_accuracy": "accuracy on original clean images",
                "target_value_following_rate": "accuracy on target_value counterfactuals",
                "header_following_rate": "accuracy on header_swap counterfactuals",
                "irrelevant_stability_rate": "accuracy on irrelevant_value or legacy irrelevant_cell counterfactuals",
                "counterfactual_grounding_score": "(TVFR * HFR * ISR) ** (1/3)",
            },
        }

    return output


def build_prediction_record(
    record: Mapping[str, Any],
    *,
    setting: str,
    prediction: str | None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build one predictions.jsonl row from a flat CF record and a generated answer.

    ``record`` is a flattened per-variant record (e.g. from
    ``flatten_group_records``), carrying a top-level ``variant`` and
    ``answer``. On failure (``error`` set, ``prediction`` None) the row is
    still returned -- marked incorrect and carrying ``error`` -- so failed
    samples are counted by ``summarize_accuracy`` rather than silently
    dropped from the totals.
    """

    answer = record.get("answer") or {}
    target = str(answer.get("raw", ""))
    exact_correct = prediction is not None and is_exact_correct(prediction, target)
    numeric_correct = prediction is not None and is_numeric_correct(prediction, target)
    return {
        "setting": setting,
        "variant": str(record.get("variant") or "clean"),
        "group_id": record["group_id"],
        "document_id": record.get("document_id"),
        "question": record["question"],
        "target_answer": target,
        "prediction": prediction,
        "exact_correct": exact_correct,
        "numeric_correct": numeric_correct,
        "scale": answer.get("scale"),
        "metric": answer.get("metric"),
        "period": answer.get("period"),
        "image_path": record.get("image_path"),
        "error": error,
    }


def record_exact_correct(record: Mapping[str, Any]) -> bool:
    prediction = record.get("prediction")
    target = record.get("target_answer")
    if prediction is not None and target is not None:
        return is_exact_correct(str(prediction), str(target))
    return bool(record.get("exact_correct", False))


def record_numeric_correct(record: Mapping[str, Any]) -> bool:
    prediction = record.get("prediction")
    target = record.get("target_answer")
    if prediction is not None and target is not None:
        return is_numeric_correct(str(prediction), str(target))
    return bool(record.get("numeric_correct", False))


def clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def padded_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return clamp_box((x0 - padding, y0 - padding, x1 + padding, y1 + padding), width, height)


def cell_from_json(raw: Mapping[str, Any]) -> PacketCell:
    bbox = tuple(int(v) for v in raw["bbox"])
    if len(bbox) != 4:
        raise ValueError(f"Invalid cell bbox: {raw!r}")
    return PacketCell(
        cell_id=str(raw["cell_id"]),
        text=str(raw.get("text") or ""),
        bbox=bbox,
    )


def patch_text_by_cell_id(patches: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for patch in patches:
        cell = patch.get("cell") or {}
        cell_id = cell.get("cell_id")
        if cell_id:
            output[str(cell_id)] = str(patch.get("new_text") or "")
    return output


def make_oracle_packet(
    image: Any,
    packet: Mapping[str, Any],
    *,
    patches: Iterable[Mapping[str, Any]] = (),
    apply_patch_text: bool = False,
    padding: int = 8,
    gap: int = 8,
    scale: int = 3,
) -> Any:
    """Render a compact evidence packet from crops of the provided page image."""

    from PIL import Image, ImageDraw, ImageFont

    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    replacements = patch_text_by_cell_id(patches)
    font = ImageFont.load_default()

    def cells_for(key: str) -> list[PacketCell]:
        return [cell_from_json(raw) for raw in packet.get(key, [])]

    value_cell = cell_from_json(packet["value_cell"])
    sections: list[tuple[str, list[PacketCell]]] = [
        ("Table context", cells_for("spanning_headers")),
        ("Column header", cells_for("column_headers")),
        ("Unit / scale", cells_for("unit_cells")),
        ("Row header", cells_for("row_headers")),
        ("Value cell", [value_cell]),
    ]

    def crop_cell(cell: PacketCell) -> Any:
        x0, y0, x1, y1 = padded_box(cell.bbox, image.width, image.height, padding)
        crop = image.crop((x0, y0, x1, y1))
        replacement = replacements.get(cell.cell_id)
        if apply_patch_text and replacement is not None and replacement != cell.text:
            crop = Image.new("RGB", crop.size, "white")
            draw = ImageDraw.Draw(crop)
            draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline=(180, 180, 180), width=1)
            draw.text((padding, padding), replacement, fill=(0, 0, 0), font=font)
        if scale == 1:
            return crop
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = 1
        crop = crop.resize((crop.width * scale, crop.height * scale), resample=resample)
        return crop

    rendered_sections: list[tuple[str, list[Any]]] = []
    for label, cells in sections:
        crops = [crop_cell(cell) for cell in cells if cell.text or cell.cell_id == value_cell.cell_id]
        if crops:
            rendered_sections.append((label, crops))

    label_width = 110
    row_heights = [max(max(crop.height for crop in crops), 24) for _, crops in rendered_sections]
    content_width = max(sum(crop.width for crop in crops) + gap * (len(crops) - 1) for _, crops in rendered_sections)
    output_width = label_width + content_width + gap * 2
    output_height = sum(row_heights) + gap * (len(row_heights) + 1)
    output = Image.new("RGB", (output_width, output_height), "white")
    draw = ImageDraw.Draw(output)

    y_cursor = gap
    for (label, crops), row_height in zip(rendered_sections, row_heights):
        draw.text((gap, y_cursor + 4), label, fill=(70, 70, 70), font=font)
        x_cursor = label_width + gap
        for crop in crops:
            y_offset = y_cursor + max(0, (row_height - crop.height) // 2)
            output.paste(crop, (x_cursor, y_offset))
            draw.rectangle(
                (x_cursor, y_offset, x_cursor + crop.width - 1, y_offset + crop.height - 1),
                outline=(210, 210, 210),
                width=1,
            )
            x_cursor += crop.width + gap
        y_cursor += row_height + gap

    return output


def make_oracle_crop(
    image: Any,
    evidence_bbox: tuple[int, int, int, int],
    *,
    mode: str = "header_context",
    padding: int = 8,
    gap: int = 8,
) -> Any:
    """Create an oracle evidence image from a rendered pilot page.

    ``cell`` returns only the padded target cell. ``header_context`` creates a
    compact three-panel image: target column above the cell, target row up to
    the cell, then the padded target cell. This gives the model the correct
    evidence and address context without showing the whole page.
    """

    from PIL import Image, ImageDraw

    width, height = image.size
    x0, y0, x1, y1 = evidence_bbox
    if mode == "cell":
        return image.crop(padded_box(evidence_bbox, width, height, padding))
    if mode != "header_context":
        raise ValueError(f"Unsupported oracle crop mode: {mode}")

    column_box = clamp_box((x0 - padding, 0, x1 + padding, y1 + padding), width, height)
    row_box = clamp_box((0, y0 - padding, x1 + padding, y1 + padding), width, height)
    cell_box = padded_box(evidence_bbox, width, height, padding)

    panels = [image.crop(column_box), image.crop(row_box), image.crop(cell_box)]
    out_width = max(panel.width for panel in panels)
    out_height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
    output = Image.new("RGB", (out_width, out_height), "white")
    draw = ImageDraw.Draw(output)

    y_cursor = 0
    for panel in panels:
        output.paste(panel, (0, y_cursor))
        draw.rectangle((0, y_cursor, panel.width - 1, y_cursor + panel.height - 1), outline=(220, 0, 0), width=2)
        y_cursor += panel.height + gap

    return output


def make_oracle_highlight(
    image: Any,
    evidence_bbox: tuple[int, int, int, int],
    *,
    padding: int = 2,
    outline: tuple[int, int, int] = (220, 0, 0),
    width: int = 5,
) -> Any:
    """Return the full page with the target evidence cell outlined."""

    from PIL import ImageDraw

    output = image.copy()
    page_width, page_height = output.size
    x0, y0, x1, y1 = padded_box(evidence_bbox, page_width, page_height, padding)
    draw = ImageDraw.Draw(output)
    draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=outline, width=width)
    return output


def make_packet_noheader_crop(
    image: Any,
    evidence_bbox: tuple[int, int, int, int],
    *,
    padding: int = 8,
    scale: int = 3,
) -> Any:
    """Return only an enlarged target-cell crop, without row or column headers."""

    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    from PIL import Image

    width, height = image.size
    crop = image.crop(padded_box(evidence_bbox, width, height, padding))
    if scale == 1:
        return crop

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = 1
    return crop.resize((crop.width * scale, crop.height * scale), resample=resample)


def make_full_page_hires(image: Any, *, scale: int = 3) -> Any:
    """Return the whole page upscaled, matching the magnification used by oracle/no-header crops.

    Unlike ``make_packet_noheader_crop`` and ``make_oracle_packet``, this does not
    crop any content out — it exists to isolate legibility (pixels-per-character)
    from the visual-search difficulty of the full page, as a resolution-matched
    control for the ``full_page`` setting.
    """

    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    from PIL import Image

    if scale == 1:
        return image.copy()

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = 1
    return image.resize((image.width * scale, image.height * scale), resample=resample)
