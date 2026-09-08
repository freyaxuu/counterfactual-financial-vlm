"""Grounding-only training target: format, parse, validate, and score.

Resolves a single unambiguous (row_header, column_header, target_bbox) triple
from a canonical_group_cf_v1 variant payload's evidence block, serializes it
into a fixed <answer>/<row>/<column>/<bbox> target string, and provides the
inverse (parsing a model's decoded prediction) plus the grounding-specific
scoring primitives (header-text normalization, bbox IoU, point-to-cell
resolution) needed by the Grounding-only LoRA training and eval scripts.

Deliberately pure (no torch/PIL/model imports) so it's fully unit-testable
against synthetic fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
import statistics
from typing import Any, Mapping, Sequence


BBOX_SCALE = 1000

UNSAFE_TEXT_CHARS = ("<", ">", "\n", "\r")

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
ROW_RE = re.compile(r"<row>(.*?)</row>", re.S)
COLUMN_RE = re.compile(r"<column>(.*?)</column>", re.S)
BBOX_RE = re.compile(r"<bbox>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*</bbox>", re.S)

GROUNDING_PROMPT_INSTRUCTION = (
    "Answer the question using the financial table.\n\n"
    "Return the answer and the evidence location in the following format:\n\n"
    "<answer>...</answer>\n"
    "<row>...</row>\n"
    "<column>...</column>\n"
    "<bbox>x1 y1 x2 y2</bbox>"
)


def grounding_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")
    return f"{GROUNDING_PROMPT_INSTRUCTION}\nQuestion: {question}"


def normalize_header_text(value: str | None) -> str:
    """Deterministic normalizer for row/column header comparison: casefold + collapse whitespace."""

    if value is None:
        return ""
    return " ".join(value.strip().split()).casefold()


def normalize_bbox_1000(bbox_normalised: Sequence[float]) -> tuple[int, int, int, int]:
    """Convert a [0,1]-normalized bbox (x1,y1,x2,y2) to clamped [0,1000] integers.

    ``bbox_normalised`` is already ``pixel / dimension`` (see
    ``financial_vlm.data.canonical_schema.normalise_bbox``), so scaling by
    BBOX_SCALE reproduces the ``round(1000 * x1 / W)`` formula directly
    without needing the original pixel coordinates or image size.
    """

    if len(bbox_normalised) != 4:
        raise ValueError(f"bbox_normalised must have 4 values, got {bbox_normalised!r}")
    scaled = [round(BBOX_SCALE * value) for value in bbox_normalised]
    return tuple(max(0, min(BBOX_SCALE, value)) for value in scaled)  # type: ignore[return-value]


def bbox01_to_pixels(bbox_normalised: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Recover the original pixel bbox from a [0,1]-normalized bbox, for debug logging only."""

    if len(bbox_normalised) != 4:
        raise ValueError(f"bbox_normalised must have 4 values, got {bbox_normalised!r}")
    x1, y1, x2, y2 = bbox_normalised
    return (round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))


def format_grounding_target(
    answer: str,
    row_header: str,
    column_header: str,
    bbox_1000: Sequence[int],
) -> str:
    """Deterministic serialization -- the sole legitimate target format."""

    x1, y1, x2, y2 = bbox_1000
    return (
        f"<answer>{answer}</answer>\n"
        f"<row>{row_header}</row>\n"
        f"<column>{column_header}</column>\n"
        f"<bbox>{x1} {y1} {x2} {y2}</bbox>"
    )


@dataclass(frozen=True)
class ParsedGroundingPrediction:
    raw_text: str
    answer: str | None
    row: str | None
    column: str | None
    bbox: tuple[int, int, int, int] | None
    bbox_valid: bool


def parse_grounding_prediction(text: str) -> ParsedGroundingPrediction:
    """Parse a decoded model prediction. Never raises -- missing/malformed fields become None."""

    answer_match = ANSWER_RE.search(text)
    row_match = ROW_RE.search(text)
    column_match = COLUMN_RE.search(text)
    bbox_match = BBOX_RE.search(text)

    bbox: tuple[int, int, int, int] | None = None
    bbox_valid = False
    if bbox_match is not None:
        x1, y1, x2, y2 = (int(group) for group in bbox_match.groups())
        in_range = all(0 <= value <= BBOX_SCALE for value in (x1, y1, x2, y2))
        if in_range and x2 > x1 and y2 > y1:
            bbox = (x1, y1, x2, y2)
            bbox_valid = True

    return ParsedGroundingPrediction(
        raw_text=text,
        answer=answer_match.group(1).strip() if answer_match else None,
        row=row_match.group(1).strip() if row_match else None,
        column=column_match.group(1).strip() if column_match else None,
        bbox=bbox,
        bbox_valid=bbox_valid,
    )


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def resolve_cell_at_point(
    cx: float,
    cy: float,
    evidence_units: Sequence[Mapping[str, Any]],
    *,
    target_id: str | None = None,
) -> str | None:
    """Map a (cx, cy) point in [0, BBOX_SCALE] space to an evidence-unit cell id.

    Only cells present in ``evidence_units`` (target, headers, unit cells) can
    be resolved -- the manifest doesn't carry the full table grid. Returns
    None if the point falls in none of them. When multiple bboxes overlap the
    point, ``target_id`` (if among the matches) always wins.
    """

    matches: list[str] = []
    for unit in evidence_units:
        x1, y1, x2, y2 = normalize_bbox_1000(unit["bbox_normalised"])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            matches.append(str(unit["id"]))

    if target_id is not None and target_id in matches:
        return target_id
    return matches[0] if matches else None


def validate_grounding_payload(payload: Mapping[str, Any]) -> str | None:
    """Return None if ``payload`` (one variant's dict) supports grounding supervision, else a reason code."""

    evidence = payload.get("evidence")
    if not evidence:
        return "missing_evidence"

    target_id = evidence.get("target_id")
    if not target_id:
        return "missing_target_cell"

    units_by_id = {str(unit["id"]): unit for unit in payload.get("evidence_units") or []}
    if target_id not in units_by_id:
        return "missing_target_cell"

    row_ids = evidence.get("row_header_ids") or []
    if len(row_ids) == 0:
        return "missing_row_header"
    if len(row_ids) > 1:
        return "ambiguous_row_header"
    if row_ids[0] not in units_by_id:
        return "missing_row_header"

    column_ids = evidence.get("column_header_ids") or []
    if len(column_ids) == 0:
        return "missing_column_header"
    if len(column_ids) > 1:
        return "ambiguous_column_header"
    if column_ids[0] not in units_by_id:
        return "missing_column_header"

    bbox = evidence.get("bbox_normalised")
    if not bbox or len(bbox) != 4:
        return "invalid_bbox"
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1 and x2 > x1 and y2 > y1):
        return "invalid_bbox"

    answer_raw = (payload.get("answer") or {}).get("raw")
    if not answer_raw or not str(answer_raw).strip():
        return "missing_answer"

    row_text = units_by_id[row_ids[0]]["text"]
    column_text = units_by_id[column_ids[0]]["text"]
    for text in (str(answer_raw), str(row_text), str(column_text)):
        if any(char in text for char in UNSAFE_TEXT_CHARS):
            return "unsafe_text_for_serialization"

    return None


@dataclass(frozen=True)
class GroundingExample:
    group_id: str
    variant: str
    image_path: str
    question: str
    answer_raw: str
    row_header: str
    column_header: str
    target_cell_id: str
    bbox_normalised_01: tuple[float, float, float, float]
    bbox_1000: tuple[int, int, int, int]
    target_text: str


def build_grounding_example(
    group_id: str,
    variant: str,
    question: str,
    payload: Mapping[str, Any],
) -> GroundingExample:
    """Build a GroundingExample from one variant payload. Raises ValueError if invalid.

    Callers doing a bulk pre-flight pass should call ``validate_grounding_payload``
    first to collect failure-reason counts instead of catching this per-record.
    """

    reason = validate_grounding_payload(payload)
    if reason is not None:
        raise ValueError(f"group_id={group_id!r} variant={variant!r}: {reason}")

    evidence = payload["evidence"]
    units_by_id = {str(unit["id"]): unit for unit in payload["evidence_units"]}
    row_header = str(units_by_id[evidence["row_header_ids"][0]]["text"])
    column_header = str(units_by_id[evidence["column_header_ids"][0]]["text"])
    answer_raw = str(payload["answer"]["raw"])
    bbox01 = tuple(float(value) for value in evidence["bbox_normalised"])
    bbox_1000 = normalize_bbox_1000(bbox01)

    target_text = format_grounding_target(answer_raw, row_header, column_header, bbox_1000)

    return GroundingExample(
        group_id=group_id,
        variant=variant,
        image_path=str(payload["image_path"]),
        question=question,
        answer_raw=answer_raw,
        row_header=row_header,
        column_header=column_header,
        target_cell_id=str(evidence["target_id"]),
        bbox_normalised_01=bbox01,  # type: ignore[assignment]
        bbox_1000=bbox_1000,
        target_text=target_text,
    )


def summarize_grounding_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate grounding-specific metrics, grouped by ``row["variant"]``.

    Each row must carry: ``variant``, ``answer_correct``, ``row_correct``,
    ``column_correct``, ``cell_correct``, ``bbox_iou`` (all bool/float).
    Reports row/column-header accuracy, CellAcc@1, mean/median bbox IoU,
    P(answer_correct AND cell_correct), and P(cell_correct | answer_correct)
    -- the diagnostic that answers alone can't establish correct evidence use.
    """

    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)

    summary: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        total = len(items)
        row_correct = sum(1 for item in items if item["row_correct"])
        column_correct = sum(1 for item in items if item["column_correct"])
        cell_correct = sum(1 for item in items if item["cell_correct"])
        answer_correct = sum(1 for item in items if item["answer_correct"])
        both_correct = sum(1 for item in items if item["answer_correct"] and item["cell_correct"])
        ious = [float(item["bbox_iou"]) for item in items]

        summary[variant] = {
            "total": total,
            "row_header_accuracy": row_correct / total if total else None,
            "column_header_accuracy": column_correct / total if total else None,
            "cell_acc_at_1": cell_correct / total if total else None,
            "bbox_iou_mean": statistics.mean(ious) if ious else None,
            "bbox_iou_median": statistics.median(ious) if ious else None,
            "answer_correct_count": answer_correct,
            "p_answer_and_cell_correct": both_correct / total if total else None,
            "p_cell_correct_given_answer_correct": (both_correct / answer_correct) if answer_correct else None,
        }
    return summary


def source_index_from_group_id(group_id: str) -> str:
    """Extract the source-table index from a group_id (syn_{source_index:06d}_{slug})."""

    parts = group_id.split("_")
    if len(parts) < 3 or parts[0] != "syn":
        raise ValueError(f"Unrecognized group_id format: {group_id!r}")
    return parts[1]


def check_split_leakage(
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Raise ValueError if any source table appears in more than one split.

    Returns per-split source-table counts on success, for logging.
    """

    train_keys = {source_index_from_group_id(str(r["group_id"])) for r in train_records}
    dev_keys = {source_index_from_group_id(str(r["group_id"])) for r in dev_records}
    test_keys = {source_index_from_group_id(str(r["group_id"])) for r in test_records}

    overlaps = {
        "train_dev": sorted(train_keys & dev_keys),
        "train_test": sorted(train_keys & test_keys),
        "dev_test": sorted(dev_keys & test_keys),
    }
    leaks = {name: ids for name, ids in overlaps.items() if ids}
    if leaks:
        raise ValueError(f"Source-table leakage detected across splits: {leaks}")

    return {"train_source_tables": len(train_keys), "dev_source_tables": len(dev_keys), "test_source_tables": len(test_keys)}


def validate_and_build_examples(
    group_records: Sequence[Mapping[str, Any]],
    *,
    variant: str = "clean",
) -> tuple[list[GroundingExample], dict[str, int]]:
    """Validate every group's ``variant`` payload before building examples.

    Never silently skips: every excluded group is counted under a specific
    ``invalid__<reason>`` key in the returned stats. A duplicate group_id
    with contradictory annotations (different question or answer) raises
    immediately rather than being counted -- that's a fail-fast data-integrity
    error, not an ordinary exclusion.
    """

    stats: dict[str, int] = {"total_groups": 0, "valid_groups": 0}
    examples: list[GroundingExample] = []
    seen: dict[str, Mapping[str, Any]] = {}

    for record in group_records:
        group_id = str(record["group_id"])
        stats["total_groups"] += 1

        if group_id in seen:
            prior_payload = seen[group_id]["variants"][variant]
            current_payload = record["variants"][variant]
            if prior_payload.get("answer") != current_payload.get("answer") or seen[group_id].get(
                "question"
            ) != record.get("question"):
                raise ValueError(f"group_id {group_id!r} appears twice with contradictory annotations")
            stats["duplicate_group_id_skipped"] = stats.get("duplicate_group_id_skipped", 0) + 1
            continue
        seen[group_id] = record

        payload = record["variants"][variant]
        reason = validate_grounding_payload(payload)
        if reason is not None:
            key = f"invalid__{reason}"
            stats[key] = stats.get(key, 0) + 1
            continue

        example = build_grounding_example(group_id, variant, str(record["question"]), payload)
        examples.append(example)
        stats["valid_groups"] += 1

    return examples, stats


# --- Evidence-first Grounding LoRA: forces bbox generation before the answer ---
#
# Ablation of Grounding-only (which put <answer> first, so the answer token
# could never be causally conditioned on the bbox). Target is bbox-then-answer
# only -- no row/column tags -- so row/column can't be "answered" by copying
# text straight out of the question without any visual grounding. Reuses
# validate_and_build_examples / build_grounding_example / check_split_leakage
# unchanged: same 995 valid groups, same resolved answer/bbox/target_cell_id,
# only the *serialized target string* differs.

EVIDENCE_FIRST_PROMPT_INSTRUCTION = (
    "Answer the question using the financial table.\n\n"
    "First identify the bounding box of the table cell containing the evidence needed to answer "
    "the question, then give the answer.\n\n"
    "Return exactly:\n\n"
    "<bbox>x1 y1 x2 y2</bbox>\n"
    "<answer>...</answer>"
)


def evidence_first_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")
    return f"{EVIDENCE_FIRST_PROMPT_INSTRUCTION}\nQuestion: {question}"


def format_evidence_first_target(answer: str, bbox_1000: Sequence[int]) -> str:
    """Deterministic serialization -- <bbox> MUST precede <answer>, no alternative order."""

    x1, y1, x2, y2 = bbox_1000
    return f"<bbox>{x1} {y1} {x2} {y2}</bbox>\n<answer>{answer}</answer>"


@dataclass(frozen=True)
class ParsedEvidenceFirstPrediction:
    raw_text: str
    bbox: tuple[int, int, int, int] | None
    bbox_valid: bool
    answer: str | None
    bbox_before_answer: bool | None  # None if either tag is missing from the decoded text


def parse_evidence_first_prediction(text: str) -> ParsedEvidenceFirstPrediction:
    """Parse a decoded evidence-first prediction. Never raises -- malformed fields become None."""

    bbox_match = BBOX_RE.search(text)
    answer_match = ANSWER_RE.search(text)

    bbox: tuple[int, int, int, int] | None = None
    bbox_valid = False
    if bbox_match is not None:
        x1, y1, x2, y2 = (int(group) for group in bbox_match.groups())
        in_range = all(0 <= value <= BBOX_SCALE for value in (x1, y1, x2, y2))
        if in_range and x2 > x1 and y2 > y1:
            bbox = (x1, y1, x2, y2)
            bbox_valid = True

    bbox_before_answer = None
    if bbox_match is not None and answer_match is not None:
        bbox_before_answer = bbox_match.start() < answer_match.start()

    return ParsedEvidenceFirstPrediction(
        raw_text=text,
        bbox=bbox,
        bbox_valid=bbox_valid,
        answer=answer_match.group(1).strip() if answer_match else None,
        bbox_before_answer=bbox_before_answer,
    )


def summarize_evidence_first_grounding_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate CellAcc@1, bbox IoU, and the answer x cell 2x2 confusion analysis per variant.

    Each row must carry: ``variant``, ``answer_correct``, ``cell_correct``,
    ``bbox_iou`` (bool/bool/float). No row/column fields -- this target has none.
    """

    by_variant: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)

    summary: dict[str, Any] = {}
    for variant, items in sorted(by_variant.items()):
        total = len(items)
        cell_correct = sum(1 for item in items if item["cell_correct"])
        answer_correct = sum(1 for item in items if item["answer_correct"])
        both = sum(1 for item in items if item["answer_correct"] and item["cell_correct"])
        cell_correct_answer_wrong = sum(1 for item in items if item["cell_correct"] and not item["answer_correct"])
        cell_wrong_answer_correct = sum(1 for item in items if not item["cell_correct"] and item["answer_correct"])
        cell_wrong_answer_wrong = sum(1 for item in items if not item["cell_correct"] and not item["answer_correct"])
        cell_wrong = total - cell_correct
        answer_wrong = total - answer_correct
        ious = [float(item["bbox_iou"]) for item in items]

        summary[variant] = {
            "total": total,
            "answer_accuracy": answer_correct / total if total else None,
            "cell_acc_at_1": cell_correct / total if total else None,
            "bbox_iou_mean": statistics.mean(ious) if ious else None,
            "bbox_iou_median": statistics.median(ious) if ious else None,
            "confusion_counts": {
                "cell_correct_answer_correct": both,
                "cell_correct_answer_wrong": cell_correct_answer_wrong,
                "cell_wrong_answer_correct": cell_wrong_answer_correct,
                "cell_wrong_answer_wrong": cell_wrong_answer_wrong,
            },
            "confusion_pct": {
                "cell_correct_answer_correct": both / total if total else None,
                "cell_correct_answer_wrong": cell_correct_answer_wrong / total if total else None,
                "cell_wrong_answer_correct": cell_wrong_answer_correct / total if total else None,
                "cell_wrong_answer_wrong": cell_wrong_answer_wrong / total if total else None,
            },
            "p_cell_correct_given_answer_correct": (both / answer_correct) if answer_correct else None,
            "p_answer_correct_given_cell_correct": (both / cell_correct) if cell_correct else None,
            "p_answer_correct_given_cell_wrong": (cell_wrong_answer_correct / cell_wrong) if cell_wrong else None,
            "p_cell_wrong_given_answer_wrong": (cell_wrong_answer_wrong / answer_wrong) if answer_wrong else None,
        }
    return summary


HFR_ERROR_CATEGORIES = (
    "A_correct_cell_correct_answer",
    "B_correct_cell_wrong_answer",
    "C_wrong_cell_correct_answer",
    "D_wrong_cell_wrong_answer",
)


def categorize_hfr_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str = "header_address_swap",
) -> dict[str, Any]:
    """Bucket every ``variant`` row into the A/B/C/D cell x answer categories, keeping group_ids."""

    group_ids: dict[str, list[str]] = {category: [] for category in HFR_ERROR_CATEGORIES}
    for row in rows:
        if str(row["variant"]) != variant:
            continue
        cell_ok = bool(row["cell_correct"])
        answer_ok = bool(row["answer_correct"])
        group_id = str(row["group_id"])
        if cell_ok and answer_ok:
            group_ids["A_correct_cell_correct_answer"].append(group_id)
        elif cell_ok and not answer_ok:
            group_ids["B_correct_cell_wrong_answer"].append(group_id)
        elif not cell_ok and answer_ok:
            group_ids["C_wrong_cell_correct_answer"].append(group_id)
        else:
            group_ids["D_wrong_cell_wrong_answer"].append(group_id)

    return {
        "variant": variant,
        "counts": {category: len(ids) for category, ids in group_ids.items()},
        "group_ids": group_ids,
    }
