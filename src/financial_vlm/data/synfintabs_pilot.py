"""Prepare SynFinTabs lookup counterfactual samples.

The pilot can render counterfactuals either by deterministic pixel-level cell
patches or by re-rendering the structured source table cells. The manifest
records the selected method without changing the evaluation schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


NUMERIC_RE = re.compile(r"^\(?-?[$£€]?\s*\d[\d,]*(?:\.\d+)?%?\)?$")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class WordRef:
    index: int
    text: str
    bbox: tuple[int, int, int, int]
    row_idx: int
    col_idx: int


@dataclass(frozen=True)
class CellRef:
    row_idx: int
    col_idx: int
    text: str
    bbox: tuple[int, int, int, int]
    label: str
    word_indices: tuple[int, ...]

    @property
    def cell_id(self) -> str:
        return f"r{self.row_idx:02d}c{self.col_idx:02d}"


@dataclass(frozen=True)
class LocatedQuestion:
    question_id: str
    question: str
    answer: str
    answer_span: tuple[int, int]
    answer_cell: CellRef


@dataclass(frozen=True)
class TextPatch:
    cell: CellRef
    old_text: str
    new_text: str


@dataclass(frozen=True)
class CellMapping:
    source_cell: CellRef
    rendered_cell: CellRef


@dataclass(frozen=True)
class VariantPlan:
    variant: str
    intervention_type: str
    answer: str
    evidence_cell: CellRef
    patches: tuple[TextPatch, ...]
    validation_status: str
    validation_notes: tuple[str, ...]
    cell_mappings: tuple[CellMapping, ...] = ()


@dataclass(frozen=True)
class EvidencePacket:
    value_cell: CellRef
    row_headers: tuple[CellRef, ...]
    column_headers: tuple[CellRef, ...]
    unit_cells: tuple[CellRef, ...]
    spanning_headers: tuple[CellRef, ...]

    @property
    def cells(self) -> tuple[CellRef, ...]:
        ordered: list[CellRef] = []
        seen: set[str] = set()
        for cell in (
            *self.spanning_headers,
            *self.column_headers,
            *self.unit_cells,
            *self.row_headers,
            self.value_cell,
        ):
            if cell.cell_id in seen:
                continue
            ordered.append(cell)
            seen.add(cell.cell_id)
        return tuple(ordered)


def bbox_tuple(raw_bbox: Sequence[int] | None) -> tuple[int, int, int, int]:
    if raw_bbox is None or len(raw_bbox) != 4:
        raise ValueError(f"Expected bbox with four coordinates, got {raw_bbox!r}")
    x0, y0, x1, y1 = (int(v) for v in raw_bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox {raw_bbox!r}")
    return x0, y0, x1, y1


def flatten_table(rows: Sequence[Mapping[str, Any]]) -> tuple[list[CellRef], list[WordRef]]:
    cells: list[CellRef] = []
    words: list[WordRef] = []

    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row.get("cells") or []):
            word_indices: list[int] = []
            for word in cell.get("words") or []:
                word_index = len(words)
                word_indices.append(word_index)
                words.append(
                    WordRef(
                        index=word_index,
                        text=str(word.get("text") or ""),
                        bbox=bbox_tuple(word.get("bbox")),
                        row_idx=row_idx,
                        col_idx=col_idx,
                    )
                )

            cells.append(
                CellRef(
                    row_idx=row_idx,
                    col_idx=col_idx,
                    text=str(cell.get("text") or "").strip(),
                    bbox=bbox_tuple(cell.get("bbox")),
                    label=str(cell.get("label") or ""),
                    word_indices=tuple(word_indices),
                )
            )

    return cells, words


def is_numeric_text(text: str) -> bool:
    compact = " ".join(text.strip().split())
    return bool(NUMERIC_RE.match(compact))


def is_unit_or_scale_text(text: str) -> bool:
    compact = " ".join(text.strip().lower().split())
    if not compact:
        return False
    unit_tokens = (
        "$",
        "£",
        "€",
        "%",
        "usd",
        "eur",
        "gbp",
        "million",
        "millions",
        "mn",
        "m",
        "000",
        "thousand",
        "thousands",
    )
    return any(token in compact for token in unit_tokens)


def locate_answer_cell(
    question: Mapping[str, Any],
    cells: Sequence[CellRef],
    words: Sequence[WordRef],
) -> LocatedQuestion | None:
    span = question.get("answer_span") or {}
    start = span.get("start")
    end = span.get("end")
    if start is None or end is None:
        return None

    start_idx = int(start)
    end_idx = int(end)
    if start_idx < 0 or end_idx <= start_idx or end_idx > len(words):
        return None

    span_words = words[start_idx:end_idx]
    coords = {(word.row_idx, word.col_idx) for word in span_words}
    if len(coords) != 1:
        return None

    row_idx, col_idx = next(iter(coords))
    answer_cell = next(
        (cell for cell in cells if cell.row_idx == row_idx and cell.col_idx == col_idx),
        None,
    )
    if answer_cell is None:
        return None

    return LocatedQuestion(
        question_id=str(question.get("id") or question.get("question_id") or ""),
        question=str(question.get("question") or ""),
        answer=str(question.get("answer") or "").strip(),
        answer_span=(start_idx, end_idx),
        answer_cell=answer_cell,
    )


def make_counterfactual_number(
    value: str,
    rng: random.Random,
    *,
    min_relative_delta: float = 0.05,
    max_relative_delta: float = 0.35,
) -> str:
    """Perturb a numeric cell's text by a magnitude-relative delta.

    The delta scales with the value's own magnitude (rather than a fixed
    small constant) so the replacement stays visually distinguishable
    regardless of scale, and its sign is randomised so replacements aren't
    systematically larger than the original across the dataset. Raises
    ``ValueError`` instead of emitting a garbled sentinel when ``value``
    doesn't parse as a clean number.
    """

    stripped = value.strip()
    prefix = ""
    suffix = ""
    core = stripped

    if core.startswith("(") and core.endswith(")"):
        prefix, suffix = "(", ")"
        core = core[1:-1]

    currency = ""
    if core[:1] in {"$", "£", "€"}:
        currency, core = core[:1], core[1:]

    percent = ""
    if core.endswith("%"):
        core, percent = core[:-1], "%"

    numeric = core.replace(",", "").strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", numeric):
        raise ValueError(f"cannot build counterfactual number from {value!r}")

    magnitude = abs(float(numeric))
    if magnitude == 0:
        delta = float(rng.choice([1, 3, 5, 7]))
    else:
        delta = magnitude * rng.uniform(min_relative_delta, max_relative_delta)
    sign = 1 if magnitude <= delta else rng.choice((1, -1))

    if "." in numeric:
        decimals = len(numeric.rsplit(".", 1)[1])
        delta = round(delta, decimals) or (10**-decimals)
        new_number = float(numeric) + sign * delta
        rendered = f"{new_number:.{decimals}f}"
    else:
        delta_int = max(1, round(delta))
        new_number = int(numeric) + sign * delta_int
        rendered = str(new_number)

    if "," in core:
        integer, dot, frac = rendered.partition(".")
        rendered = f"{int(integer):,}{dot}{frac}"

    return f"{prefix}{currency}{rendered}{percent}{suffix}"


def _nearest_spanning_header_text(
    cells: Sequence[CellRef],
    col_idx: int,
    header_row_idx: int,
) -> str | None:
    candidates = [
        cell
        for cell in cells
        if cell.col_idx == col_idx
        and cell.row_idx < header_row_idx
        and cell.text
        and not is_numeric_text(cell.text)
        and cell.label == "column_header"
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda cell: cell.row_idx)[-1].text


def find_header_swap(
    cells: Sequence[CellRef],
    answer_cell: CellRef,
) -> tuple[CellRef, CellRef, CellRef] | None:
    def is_column_address_header(cell: CellRef) -> bool:
        if not cell.text or is_unit_or_scale_text(cell.text):
            return False
        return cell.label == "column_header" or bool(YEAR_RE.search(cell.text))

    rows_by_col = {
        cell.row_idx: cell
        for cell in cells
        if cell.col_idx == answer_cell.col_idx
        and cell.row_idx < answer_cell.row_idx
        and is_column_address_header(cell)
    }
    header_candidates = sorted(rows_by_col.values(), key=lambda cell: cell.row_idx, reverse=True)

    for target_header in header_candidates:
        peer_headers = [
            cell
            for cell in cells
            if cell.row_idx == target_header.row_idx
            and cell.col_idx != answer_cell.col_idx
            and is_column_address_header(cell)
        ]

        if YEAR_RE.search(target_header.text):
            peer_headers.sort(key=lambda cell: (0 if YEAR_RE.search(cell.text) else 1, cell.col_idx))
        else:
            peer_headers.sort(key=lambda cell: cell.col_idx)

        target_parent = _nearest_spanning_header_text(cells, target_header.col_idx, target_header.row_idx)
        for peer_header in peer_headers:
            peer_parent = _nearest_spanning_header_text(cells, peer_header.col_idx, peer_header.row_idx)
            if peer_parent != target_parent:
                continue
            peer_answer_cell = next(
                (
                    cell
                    for cell in cells
                    if cell.row_idx == answer_cell.row_idx
                    and cell.col_idx == peer_header.col_idx
                    and is_numeric_text(cell.text)
                    and cell.text != answer_cell.text
                ),
                None,
            )
            if peer_answer_cell is not None:
                return target_header, peer_header, peer_answer_cell

    return None


def find_irrelevant_cell(
    cells: Sequence[CellRef],
    answer_cell: CellRef,
    forbidden: Iterable[tuple[int, int]],
    rng: random.Random,
) -> CellRef | None:
    forbidden_set = set(forbidden)
    candidates = [
        cell
        for cell in cells
        if (cell.row_idx, cell.col_idx) not in forbidden_set
        and cell.row_idx != answer_cell.row_idx
        and is_numeric_text(cell.text)
        and cell.text != answer_cell.text
    ]
    if not candidates:
        candidates = [
            cell
            for cell in cells
            if (cell.row_idx, cell.col_idx) not in forbidden_set
            and is_numeric_text(cell.text)
            and cell.text != answer_cell.text
        ]
    if not candidates:
        return None
    if len(candidates) < 2:
        return rng.choice(candidates)

    def distance(cell: CellRef) -> int:
        return abs(cell.row_idx - answer_cell.row_idx) + abs(cell.col_idx - answer_cell.col_idx)

    ordered = sorted(candidates, key=distance)
    midpoint = len(ordered) // 2
    near_half, far_half = ordered[:midpoint] or ordered[:1], ordered[midpoint:]
    pool = near_half if rng.random() < 0.5 else far_half
    return rng.choice(pool)


def identity_cell_mappings(cells: Sequence[CellRef]) -> tuple[CellMapping, ...]:
    return tuple(CellMapping(cell, cell) for cell in cells)


def mapped_cell(source_cell: CellRef, rendered_cell: CellRef, *, text: str | None = None) -> CellRef:
    return CellRef(
        row_idx=rendered_cell.row_idx,
        col_idx=rendered_cell.col_idx,
        text=source_cell.text if text is None else text,
        bbox=rendered_cell.bbox,
        label=source_cell.label,
        word_indices=source_cell.word_indices,
    )


def plan_patch_text_by_source_cell(plan: VariantPlan) -> dict[str, str]:
    return {patch.cell.cell_id: patch.new_text for patch in plan.patches}


def rendered_text_by_cell_id(plan: VariantPlan) -> dict[str, str]:
    patched_text = plan_patch_text_by_source_cell(plan)
    rendered_text: dict[str, str] = {}
    for mapping in plan.cell_mappings:
        rendered_text[mapping.rendered_cell.cell_id] = patched_text.get(
            mapping.source_cell.cell_id,
            mapping.source_cell.text,
        )
    return rendered_text


def rendered_cells_for_plan(cells: Sequence[CellRef], plan: VariantPlan) -> tuple[CellRef, ...]:
    rendered_text = rendered_text_by_cell_id(plan)
    return tuple(
        CellRef(
            row_idx=cell.row_idx,
            col_idx=cell.col_idx,
            text=rendered_text.get(cell.cell_id, cell.text),
            bbox=cell.bbox,
            label=cell.label,
            word_indices=cell.word_indices,
        )
        for cell in cells
    )


def non_identity_mappings(plan: VariantPlan) -> tuple[CellMapping, ...]:
    return tuple(
        mapping
        for mapping in plan.cell_mappings
        if mapping.source_cell.cell_id != mapping.rendered_cell.cell_id
    )


def build_evidence_packet(cells: Sequence[CellRef], value_cell: CellRef) -> EvidencePacket:
    row_headers = tuple(
        cell
        for cell in cells
        if cell.row_idx == value_cell.row_idx
        and cell.col_idx < value_cell.col_idx
        and cell.text
        and not is_numeric_text(cell.text)
    )
    column_context = [
        cell
        for cell in cells
        if cell.col_idx == value_cell.col_idx and cell.row_idx < value_cell.row_idx and cell.text
    ]
    column_headers = tuple(
        cell
        for cell in column_context
        if not is_unit_or_scale_text(cell.text)
        and (cell.label == "column_header" or YEAR_RE.search(cell.text))
    )
    unit_cells = tuple(cell for cell in column_context if is_unit_or_scale_text(cell.text))
    spanning_headers = tuple(
        cell
        for cell in cells
        if cell.row_idx < value_cell.row_idx
        and cell.col_idx < value_cell.col_idx
        and cell.text
        and not is_numeric_text(cell.text)
        and cell.label == "column_header"
    )[-3:]

    return EvidencePacket(
        value_cell=value_cell,
        row_headers=row_headers[-3:],
        column_headers=column_headers[-4:],
        unit_cells=unit_cells[-2:],
        spanning_headers=spanning_headers,
    )


def build_variant_plans(
    located: LocatedQuestion,
    cells: Sequence[CellRef],
    rng: random.Random,
) -> tuple[VariantPlan, ...] | None:
    answer_cell = located.answer_cell
    if not is_numeric_text(located.answer) or not is_numeric_text(answer_cell.text):
        return None

    header_swap = find_header_swap(cells, answer_cell)
    if header_swap is None:
        return None

    target_header, peer_header, peer_answer_cell = header_swap
    irrelevant_cell = find_irrelevant_cell(
        cells,
        answer_cell,
        forbidden={
            (answer_cell.row_idx, answer_cell.col_idx),
            (target_header.row_idx, target_header.col_idx),
            (peer_header.row_idx, peer_header.col_idx),
            (peer_answer_cell.row_idx, peer_answer_cell.col_idx),
        },
        rng=rng,
    )
    if irrelevant_cell is None:
        return None

    identity_mappings = identity_cell_mappings(cells)
    try:
        target_value = make_counterfactual_number(located.answer, rng)
        irrelevant_value = make_counterfactual_number(irrelevant_cell.text, rng)
    except ValueError:
        return None

    return (
        VariantPlan(
            variant="clean",
            intervention_type="none",
            answer=located.answer,
            evidence_cell=answer_cell,
            patches=(),
            validation_status="passed",
            validation_notes=("original sample",),
            cell_mappings=identity_mappings,
        ),
        VariantPlan(
            variant="target_value_replacement",
            intervention_type="target-value replacement",
            answer=target_value,
            evidence_cell=mapped_cell(answer_cell, answer_cell, text=target_value),
            patches=(TextPatch(answer_cell, answer_cell.text, target_value),),
            validation_status="passed",
            validation_notes=("answer cell text was replaced; target evidence unchanged",),
            cell_mappings=identity_mappings,
        ),
        VariantPlan(
            variant="header_address_swap",
            intervention_type="header/address swap",
            answer=peer_answer_cell.text,
            evidence_cell=peer_answer_cell,
            patches=(
                TextPatch(target_header, target_header.text, peer_header.text),
                TextPatch(peer_header, peer_header.text, target_header.text),
            ),
            validation_status="passed",
            validation_notes=("question address now resolves to peer column after header swap",),
            cell_mappings=identity_mappings,
        ),
        VariantPlan(
            variant="irrelevant_value_replacement",
            intervention_type="irrelevant-value replacement",
            answer=located.answer,
            evidence_cell=answer_cell,
            patches=(TextPatch(irrelevant_cell, irrelevant_cell.text, irrelevant_value),),
            validation_status="passed",
            validation_notes=("non-target numeric cell was replaced; answer should remain stable",),
            cell_mappings=identity_mappings,
        ),
    )


def image_to_pil(image_obj: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to render pilot images") from exc

    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    if isinstance(image_obj, Mapping) and image_obj.get("bytes") is not None:
        from io import BytesIO

        return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image object {type(image_obj)!r}")


def render_variant_image(image: Any, table_bbox: Sequence[int], plan: VariantPlan) -> Any:
    from PIL import ImageDraw, ImageFont, ImageStat

    if non_identity_mappings(plan):
        raise ValueError("pixel_cell_patch rendering cannot represent non-identity bbox mappings")

    cropped = image_to_pil(image).crop(bbox_tuple(table_bbox))
    rendered = cropped.copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()

    for patch in plan.patches:
        x0, y0, x1, y1 = patch.cell.bbox
        interior = rendered.crop((max(x0 + 2, x0), max(y0 + 2, y0), max(x1 - 2, x0 + 1), max(y1 - 2, y0 + 1)))
        bg = tuple(int(v) for v in ImageStat.Stat(interior).median[:3])
        draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), fill=bg)
        draw.text((x0 + 6, y0 + 5), patch.new_text, fill=(0, 0, 0), font=font)

    return rendered


def render_source_table_image(
    cells: Sequence[CellRef],
    table_bbox: Sequence[int],
    plan: VariantPlan,
    *,
    cell_padding: int = 4,
) -> Any:
    """Render the table from structured cell text and bboxes.

    This keeps the same table-relative cell coordinate system used by the
    manifest while avoiding pixel overlays on the original image.
    """

    from PIL import Image, ImageDraw, ImageFont

    table_x0, table_y0, table_x1, table_y1 = bbox_tuple(table_bbox)
    width = table_x1 - table_x0
    height = table_y1 - table_y0
    if cells:
        width = max(width, max(cell.bbox[2] for cell in cells))
        height = max(height, max(cell.bbox[3] for cell in cells))
    width = max(1, width)
    height = max(1, height)

    rendered = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    replacements = rendered_text_by_cell_id(plan) if plan.cell_mappings else plan_patch_text_by_source_cell(plan)

    def text_size(text: str) -> tuple[int, int]:
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]

    line_height = max(text_size("Ag")[1] + 2, 10)

    def wrap_text(text: str, max_width: int, max_lines: int) -> list[str]:
        words = text.split()
        if not words or max_lines <= 0:
            return []
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_size(candidate)[0] <= max_width:
                current = candidate
                continue
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                return lines
        lines.append(current)
        return lines[:max_lines]

    def draw_cell_text(cell: CellRef, text: str, fill: tuple[int, int, int] | str) -> None:
        x0, y0, x1, y1 = cell.bbox
        inner_x0 = min(max(0, x0 + cell_padding), width)
        inner_y0 = min(max(0, y0 + cell_padding), height)
        inner_x1 = min(max(inner_x0, x1 - cell_padding), width)
        inner_y1 = min(max(inner_y0, y1 - cell_padding), height)
        inner_width = max(1, inner_x1 - inner_x0)
        inner_height = max(1, inner_y1 - inner_y0)
        max_lines = max(1, inner_height // line_height)

        clipped = Image.new("RGB", (inner_width, inner_height), fill)
        clipped_draw = ImageDraw.Draw(clipped)
        for line_index, line in enumerate(wrap_text(" ".join(text.split()), inner_width, max_lines)):
            clipped_draw.text((0, line_index * line_height), line, fill=(0, 0, 0), font=font)
        rendered.paste(clipped, (inner_x0, inner_y0))

    for cell in sorted(cells, key=lambda item: (item.row_idx, item.col_idx)):
        x0, y0, x1, y1 = cell.bbox
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue

        if cell.label == "column_header":
            fill = (236, 240, 245)
        elif cell.label in {"row_header", "stub_header"}:
            fill = (248, 249, 250)
        else:
            fill = "white"
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(170, 170, 170), width=1)
        draw_cell_text(cell, replacements.get(cell.cell_id, cell.text), fill)

    return rendered


def cell_text_for_plan(cell: CellRef, plan: VariantPlan) -> str:
    for patch in plan.patches:
        if patch.cell.cell_id == cell.cell_id:
            return patch.new_text
    return cell.text


def render_evidence_packet_image(
    page_image: Any,
    packet: EvidencePacket,
    plan: VariantPlan,
    *,
    apply_patch_text: bool = False,
    padding: int = 8,
    gap: int = 8,
    scale: int = 3,
) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    source = image_to_pil(page_image)
    font = ImageFont.load_default()

    def crop_cell(cell: CellRef) -> Any:
        x0, y0, x1, y1 = cell.bbox
        box = (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(source.width, x1 + padding),
            min(source.height, y1 + padding),
        )
        crop = source.crop(box)
        replacement = cell_text_for_plan(cell, plan)
        if apply_patch_text and replacement != cell.text:
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

    sections: list[tuple[str, tuple[CellRef, ...]]] = [
        ("Table context", packet.spanning_headers),
        ("Column header", packet.column_headers),
        ("Unit / scale", packet.unit_cells),
        ("Row header", packet.row_headers),
        ("Value cell", (packet.value_cell,)),
    ]
    rendered_sections: list[tuple[str, list[Any]]] = []
    for label, cells in sections:
        crops = [crop_cell(cell) for cell in cells if cell.text or cell.cell_id == packet.value_cell.cell_id]
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


def cell_to_json(cell: CellRef) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "row_idx": cell.row_idx,
        "col_idx": cell.col_idx,
        "text": cell.text,
        "bbox": list(cell.bbox),
        "label": cell.label,
    }


def evidence_packet_to_json(packet: EvidencePacket) -> dict[str, Any]:
    return {
        "value_cell": cell_to_json(packet.value_cell),
        "row_headers": [cell_to_json(cell) for cell in packet.row_headers],
        "column_headers": [cell_to_json(cell) for cell in packet.column_headers],
        "unit_cells": [cell_to_json(cell) for cell in packet.unit_cells],
        "spanning_headers": [cell_to_json(cell) for cell in packet.spanning_headers],
        "all_cells": [cell_to_json(cell) for cell in packet.cells],
    }


def source_evidence_cell_for_plan(plan: VariantPlan) -> CellRef:
    for mapping in plan.cell_mappings:
        if mapping.rendered_cell.cell_id == plan.evidence_cell.cell_id:
            return mapping.source_cell
    return plan.evidence_cell


def cell_mapping_to_json(mapping: CellMapping, plan: VariantPlan) -> dict[str, Any]:
    patched_text = plan_patch_text_by_source_cell(plan)
    rendered_text = patched_text.get(mapping.source_cell.cell_id, mapping.source_cell.text)
    return {
        "source_cell_id": mapping.source_cell.cell_id,
        "rendered_cell_id": mapping.rendered_cell.cell_id,
        "source_bbox": list(mapping.source_cell.bbox),
        "bbox": list(mapping.rendered_cell.bbox),
        "rendered_bbox": list(mapping.rendered_cell.bbox),
        "source_text": mapping.source_cell.text,
        "rendered_text": rendered_text,
        "source_row_idx": mapping.source_cell.row_idx,
        "source_col_idx": mapping.source_cell.col_idx,
        "rendered_row_idx": mapping.rendered_cell.row_idx,
        "rendered_col_idx": mapping.rendered_cell.col_idx,
    }


def box_mapping_to_json(plan: VariantPlan) -> dict[str, Any]:
    moved = non_identity_mappings(plan)
    return {
        "type": "cell_mapping",
        "mapping_policy": "explicit" if moved else "identity",
        "changed_source_cell_ids": [
            mapping.source_cell.cell_id
            for mapping in plan.cell_mappings
            if mapping.source_cell.cell_id != mapping.rendered_cell.cell_id
            or plan_patch_text_by_source_cell(plan).get(mapping.source_cell.cell_id) is not None
        ],
        "source_to_rendered": {
            mapping.source_cell.cell_id: cell_mapping_to_json(mapping, plan)
            for mapping in plan.cell_mappings
        },
        "cells": [cell_mapping_to_json(mapping, plan) for mapping in plan.cell_mappings],
    }


def plan_to_json(plan: VariantPlan) -> dict[str, Any]:
    source_evidence_cell = source_evidence_cell_for_plan(plan)
    return {
        "variant": plan.variant,
        "intervention_type": plan.intervention_type,
        "answer": plan.answer,
        "evidence_cell": cell_to_json(plan.evidence_cell),
        "gold_evidence_ids": [source_evidence_cell.cell_id],
        "gold_evidence_text": plan.evidence_cell.text,
        "source_evidence_cell": cell_to_json(source_evidence_cell),
        "box_mapping": box_mapping_to_json(plan),
        "patches": [
            {
                "cell": cell_to_json(patch.cell),
                "old_text": patch.old_text,
                "new_text": patch.new_text,
            }
            for patch in plan.patches
        ],
        "validation_status": plan.validation_status,
        "validation_notes": list(plan.validation_notes),
    }


def prepare_output_dir(output_root: Path, config_path: Path) -> tuple[Path, Path]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use a fresh run directory to keep the pilot reproducible."
        )

    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "config.yaml")
    return output_root, images_dir


def write_jsonl_record(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def dataclass_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
