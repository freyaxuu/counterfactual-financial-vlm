"""Source-level table re-rendering and counterfactual planning for TAT-QA.

TAT-QA (the original tabular-and-textual QA dataset, not the TAT-DQA PDF-page
variant already integrated in this repo) ships each table as a fully expanded
text grid (``table.table``: list of rows of strings) with no accompanying
pixel image. That makes it a good fit for the "prefer source-level
re-rendering over pixel overlays" rule in ``AGENTS.md``: every image, clean or
counterfactual, can be laid out and rendered directly from the structured
grid, mirroring the ``source_table_rerender`` method used for SynFinTabs
(``financial_vlm.data.synfintabs_pilot``) but without needing pre-existing
cell bboxes to reconcile against.

This module only builds counterfactual *plans* and renders the resulting
grid images; it intentionally does not decide dataset roles or write
canonical-schema records. Per ``AGENTS.md`` dataset roles, TAT-QA is public
real-world data and may only be used for synthetic-to-real evaluation /
public benchmarking, never for training or hyperparameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from financial_vlm.data.synfintabs_pilot import make_counterfactual_number
from financial_vlm.data.tatdqa_pilot import normalize_number_text, scalar_answer_text


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
NUMERIC_RE = re.compile(r"^\(?-?[$£€]?\s*\d[\d,]*(?:\.\d+)?%?\)?$")
# Word-boundary unit/scale matcher. Unlike synfintabs_pilot.is_unit_or_scale_text
# (which substring-matches a bare "m" token), TAT-QA header cells are full
# natural-language phrases (e.g. "Years Ended December 31,") where a bare "m"
# substring-matches inside ordinary words like "December" -- so this needs
# word boundaries to avoid misclassifying spanning headers as unit cells.
UNIT_OR_SCALE_RE = re.compile(
    r"[$£€%]|\b(?:usd|eur|gbp|mn|bn|000|million|millions|thousand|thousands|billion|billions)\b",
    re.IGNORECASE,
)


def is_unit_or_scale_text(text: str) -> bool:
    compact = " ".join(text.strip().split())
    if not compact:
        return False
    return bool(UNIT_OR_SCALE_RE.search(compact))


@dataclass(frozen=True)
class GridCell:
    row_idx: int
    col_idx: int
    text: str

    @property
    def cell_id(self) -> str:
        return f"r{self.row_idx:02d}c{self.col_idx:02d}"


@dataclass(frozen=True)
class LocatedQuestion:
    question_id: str
    question: str
    answer: str
    answer_cell: GridCell


@dataclass(frozen=True)
class CellPatch:
    cell: GridCell
    old_text: str
    new_text: str


@dataclass(frozen=True)
class VariantPlan:
    variant: str
    intervention_type: str
    answer: str
    evidence_cell: GridCell
    patches: tuple[CellPatch, ...]
    validation_status: str
    validation_notes: tuple[str, ...]


def is_numeric_text(text: str) -> bool:
    compact = " ".join(text.strip().split())
    return bool(NUMERIC_RE.match(compact))


def flatten_grid(rows: Sequence[Sequence[Any]]) -> tuple[GridCell, ...]:
    cells: list[GridCell] = []
    for row_idx, row in enumerate(rows):
        for col_idx, raw_text in enumerate(row):
            text = "" if raw_text is None else str(raw_text).strip()
            cells.append(GridCell(row_idx=row_idx, col_idx=col_idx, text=text))
    return tuple(cells)


def locate_answer_cell(question: Mapping[str, Any], cells: Sequence[GridCell]) -> LocatedQuestion | None:
    """Locate the unique grid cell whose text matches the scalar answer.

    TAT-QA carries no explicit cell-level answer localisation (unlike
    TAT-DQA's ``block_mapping``), so this matches by normalised text against
    every non-empty cell and requires a single unambiguous hit.
    """

    answer_text = scalar_answer_text(question.get("answer"))
    if answer_text is None:
        return None
    normalised_answer = normalize_number_text(answer_text)
    if not normalised_answer:
        return None

    matches = [cell for cell in cells if cell.text and normalize_number_text(cell.text) == normalised_answer]
    if len(matches) != 1:
        return None

    question_id = str(question.get("uid") or question.get("id") or "")
    return LocatedQuestion(
        question_id=question_id,
        question=str(question.get("question") or ""),
        answer=answer_text,
        answer_cell=matches[0],
    )


def find_header_swap(
    cells: Sequence[GridCell],
    answer_cell: GridCell,
) -> tuple[GridCell, GridCell, GridCell] | None:
    """Find a year-like column header above the answer cell and a peer header/value pair.

    Mirrors ``synfintabs_pilot.find_header_swap`` but works on an unlabelled
    text grid: "header-ness" is approximated by year-token presence, which
    covers the dominant column-header style in TAT-QA's financial tables.
    """

    by_row_col = {(cell.row_idx, cell.col_idx): cell for cell in cells}

    header = None
    for row_idx in range(answer_cell.row_idx - 1, -1, -1):
        candidate = by_row_col.get((row_idx, answer_cell.col_idx))
        if candidate is not None and YEAR_RE.search(candidate.text):
            header = candidate
            break
    if header is None:
        return None

    header_row_cells = [cell for cell in cells if cell.row_idx == header.row_idx]
    peer_headers = [
        cell for cell in header_row_cells if cell.col_idx != header.col_idx and YEAR_RE.search(cell.text)
    ]
    peer_headers.sort(key=lambda cell: cell.col_idx)

    for peer_header in peer_headers:
        peer_value = by_row_col.get((answer_cell.row_idx, peer_header.col_idx))
        if (
            peer_value is not None
            and peer_value.text
            and is_numeric_text(peer_value.text)
            and normalize_number_text(peer_value.text) != normalize_number_text(answer_cell.text)
        ):
            return header, peer_header, peer_value

    return None


def find_irrelevant_cell(
    cells: Sequence[GridCell],
    answer_cell: GridCell,
    forbidden: set[tuple[int, int]],
) -> GridCell | None:
    candidates = [
        cell
        for cell in cells
        if (cell.row_idx, cell.col_idx) not in forbidden
        and cell.row_idx != answer_cell.row_idx
        and cell.text
        and is_numeric_text(cell.text)
        and not YEAR_RE.search(cell.text)
        and normalize_number_text(cell.text) != normalize_number_text(answer_cell.text)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda cell: abs(cell.row_idx - answer_cell.row_idx) + abs(cell.col_idx - answer_cell.col_idx),
    )


def build_variant_plans(
    located: LocatedQuestion,
    cells: Sequence[GridCell],
    rng: Any,
) -> tuple[VariantPlan, ...] | None:
    """Build the four SynFinTabs-style variants (clean + 3 counterfactuals).

    Returns ``None`` when the source table cannot support a full variant set
    (no year-style header pair, or no safe irrelevant cell to perturb) — the
    caller should count and report this as a rejection, not a silent drop.
    """

    answer_cell = located.answer_cell
    if not is_numeric_text(located.answer) or not is_numeric_text(answer_cell.text):
        return None

    header_swap = find_header_swap(cells, answer_cell)
    if header_swap is None:
        return None
    header, peer_header, peer_value = header_swap

    # The irrelevant-value patch must not touch any cell the clean variant
    # reports as gold evidence (row/column headers, unit/scale cells,
    # spanning headers) -- not just the answer cell and header-swap cells.
    # cf_cycle_sampler.parse_group_records enforces this at training-manifest
    # load time; computing it here means a bad pick is rejected (None) rather
    # than surfacing as a training-time ValueError.
    clean_evidence = build_evidence_packet(cells, answer_cell)
    clean_evidence_coords = {
        (cell.row_idx, cell.col_idx)
        for cell in (
            clean_evidence.value_cell,
            *clean_evidence.row_headers,
            *clean_evidence.column_headers,
            *clean_evidence.unit_cells,
            *clean_evidence.spanning_headers,
        )
    }

    irrelevant_cell = find_irrelevant_cell(
        cells,
        answer_cell,
        forbidden={
            *clean_evidence_coords,
            (header.row_idx, header.col_idx),
            (peer_header.row_idx, peer_header.col_idx),
            (peer_value.row_idx, peer_value.col_idx),
        },
    )
    if irrelevant_cell is None:
        return None

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
        ),
        VariantPlan(
            variant="target_value_replacement",
            intervention_type="target-value replacement",
            answer=target_value,
            evidence_cell=GridCell(answer_cell.row_idx, answer_cell.col_idx, target_value),
            patches=(CellPatch(answer_cell, answer_cell.text, target_value),),
            validation_status="passed",
            validation_notes=("answer cell text was replaced; target evidence unchanged",),
        ),
        VariantPlan(
            variant="header_address_swap",
            intervention_type="header/address swap",
            answer=peer_value.text,
            evidence_cell=peer_value,
            patches=(
                CellPatch(header, header.text, peer_header.text),
                CellPatch(peer_header, peer_header.text, header.text),
            ),
            validation_status="passed",
            validation_notes=("question address now resolves to peer column after header swap",),
        ),
        VariantPlan(
            variant="irrelevant_value_replacement",
            intervention_type="irrelevant-value replacement",
            answer=located.answer,
            evidence_cell=answer_cell,
            patches=(CellPatch(irrelevant_cell, irrelevant_cell.text, irrelevant_value),),
            validation_status="passed",
            validation_notes=("non-target numeric cell was replaced; answer should remain stable",),
        ),
    )


def patched_text_by_cell_id(plan: VariantPlan) -> dict[str, str]:
    return {patch.cell.cell_id: patch.new_text for patch in plan.patches}


def rendered_cells_for_plan(cells: Sequence[GridCell], plan: VariantPlan) -> tuple[GridCell, ...]:
    """Apply a plan's patches, returning cells with the text that would be rendered."""

    patched = patched_text_by_cell_id(plan)
    return tuple(
        GridCell(cell.row_idx, cell.col_idx, patched.get(cell.cell_id, cell.text)) for cell in cells
    )


@dataclass(frozen=True)
class EvidencePacket:
    value_cell: GridCell
    row_headers: tuple[GridCell, ...]
    column_headers: tuple[GridCell, ...]
    unit_cells: tuple[GridCell, ...]
    spanning_headers: tuple[GridCell, ...]


def build_evidence_packet(cells: Sequence[GridCell], value_cell: GridCell) -> EvidencePacket:
    """Classify the header context around a value cell from an unlabelled grid.

    Row headers are non-numeric cells to the left in the same row (typically
    the metric/line-item label in column 0). Column context is every
    non-empty cell above in the same column, split into year-token column
    headers, unit/scale cells (``is_unit_or_scale_text``), and any remaining
    higher-row cells as spanning headers (e.g. a "Years Ended December 31,"
    title row above the year row).
    """

    row_headers = tuple(
        cell
        for cell in cells
        if cell.row_idx == value_cell.row_idx and cell.col_idx < value_cell.col_idx and cell.text and not is_numeric_text(cell.text)
    )
    column_context = [
        cell for cell in cells if cell.col_idx == value_cell.col_idx and cell.row_idx < value_cell.row_idx and cell.text
    ]
    column_headers = tuple(cell for cell in column_context if YEAR_RE.search(cell.text))
    unit_cells = tuple(
        cell for cell in column_context if cell not in column_headers and is_unit_or_scale_text(cell.text)
    )
    spanning_headers = tuple(
        cell for cell in column_context if cell not in column_headers and cell not in unit_cells
    )

    return EvidencePacket(
        value_cell=value_cell,
        row_headers=row_headers,
        column_headers=column_headers,
        unit_cells=unit_cells,
        spanning_headers=spanning_headers,
    )


Bbox = tuple[int, int, int, int]


def layout(
    cells: Sequence[GridCell],
    nrows: int,
    ncols: int,
    *,
    font: Any,
    col_padding: int = 12,
    row_padding: int = 6,
    min_col_width: int = 40,
    min_row_height: int = 20,
) -> tuple[list[int], list[int]]:
    """Compute cumulative column-x and row-y boundaries from measured cell text.

    No pre-existing bboxes are used or required: this is a from-scratch grid
    layout, which is the point of the feasibility check — TAT-QA's source has
    no pixel geometry to begin with. ``cells`` must already carry the text to
    be rendered (see ``rendered_cells_for_plan``).
    """

    from PIL import Image, ImageDraw

    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    col_widths = [min_col_width] * ncols
    row_heights = [min_row_height] * nrows
    by_row_col = {(cell.row_idx, cell.col_idx): cell for cell in cells}

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            cell = by_row_col.get((row_idx, col_idx))
            text = cell.text if cell is not None else ""
            if not text:
                continue
            box = measurer.textbbox((0, 0), text, font=font)
            width = box[2] - box[0] + 2 * col_padding
            height = box[3] - box[1] + 2 * row_padding
            col_widths[col_idx] = max(col_widths[col_idx], width)
            row_heights[row_idx] = max(row_heights[row_idx], height)

    col_x = [0]
    for width in col_widths:
        col_x.append(col_x[-1] + width)
    row_y = [0]
    for height in row_heights:
        row_y.append(row_y[-1] + height)
    return col_x, row_y


def render_grid_image(
    cells: Sequence[GridCell],
    nrows: int,
    ncols: int,
    *,
    highlighted_cell_ids: frozenset[str] = frozenset(),
    col_padding: int = 12,
    row_padding: int = 6,
) -> tuple[Any, dict[str, Bbox]]:
    """Render a grid from cells that already carry their final display text.

    Returns the image and a ``{cell_id: pixel_bbox}`` map computed from the
    exact same layout used to draw it, so evidence bboxes never drift from
    what is actually on the rendered image. ``col_padding``/``row_padding``
    are exposed so callers (e.g. a compute-matched "clean B" re-render) can
    jitter the layout to be pixel-distinct while keeping the same content.
    """

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    col_x, row_y = layout(cells, nrows, ncols, font=font, col_padding=col_padding, row_padding=row_padding)

    image = Image.new("RGB", (col_x[-1] or 1, row_y[-1] or 1), "white")
    draw = ImageDraw.Draw(image)
    by_row_col = {(cell.row_idx, cell.col_idx): cell for cell in cells}
    bboxes: dict[str, Bbox] = {}

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            cell = by_row_col.get((row_idx, col_idx))
            x0, x1 = col_x[col_idx], col_x[col_idx + 1]
            y0, y1 = row_y[row_idx], row_y[row_idx + 1]
            if cell is None:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill="white", outline=(170, 170, 170), width=1)
                continue
            bboxes[cell.cell_id] = (x0, y0, x1, y1)
            text = cell.text
            fill = (255, 246, 214) if cell.cell_id in highlighted_cell_ids else "white"
            if fill == "white" and YEAR_RE.search(text):
                fill = (236, 240, 245)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill, outline=(170, 170, 170), width=1)
            if text:
                draw.text((x0 + 6, y0 + 4), text, fill=(0, 0, 0), font=font)

    return image, bboxes


def render_variant_image(
    cells: Sequence[GridCell],
    nrows: int,
    ncols: int,
    plan: VariantPlan,
    *,
    highlight_changed: bool = True,
) -> tuple[Any, dict[str, Bbox], tuple[GridCell, ...]]:
    """Render one variant, returning its image, cell bboxes, and rendered cells.

    The rendered cells are also returned so callers can build an
    ``EvidencePacket`` against the same post-patch text state that produced
    the image (headers may have swapped text for ``header_address_swap``).

    ``highlight_changed`` tints the intervened cell(s) so a reviewer can see
    what an intervention touched. It defaults to ``True`` for backwards
    compatibility, but marking the edited cells is a visible cue a clean
    image does not carry (and SynFinTabs' renderer does not add), so pass
    ``highlight_changed=False`` when the rendered surface must be free of
    that confound.
    """

    rendered_cells = rendered_cells_for_plan(cells, plan)
    changed_ids = (
        frozenset(patch.cell.cell_id for patch in plan.patches) if highlight_changed else frozenset()
    )
    image, bboxes = render_grid_image(rendered_cells, nrows, ncols, highlighted_cell_ids=changed_ids)
    return image, bboxes, rendered_cells
