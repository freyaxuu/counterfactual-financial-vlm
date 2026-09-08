"""Adapter for the private `evolution-ai-datasets` package.

Scope: `company_profile` field-group extraction for the annotation-reliability
audit only. All `evolution_ai_datasets` imports stay inside this module, and
the import is deferred so this module (and its dataclasses/constants) can be
imported from unit tests and other business logic without the private package
installed, per AGENTS.md's private-package boundary.

Field maps below were derived from `document_types_schema.json` and one
approved sample document (the private dataset, document type
`quarterly_report`, inspected on the training server via the dataset access group). `tables` and
`grouped_fields` are two independent annotation surfaces for the same
underlying company data:

- `grouped_fields["company_profile"]` — one instance per portfolio company,
  usually on a dedicated "company profile" detail page, with the 26 fields in
  `COMPANY_PROFILE_GROUPED_FIELDS`.
- `tables["company_profile"]` — a per-page summary table listing many
  portfolio companies as rows; besides the fields shared with the grouped
  representation (mapped in `TABLE_TO_GROUPED_FIELD_MAP`, some suffixed
  `_table`), it also carries extra financial-performance columns
  (LTM/YTD/budget/forecast figures, net debt, valuation multiples, ...) that
  have no `company_profile` grouped-field counterpart at all.

WARNING -- undocumented attribute dependency: `Field.textblock.coords`
(`BoundingBox.top/left/bottom/right`) is used below to detect table-region
conflation, but neither `textblock` nor `TextBlock`/`BoundingBox` appear in
`docs/evolution-ai-datasets.md`'s documented public API (only
`fields`/`page_fields`/`grouped_fields`/`tables`/`document_type_ref`/`image`
are documented). This was found by introspecting the installed package
(`evolution-ai-datasets==0.1.1`) on the training server, not from the docs. It is a
deliberate, risk-accepted exception to AGENTS.md's "no undocumented
attributes" rule -- geometric evidence of table conflation was judged worth
the risk of breaking silently on a private-package upgrade. Any code path
touching `cell_bboxes` should tolerate `None`/missing coords rather than
assume they exist.

Matching an instance to a table row has no shared numeric key (row indices
and instance indices are independent counters) — the only reliable join key
observed is the normalized `company_name` text, within the same document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

COMPANY_PROFILE_GROUPED_FIELDS: tuple[str, ...] = (
    "FYE",
    "additional_realizations_1",
    "additional_realizations_2",
    "company_name",
    "country/City",
    "current_invested_capital",
    "deal_source",
    "description",
    "entry_date",
    "entry_ownership",
    "exit_date",
    "exit_status",
    "exit_style",
    "geography",
    "gross_IRR",
    "gross_MOIC",
    "industry",
    "minus_realizations",
    "ownership",
    "realised_cost",
    "realized_value",
    "stock_exchange",
    "ticker",
    "total_invested_capital",
    "total_value",
    "unrealized_value",
    "website",
)

# `da` (data_type) per field from document_types_schema.json.
COMPANY_PROFILE_FIELD_DATA_TYPES: Mapping[str, str] = {
    "FYE": "date",
    "additional_realizations_1": "monetary",
    "additional_realizations_2": "monetary",
    "company_name": "text",
    "country/City": "text",
    "current_invested_capital": "monetary",
    "deal_source": "text",
    "description": "text",
    "entry_date": "date",
    "entry_ownership": "numerical",
    "exit_date": "date",
    "exit_status": "text",
    "exit_style": "text",
    "geography": "text",
    "gross_IRR": "numerical",
    "gross_MOIC": "numerical",
    "industry": "text",
    "minus_realizations": "monetary",
    "ownership": "numerical",
    "realised_cost": "monetary",
    "realized_value": "monetary",
    "stock_exchange": "text",
    "ticker": "text",
    "total_invested_capital": "monetary",
    "total_value": "monetary",
    "unrealized_value": "monetary",
    "website": "text",
}

# table cell field name -> grouped_fields field name, restricted to fields
# present in both representations of company_profile.
TABLE_TO_GROUPED_FIELD_MAP: Mapping[str, str] = {
    "company_name": "company_name",
    "country/City": "country/City",
    "industry": "industry",
    "exit_style": "exit_style",
    "current_invested_capital_table": "current_invested_capital",
    "entry_date_table": "entry_date",
    "exit_date_table": "exit_date",
    "gross_IRR_table": "gross_IRR",
    "gross_MOIC_table": "gross_MOIC",
    "ownership_table": "ownership",
    "realised_cost_table": "realised_cost",
    "realized_value_table": "realized_value",
    "total_invested_capital_table": "total_invested_capital",
    "total_value_table": "total_value",
    "unrealized_value_table": "unrealized_value",
    "entry_ownership_table": "entry_ownership",
    "minus_realizations_table": "minus_realizations",
    "additional_realizations_1_table": "additional_realizations_1",
    "additional_realizations_2_table": "additional_realizations_2",
}


@dataclass(frozen=True)
class CompanyProfileInstance:
    """One `grouped_fields["company_profile"]` instance (one portfolio company)."""

    document_id: str
    page_id: str
    instance_index: int
    field_values: Mapping[str, str]


# (top, left, bottom, right) pixel coordinates, from Field.textblock.coords.
BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class CompanyProfileTableRow:
    """One row of a `tables["company_profile"]` summary table."""

    document_id: str
    page_id: str
    row_index: int
    cell_values: Mapping[str, str]
    cell_bboxes: Mapping[str, BoundingBox]


@dataclass(frozen=True)
class CompanyProfileDocument:
    document_id: str
    instances: tuple[CompanyProfileInstance, ...]
    table_rows: tuple[CompanyProfileTableRow, ...]


def load_company_profile_documents(dataset_root: Path) -> tuple[CompanyProfileDocument, ...]:
    """Load `company_profile` grouped-field instances and table rows per document.

    Deferred import of the private `evolution_ai_datasets` package (unavailable
    locally; installed on the the training server venv). Uses only the documented
    loading API (`evolution_ai_datasets.serialization.load_dataset`) and the
    documented public attributes (`page.grouped_fields`, `page.tables`,
    `field.value`).
    """

    from evolution_ai_datasets.serialization import load_dataset  # deferred import

    dataset, _document_types = load_dataset(dataset_root, load_ocr=False)

    documents: list[CompanyProfileDocument] = []
    for doc in dataset.documents:
        instances: list[CompanyProfileInstance] = []
        table_rows: list[CompanyProfileTableRow] = []

        for page in doc.pages:
            grouped = page.grouped_fields.get("company_profile") or {}
            for instance_index, fields in grouped.items():
                values = {
                    name: str(field.value)
                    for name, field in fields.items()
                    if str(field.value or "").strip()
                }
                if not values:
                    continue
                instances.append(
                    CompanyProfileInstance(
                        document_id=doc.id,
                        page_id=page.id,
                        instance_index=int(instance_index),
                        field_values=values,
                    )
                )

            table = page.tables.get("company_profile") or {}
            for row_index, cells in table.items():
                values = {
                    name: str(field.value)
                    for name, field in cells.items()
                    if str(field.value or "").strip()
                }
                if not values:
                    continue
                bboxes: dict[str, BoundingBox] = {}
                for name in values:
                    textblock = cells[name].textblock
                    if textblock is None or textblock.coords is None:
                        continue
                    coords = textblock.coords
                    bboxes[name] = (coords.top, coords.left, coords.bottom, coords.right)
                table_rows.append(
                    CompanyProfileTableRow(
                        document_id=doc.id,
                        page_id=page.id,
                        row_index=int(row_index),
                        cell_values=values,
                        cell_bboxes=bboxes,
                    )
                )

        documents.append(
            CompanyProfileDocument(
                document_id=doc.id,
                instances=tuple(instances),
                table_rows=tuple(table_rows),
            )
        )

    return tuple(documents)


# --- Generic grouped-field / document-level-field / OCR loading -----------
#
# Added for the company-benchmark diagnosis in
# `docs/company-benchmark-diagnosis-report.md`. Unlike the `company_profile`-
# specific dataclasses/loader above, these are generic across
# `document_types_schema.json`'s `grouped_fields` groups (`KPIs`, `table`) so
# a new group doesn't need a bespoke dataclass. The existing
# `CompanyProfileInstance`/`load_company_profile_documents` are left
# untouched to avoid disturbing `company_profile_audit.py` and its callers.

KPI_METRIC_FIELDS: tuple[str, ...] = (
    "sales",
    "EBITDA",
    "net_debt",
    "EV",
    "valuation_multiple",
    "net_leverage_multiple",
    "Cash and Cash Equivalents",
    "total_debt",
)

KPI_GROUPED_FIELDS: tuple[str, ...] = (
    "company_name",
    "year",
    "month",
    "period",
    *KPI_METRIC_FIELDS,
)


@dataclass(frozen=True)
class GroupedFieldInstance:
    """One instance of an arbitrary `grouped_fields[group_name]` group.

    Generic counterpart to `CompanyProfileInstance` above -- used for groups
    (`KPIs`, `table`) that don't warrant their own bespoke dataclass. Carries
    bounding boxes (see the WARNING above re: `Field.textblock.coords`)
    because the `KPIs` group is only usable for confusion-pair construction
    if each instance's on-page position is known -- see
    `financial_vlm.evaluation.kpi_confusion_pairs`.
    """

    document_id: str
    page_id: str
    group_name: str
    instance_index: int
    field_values: Mapping[str, str]
    field_bboxes: Mapping[str, BoundingBox]


def load_grouped_field_instances(dataset_root: Path, group_name: str) -> tuple[GroupedFieldInstance, ...]:
    """Load every populated instance of `grouped_fields[group_name]`.

    Generic -- unlike `load_company_profile_documents`, this doesn't assume
    `company_profile` specifically. Used for the `KPIs` and `table` groups
    (see `document_types_schema.json`'s three `grouped_fields` groups for
    `quarterly_report`).
    """

    from evolution_ai_datasets.serialization import load_dataset  # deferred import

    dataset, _document_types = load_dataset(dataset_root, load_ocr=False)

    instances: list[GroupedFieldInstance] = []
    for doc in dataset.documents:
        for page in doc.pages:
            group = (page.grouped_fields or {}).get(group_name) or {}
            for instance_index, fields in group.items():
                values: dict[str, str] = {}
                bboxes: dict[str, BoundingBox] = {}
                for name, field in fields.items():
                    value = str(field.value).strip() if field.value is not None else ""
                    if not value:
                        continue
                    values[name] = value
                    textblock = field.textblock
                    if textblock is not None and textblock.coords is not None:
                        coords = textblock.coords
                        bboxes[name] = (coords.top, coords.left, coords.bottom, coords.right)
                if not values:
                    continue
                instances.append(
                    GroupedFieldInstance(
                        document_id=doc.id,
                        page_id=page.id,
                        group_name=group_name,
                        instance_index=int(instance_index),
                        field_values=values,
                        field_bboxes=bboxes,
                    )
                )
    return tuple(instances)


@dataclass(frozen=True)
class DocumentLevelField:
    """One page's populated `fields` (document-level, non-grouped) values."""

    document_id: str
    page_id: str
    field_values: Mapping[str, str]


def load_document_level_fields(dataset_root: Path) -> tuple[DocumentLevelField, ...]:
    """Load every page's populated `fields` (e.g. fund NAV/IRR/vintage)."""

    from evolution_ai_datasets.serialization import load_dataset  # deferred import

    dataset, _document_types = load_dataset(dataset_root, load_ocr=False)

    records: list[DocumentLevelField] = []
    for doc in dataset.documents:
        for page in doc.pages:
            fields = page.fields or {}
            values = {
                name: str(field.value).strip() for name, field in fields.items() if str(field.value or "").strip()
            }
            if values:
                records.append(DocumentLevelField(document_id=doc.id, page_id=page.id, field_values=values))
    return tuple(records)


def load_document_page_ids(dataset_root: Path) -> tuple[tuple[str, str], ...]:
    """Every (document_id, page_id) pair in the dataset -- the page universe,
    used to compute how many pages carry no annotation in any surface."""

    from evolution_ai_datasets.serialization import load_dataset  # deferred import

    dataset, _document_types = load_dataset(dataset_root, load_ocr=False)
    return tuple((doc.id, page.id) for doc in dataset.documents for page in doc.pages)


@dataclass(frozen=True)
class OCRToken:
    """One OCR word box, from a page's `ocr.json` sidecar file.

    WARNING -- undocumented on-disk schema dependency: `docs/evolution-ai-datasets.md`
    documents the *location* of the per-page OCR sidecar file
    (`files/<doc-id>/pages/<page-id>/ocr.json`) but not the JSON schema
    inside it. The `c`/`cf`/`t` keys below (coords/confidence/text) were
    reverse-engineered by inspecting real sidecar files on the training server, not
    from the docs. Same deliberate, risk-accepted exception as
    `Field.textblock.coords` above -- treat this as fragile to a private
    package/export-format upgrade, and re-confirm the shape before trusting
    it again.
    """

    top: int
    left: int
    bottom: int
    right: int
    text: str
    confidence: int | None


def load_ocr_tokens(dataset_root: Path, document_id: str, page_id: str) -> tuple[OCRToken, ...]:
    """Read one page's OCR sidecar file directly (bypasses `load_dataset`;
    see the WARNING on `OCRToken`). Returns an empty tuple if the file is
    missing, or if an entry is malformed, rather than raising -- not every
    page necessarily has OCR, and malformed entries shouldn't abort a whole
    audit run over hundreds of pages."""

    path = dataset_root / "files" / document_id / "pages" / page_id / "ocr.json"
    if not path.exists():
        return ()
    raw = json.loads(path.read_text())
    tokens: list[OCRToken] = []
    for item in raw:
        coords = item.get("c")
        text = item.get("t")
        if not coords or len(coords) != 4 or not text:
            continue
        top, left, bottom, right = coords
        tokens.append(OCRToken(top=top, left=left, bottom=bottom, right=right, text=text, confidence=item.get("cf")))
    return tuple(tokens)
