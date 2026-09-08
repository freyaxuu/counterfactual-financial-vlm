"""Prepare TAT-DQA clean extraction samples for the SynFinTabs-style manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence


NUMERIC_ANSWER_RE = re.compile(r"^\(?-?[$£€]?\s*\d[\d,]*(?:\.\d+)?%?\)?$")
BINARY_MINUS_RE = re.compile(r"\d\s*-\s*\d")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
QUARTER_RE = re.compile(r"\b(?:Q[1-4]|first|second|third|fourth)\s+quarter\b", re.IGNORECASE)
MONTH_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
    r"\d{1,2},?\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z&'-]*|\d{4}")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "company",
    "companies",
    "did",
    "does",
    "during",
    "ended",
    "ending",
    "fiscal",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "period",
    "reported",
    "shown",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "year",
    "years",
}


@dataclass(frozen=True)
class TatDqaWord:
    index: int
    text: str
    bbox: tuple[int, int, int, int]
    char_start: int
    char_end: int


@dataclass(frozen=True)
class TatDqaBlock:
    uuid: str
    page_index: int
    order: int
    text: str
    bbox: tuple[int, int, int, int]
    words: tuple[TatDqaWord, ...]


@dataclass(frozen=True)
class TatDqaEvidence:
    block: TatDqaBlock
    span: tuple[int, int]
    span_mode: str
    text: str
    bbox: tuple[int, int, int, int]
    word_indices: tuple[int, ...]

    @property
    def cell_id(self) -> str:
        start, end = self.span
        short_uuid = self.block.uuid[:8] or "block"
        return f"p{self.block.page_index:02d}_{short_uuid}_{self.span_mode}_{start}_{end}"


@dataclass(frozen=True)
class MetricPeriodMatch:
    accepted: bool
    reason: str
    metric_candidate: str | None
    metric_tokens: tuple[str, ...]
    matched_metric_tokens: tuple[str, ...]
    metric_overlap_ratio: float
    period_candidates: tuple[str, ...]
    matched_periods: tuple[str, ...]
    context_block_ids: tuple[str, ...]


def bbox_tuple(raw_bbox: Sequence[Any] | None) -> tuple[int, int, int, int]:
    if raw_bbox is None or len(raw_bbox) != 4:
        raise ValueError(f"Expected bbox with four coordinates, got {raw_bbox!r}")
    x0, y0, x1, y1 = (int(round(float(v))) for v in raw_bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox {raw_bbox!r}")
    return x0, y0, x1, y1


def union_bbox(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("Cannot union an empty bbox list")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def compact_text(value: Any) -> str:
    return " ".join(("" if value is None else str(value)).strip().split())


def scalar_answer_text(answer: Any) -> str | None:
    if isinstance(answer, (list, tuple)) and len(answer) == 1:
        answer = answer[0]
    if isinstance(answer, (list, tuple, dict)):
        return None
    text = compact_text(answer)
    return text or None


def is_numeric_answer(text: str) -> bool:
    return bool(NUMERIC_ANSWER_RE.match(compact_text(text)))


def normalize_number_text(value: Any) -> str:
    text = compact_text(value).casefold()
    text = text.strip("\"'`")
    text = text.replace(",", "")
    text = text.replace("$", "").replace("£", "").replace("€", "")
    text = text.replace("%", "")
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1].strip()}"
    return text


def has_arithmetic_operator(derivation: str) -> bool:
    compact = derivation.strip()
    if any(operator in compact for operator in ("+", "*", "/", "=")):
        return True
    return bool(BINARY_MINUS_RE.search(compact))


def normalise_token(token: str) -> str:
    return token.strip("'_-").casefold()


def question_without_periods(question_text: str) -> str:
    text = MONTH_DATE_RE.sub(" ", question_text)
    text = YEAR_RE.sub(" ", text)
    text = QUARTER_RE.sub(" ", text)
    return text


def content_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in TOKEN_RE.findall(text):
        token = normalise_token(raw)
        if not token or token in STOPWORDS or token.isdigit():
            continue
        if token not in seen:
            tokens.append(token)
            seen.add(token)
    return tuple(tokens)


def extract_period_candidates(question_text: str) -> tuple[str, ...]:
    periods: list[str] = []
    seen: set[str] = set()

    for match in MONTH_DATE_RE.finditer(question_text):
        value = compact_text(match.group(0))
        if value.casefold() not in seen:
            periods.append(value)
            seen.add(value.casefold())

    for match in YEAR_RE.finditer(question_text):
        value = match.group(0)
        if value not in seen:
            periods.append(value)
            seen.add(value)

    for match in QUARTER_RE.finditer(question_text):
        value = compact_text(match.group(0))
        years = YEAR_RE.findall(question_text[match.end() : match.end() + 24])
        if years:
            value = f"{value} {years[0]}"
        key = value.casefold()
        if key not in seen:
            periods.append(value)
            seen.add(key)

    return tuple(periods)


def extract_metric_tokens(question_text: str) -> tuple[str, ...]:
    return content_tokens(question_without_periods(question_text))


def period_matches_context(period: str, context_text: str) -> bool:
    period_text = compact_text(period)
    if not period_text:
        return False
    if period_text.casefold() in context_text.casefold():
        return True
    years = YEAR_RE.findall(period_text)
    return bool(years) and all(year in context_text for year in years)


def context_blocks_for_evidence(
    blocks_by_uuid: Mapping[str, TatDqaBlock],
    evidence: TatDqaEvidence,
    *,
    window: int,
) -> tuple[TatDqaBlock, ...]:
    same_page = sorted(
        (block for block in blocks_by_uuid.values() if block.page_index == evidence.block.page_index),
        key=lambda block: block.order,
    )
    selected = [
        block
        for block in same_page
        if abs(block.order - evidence.block.order) <= window
    ]
    if evidence.block not in selected:
        selected.append(evidence.block)
        selected.sort(key=lambda block: block.order)
    return tuple(selected)


def match_metric_period(
    question: Mapping[str, Any],
    evidence: TatDqaEvidence,
    blocks_by_uuid: Mapping[str, TatDqaBlock],
    *,
    context_window: int = 2,
    require_metric: bool = True,
    require_period: bool = True,
    min_metric_overlap: int = 1,
    min_metric_overlap_ratio: float = 0.5,
) -> MetricPeriodMatch:
    question_text = str(question.get("question") or "")
    metric_tokens = extract_metric_tokens(question_text)
    period_candidates = extract_period_candidates(question_text)
    context_blocks = context_blocks_for_evidence(blocks_by_uuid, evidence, window=context_window)
    context_text = " ".join(block.text for block in context_blocks)
    context_token_set = set(content_tokens(context_text))

    matched_metric_tokens = tuple(token for token in metric_tokens if token in context_token_set)
    metric_overlap_ratio = (
        len(matched_metric_tokens) / len(metric_tokens)
        if metric_tokens
        else 0.0
    )
    matched_periods = tuple(
        period for period in period_candidates if period_matches_context(period, context_text)
    )

    if require_metric and not metric_tokens:
        reason = "missing_metric_candidate"
    elif require_metric and len(matched_metric_tokens) < min_metric_overlap:
        reason = "metric_not_matched"
    elif require_metric and metric_overlap_ratio < min_metric_overlap_ratio:
        reason = "metric_overlap_below_threshold"
    elif require_period and not period_candidates:
        reason = "missing_period_candidate"
    elif require_period and not matched_periods:
        reason = "period_not_matched"
    else:
        reason = "accepted"

    return MetricPeriodMatch(
        accepted=reason == "accepted",
        reason=reason,
        metric_candidate=" ".join(metric_tokens) or None,
        metric_tokens=metric_tokens,
        matched_metric_tokens=matched_metric_tokens,
        metric_overlap_ratio=round(metric_overlap_ratio, 6),
        period_candidates=period_candidates,
        matched_periods=matched_periods,
        context_block_ids=tuple(block.uuid for block in context_blocks),
    )


def is_direct_extraction_question(
    question: Mapping[str, Any],
    *,
    require_numeric_answer: bool = True,
) -> tuple[bool, str]:
    if bool(question.get("req_comparison")):
        return False, "comparison_required"

    answer_text = scalar_answer_text(question.get("answer"))
    if answer_text is None:
        return False, "non_scalar_answer"
    if require_numeric_answer and not is_numeric_answer(answer_text):
        return False, "non_numeric_answer"

    answer_type = str(question.get("answer_type") or "").casefold()
    if answer_type not in {"span", "arithmetic"}:
        return False, f"unsupported_answer_type_{answer_type or 'missing'}"

    facts = [compact_text(fact) for fact in question.get("facts") or [] if compact_text(fact)]

    if answer_type == "arithmetic":
        derivation = compact_text(question.get("derivation"))
        if not derivation:
            return False, "missing_direct_derivation"
        if has_arithmetic_operator(derivation):
            return False, "arithmetic_derivation"
        normalized_derivation = normalize_number_text(derivation)
        normalized_answer = normalize_number_text(answer_text)
        normalized_facts = {normalize_number_text(fact) for fact in facts}
        if normalized_derivation not in {normalized_answer, *normalized_facts}:
            return False, "derivation_not_direct_answer"

    return True, "accepted"


def load_json(path: Path) -> Any:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_qa_records(raw_qa: Any) -> Iterable[tuple[int, Mapping[str, Any], int, Mapping[str, Any]]]:
    records = raw_qa.get("data") if isinstance(raw_qa, Mapping) and "data" in raw_qa else raw_qa
    if not isinstance(records, list):
        raise ValueError("TAT-DQA QA file must contain a list of document records")
    for doc_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        doc = record.get("doc") or {}
        questions = record.get("questions") or []
        if not isinstance(doc, Mapping) or not isinstance(questions, list):
            continue
        for question_index, question in enumerate(questions):
            if isinstance(question, Mapping):
                yield doc_index, doc, question_index, question


def build_doc_index(raw_doc: Mapping[str, Any]) -> tuple[dict[str, TatDqaBlock], list[Mapping[str, Any]]]:
    pages = raw_doc.get("pages") or []
    if not isinstance(pages, list):
        raise ValueError("TAT-DQA document JSON must contain a pages list")

    blocks: dict[str, TatDqaBlock] = {}
    for page_index, page in enumerate(pages):
        for block in page.get("blocks") or []:
            uuid = str(block.get("uuid") or "")
            if not uuid:
                continue
            if uuid in blocks:
                raise ValueError(f"Duplicate block uuid {uuid!r}")
            blocks[uuid] = TatDqaBlock(
                uuid=uuid,
                page_index=page_index,
                order=int(block.get("order") or 0),
                text=str(block.get("text") or ""),
                bbox=bbox_tuple(block.get("bbox")),
                words=tuple(build_words(block)),
            )
    return blocks, pages


def build_words(block: Mapping[str, Any]) -> list[TatDqaWord]:
    words_raw = block.get("words") or {}
    word_list = list(words_raw.get("word_list") or [])
    bbox_list = list(words_raw.get("bbox_list") or [])
    text = str(block.get("text") or "")
    words: list[TatDqaWord] = []
    cursor = 0

    for index, word in enumerate(word_list):
        if index >= len(bbox_list):
            break
        token = str(word)
        start = text.find(token, cursor)
        if start < 0:
            start = cursor
        end = start + len(token)
        words.append(
            TatDqaWord(
                index=index,
                text=token,
                bbox=bbox_tuple(bbox_list[index]),
                char_start=start,
                char_end=end,
            )
        )
        cursor = end
    return words


def words_for_char_span(block: TatDqaBlock, start: int, end: int) -> tuple[TatDqaWord, ...]:
    if start < 0 or end < start or start >= len(block.text):
        return ()
    inclusive_end = min(len(block.text), end + 1)
    exclusive_end = min(len(block.text), max(end, start + 1))
    candidates = [
        tuple(word for word in block.words if word.char_end > start and word.char_start < boundary)
        for boundary in (exclusive_end, inclusive_end)
    ]
    non_empty = [candidate for candidate in candidates if candidate]
    if not non_empty:
        return ()
    return min(non_empty, key=len)


def words_for_word_span(block: TatDqaBlock, start: int, end: int) -> tuple[TatDqaWord, ...]:
    if start < 0 or end <= start or end > len(block.words):
        return ()
    return block.words[start:end]


def evidence_text_for_words(words: Sequence[TatDqaWord]) -> str:
    return compact_text(" ".join(word.text for word in words))


def locate_evidence(
    question: Mapping[str, Any],
    blocks_by_uuid: Mapping[str, TatDqaBlock],
) -> tuple[TatDqaEvidence | None, str]:
    mappings = question.get("block_mapping") or []
    if len(mappings) != 1 or not isinstance(mappings[0], Mapping):
        return None, "mapping_not_unique"

    mapping_items = list(mappings[0].items())
    if len(mapping_items) != 1:
        return None, "mapping_block_not_unique"
    block_uuid, raw_span = mapping_items[0]
    block = blocks_by_uuid.get(str(block_uuid))
    if block is None:
        return None, "mapped_block_missing"
    if not isinstance(raw_span, list | tuple) or len(raw_span) != 2:
        return None, "mapped_span_invalid"

    start, end = int(raw_span[0]), int(raw_span[1])
    answer_text = scalar_answer_text(question.get("answer"))
    facts = [compact_text(fact) for fact in question.get("facts") or [] if compact_text(fact)]
    expected_texts = {normalize_number_text(answer_text)}
    expected_texts.update(normalize_number_text(fact) for fact in facts)

    char_words = words_for_char_span(block, start, end)
    char_text = evidence_text_for_words(char_words) if char_words else compact_text(block.text[start : end + 1])
    if char_text and normalize_number_text(char_text) in expected_texts:
        return build_evidence(block, (start, end), "char", char_text, char_words), "accepted"

    word_words = words_for_word_span(block, start, end)
    word_text = evidence_text_for_words(word_words)
    if word_text and normalize_number_text(word_text) in expected_texts:
        return build_evidence(block, (start, end), "word", word_text, word_words), "accepted"

    if char_words:
        return build_evidence(block, (start, end), "char_fuzzy", char_text, char_words), "accepted_fuzzy_text"
    return (
        TatDqaEvidence(
            block=block,
            span=(start, end),
            span_mode="block",
            text=compact_text(block.text),
            bbox=block.bbox,
            word_indices=(),
        ),
        "accepted_block_only",
    )


def build_evidence(
    block: TatDqaBlock,
    span: tuple[int, int],
    mode: str,
    text: str,
    words: Sequence[TatDqaWord],
) -> TatDqaEvidence:
    bbox = union_bbox(word.bbox for word in words) if words else block.bbox
    return TatDqaEvidence(
        block=block,
        span=span,
        span_mode=mode,
        text=compact_text(text),
        bbox=bbox,
        word_indices=tuple(word.index for word in words),
    )


def cell_to_json(evidence: TatDqaEvidence) -> dict[str, Any]:
    return {
        "cell_id": evidence.cell_id,
        "row_idx": evidence.block.page_index,
        "col_idx": evidence.block.order,
        "text": evidence.text,
        "bbox": list(evidence.bbox),
        "label": "structure_region" if evidence.span_mode == "block" else "word_span_region",
        "page_index": evidence.block.page_index,
        "block_uuid": evidence.block.uuid,
        "block_order": evidence.block.order,
        "span": list(evidence.span),
        "span_mode": evidence.span_mode,
        "word_indices": list(evidence.word_indices),
    }


def evidence_packet_to_json(evidence: TatDqaEvidence) -> dict[str, Any]:
    cell = cell_to_json(evidence)
    return {
        "value_cell": cell,
        "row_headers": [],
        "column_headers": [],
        "unit_cells": [],
        "spanning_headers": [],
        "all_cells": [cell],
    }


def plan_to_json(answer: str, evidence: TatDqaEvidence, notes: Sequence[str]) -> dict[str, Any]:
    return {
        "variant": "clean",
        "intervention_type": "none",
        "answer": answer,
        "evidence_cell": cell_to_json(evidence),
        "patches": [],
        "validation_status": "passed",
        "validation_notes": list(notes),
    }


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_")[:120] or "unknown"


def prepare_output_dir(output_root: Path, config_path: Path) -> tuple[Path, Path]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use a fresh run directory for reproducibility."
        )
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "config.yaml")
    return output_root, images_dir


def find_uid_file(root: Path, uid: str, suffix: str) -> Path | None:
    root = root.expanduser()
    direct = root / f"{uid}{suffix}"
    if direct.exists():
        return direct
    matches = sorted(root.rglob(f"{uid}{suffix}"))
    return matches[0] if matches else None


def page_pixel_size(page: Mapping[str, Any]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox_tuple(page.get("bbox"))
    return x1 - x0, y1 - y0


def render_pdf_page_to_png(
    pdf_path: Path,
    page_index: int,
    target_size: tuple[int, int],
    output_path: Path,
) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required to render TAT-DQA PDF pages; install project dependencies.") from exc

    with fitz.open(str(pdf_path)) as pdf:
        if page_index < 0 or page_index >= len(pdf):
            raise ValueError(f"Page index {page_index} is outside PDF page count {len(pdf)} for {pdf_path}")
        page = pdf[page_index]
        width, height = target_size
        matrix = fitz.Matrix(width / page.rect.width, height / page.rect.height)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(str(output_path))


def write_jsonl_record(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def stats_key(reason: str) -> str:
    return f"rejected_{reason}"
