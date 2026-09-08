"""Prompt and record helpers for answer-only financial QA SFT."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


STANDARD_QA_INSTRUCTION = (
    "Read the financial table image and answer the question. "
    "Return only the answer value, without explanation or units unless the value itself includes a percent sign."
)


def standard_qa_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")
    return f"{STANDARD_QA_INSTRUCTION}\nQuestion: {question}"


def answer_target(record: Mapping[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, Mapping):
        raw = answer.get("raw")
    else:
        raw = answer
    target = "" if raw is None else str(raw).strip()
    if not target:
        raise ValueError("record answer target must be non-empty")
    return target


def record_variant(record: Mapping[str, Any]) -> str:
    return str(record.get("variant") or "clean")


def filter_records_by_variant(
    records: Sequence[Mapping[str, Any]],
    allowed_variants: Sequence[str],
) -> list[dict[str, Any]]:
    allowed = set(allowed_variants)
    return [dict(record) for record in records if record_variant(record) in allowed]


def normalize_interval_strategy(value: Any, *, default: str = "no") -> str:
    """Return a Transformers interval strategy string from YAML-loaded values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return "steps" if value else "no"
    normalized = str(value).strip().lower()
    if normalized in {"no", "steps", "epoch"}:
        return normalized
    raise ValueError(f"Invalid interval strategy {value!r}; expected one of: no, steps, epoch")
