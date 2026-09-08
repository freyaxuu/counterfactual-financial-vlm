"""Pair-outcome metrics for the frozen `company_confusion_v1` benchmark
(200 period/basis confusion pairs -- `docs/company-benchmark-diagnosis-report.md`
section 16).

Analogous to `financial_vlm.evaluation.tatqa_natural_pairs_eval`'s
`PairOutcome`/`evaluate_pair`/`micro_pair_accuracy`/
`competing_fact_capture_rate`, reimplemented against this benchmark's own
record schema (`pair_id`/`document_id`/`page_a`/`gold_answer_a`/
`gold_answer_b`) rather than TAT-QA's (`table_id`/`target_cell_id`) -- the
two aren't interchangeable, so this is a parallel module, not a shared one,
matching this project's existing pattern of one eval module per dataset
(synfintabs, tatqa, now company).

`PairOutcome` here is structurally compatible with TAT-QA's (same
`correct_a`/`correct_b`/`document_id` fields), so the frozen Company
protocol reuses `tatqa_natural_pairs_eval.semantic_switch_success_rate` and
`.clustered_bootstrap_delta`/`clustered_bootstrap_delta_generic` directly
(see `scripts/aggregate_company_confusion_results.py`) -- the exact same
functions, not a reimplementation, per the frozen-protocol instruction to
match TAT-QA's evaluation code exactly.

Pure functions over plain dicts/dataclasses -- no private-package
dependency, testable with synthetic fixtures. Never touches raw prediction
or gold text beyond the numeric-correctness check itself; every public
function here returns only booleans/rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from financial_vlm.evaluation.pilot_accuracy import is_numeric_correct, normalize_numeric_answer


@dataclass(frozen=True)
class QuestionPrediction:
    prediction: str | None
    numeric_correct: bool


@dataclass(frozen=True)
class ScoredQuestion:
    """A single unique question's per-checkpoint outcome, for Individual
    Accuracy bootstrap deltas (clustered by document, matching
    `tatqa_natural_pairs_eval.ScoredQuestion`)."""

    document_id: str
    correct: bool


@dataclass(frozen=True)
class PairOutcome:
    pair_id: str
    pair_type: str  # "period" | "basis"
    document_id: str
    correct_a: bool
    correct_b: bool
    pair_correct: bool
    a_captured_by_b: bool  # side A's prediction numerically matches side B's gold instead
    b_captured_by_a: bool
    collapsed: bool  # both sides predicted the same (wrong-for-at-least-one) value


def is_captured(prediction: str | None, other_gold: str) -> bool:
    """True if `prediction` numerically matches the OTHER side's gold answer."""

    return prediction is not None and is_numeric_correct(prediction, other_gold)


def is_collapsed(pred_a: str | None, pred_b: str | None) -> bool:
    if pred_a is None or pred_b is None:
        return False
    norm_a = normalize_numeric_answer(pred_a)
    norm_b = normalize_numeric_answer(pred_b)
    return norm_a is not None and norm_a == norm_b


def evaluate_pair(
    pair: Mapping[str, Any],
    predictions_by_key: Mapping[str, QuestionPrediction],
) -> PairOutcome | None:
    """`pair` is one row from `company_confusion_v1.jsonl`. Returns `None` if
    either side's question has no cached prediction (should not happen once
    inference has run over the full unique-question index, but guarded
    rather than assumed)."""

    key_a = f"{pair['pair_id']}_A"
    key_b = f"{pair['pair_id']}_B"
    pred_a = predictions_by_key.get(key_a)
    pred_b = predictions_by_key.get(key_b)
    if pred_a is None or pred_b is None:
        return None

    return PairOutcome(
        pair_id=pair["pair_id"],
        pair_type=pair["pair_type"],
        document_id=pair["document_id"],
        correct_a=pred_a.numeric_correct,
        correct_b=pred_b.numeric_correct,
        pair_correct=pred_a.numeric_correct and pred_b.numeric_correct,
        a_captured_by_b=is_captured(pred_a.prediction, pair["gold_answer_b"]),
        b_captured_by_a=is_captured(pred_b.prediction, pair["gold_answer_a"]),
        collapsed=is_collapsed(pred_a.prediction, pred_b.prediction),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def question_accuracy(items: Sequence[ScoredQuestion]) -> float | None:
    return _rate(sum(1 for i in items if i.correct), len(items))


def micro_pair_accuracy(outcomes: Sequence[PairOutcome]) -> float | None:
    return _rate(sum(1 for o in outcomes if o.pair_correct), len(outcomes))


def competing_fact_capture_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    """Among all individual answers (both sides of every pair, regardless of
    correctness), what fraction are specifically the OTHER side's value --
    the confusion-pair-specific failure mode this benchmark exists to
    measure, not just "wrong in general"."""

    if not outcomes:
        return None
    total = sum((1 if o.a_captured_by_b else 0) + (1 if o.b_captured_by_a else 0) for o in outcomes)
    return total / (2 * len(outcomes))


def conditional_capture_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    """Among individual WRONG answers only (each side counted separately),
    what fraction were specifically the paired competing fact? `None`
    (report as NA) if there are zero wrong answers."""

    wrong = 0
    wrong_and_captured = 0
    for o in outcomes:
        if not o.correct_a:
            wrong += 1
            if o.a_captured_by_b:
                wrong_and_captured += 1
        if not o.correct_b:
            wrong += 1
            if o.b_captured_by_a:
                wrong_and_captured += 1
    return _rate(wrong_and_captured, wrong)


def collapsed_pair_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    return _rate(sum(1 for o in outcomes if o.collapsed), len(outcomes))
