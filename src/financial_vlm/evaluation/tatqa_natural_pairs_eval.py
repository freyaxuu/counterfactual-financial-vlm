"""Pair-level evaluation metrics for the frozen TAT-QA Natural Pair benchmark.

Reuses the project's existing numeric answer normalizer
(``financial_vlm.evaluation.pilot_accuracy.is_numeric_correct`` /
``normalize_numeric_answer``) for every comparison here -- both "is this
prediction correct" and "does this prediction match the OTHER side's gold
answer" (competing-fact capture) -- so results stay comparable to the
existing clean/TVFR/HFR/ISR/CGS numbers, which are reported on
``clean_accuracy_numeric``.

This module only computes metrics from already-generated model
predictions. It does not run a model and does not alter benchmark
construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import random
from typing import Any, Callable, Collection, Mapping, Sequence

from financial_vlm.evaluation.pilot_accuracy import is_numeric_correct, normalize_numeric_answer

PAIR_SUBSETS = ("human_authored", "period_derived", "metric_candidate")


def question_key(document_id: str, question_text: str, target_cell_id: str | None) -> str:
    """Stable dedup key: (document_id, question_text, target_cell_id).

    ``table_id`` is not included separately -- in this benchmark
    ``document_id == table_id`` always (enforced by
    ``tatqa_natural_pairs.validate_natural_pairs``).
    """

    raw = f"{document_id}\x1f{question_text}\x1f{target_cell_id or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class PairSide:
    question_key: str
    document_id: str
    table_id: str
    question: str
    answer_raw: str
    target_cell_id: str | None


def pair_sides(record: Mapping[str, Any]) -> tuple[PairSide, PairSide]:
    def side(suffix: str) -> PairSide:
        evidence = record.get(f"evidence_{suffix}") or {}
        question = record[f"question_{suffix}"]
        document_id = record["document_id"]
        target_cell_id = evidence.get("cell_id")
        return PairSide(
            question_key=question_key(document_id, question, target_cell_id),
            document_id=document_id,
            table_id=record["table_id"],
            question=question,
            answer_raw=record[f"answer_{suffix}"],
            target_cell_id=target_cell_id,
        )

    return side("a"), side("b")


@dataclass(frozen=True)
class QuestionPrediction:
    prediction: str | None
    numeric_correct: bool


def is_captured(prediction: str | None, other_gold: str) -> bool:
    """True if ``prediction`` numerically matches the OTHER side's gold answer."""

    return prediction is not None and is_numeric_correct(prediction, other_gold)


def is_collapsed(pred_a: str | None, pred_b: str | None) -> bool:
    if pred_a is None or pred_b is None:
        return False
    norm_a = normalize_numeric_answer(pred_a)
    norm_b = normalize_numeric_answer(pred_b)
    return norm_a is not None and norm_a == norm_b


# --- Error-mode taxonomy (root-cause diagnosis, docs/tatqa-natural-pairs-
# evaluation-report.md) ------------------------------------------------
#
# Every WRONG prediction on the Natural Pair benchmark falls into exactly
# one of these buckets, checked in this priority order:
#   1. captured_by_pair -- the prediction exactly equals the OTHER side's
#      gold answer (the specific failure mode the header/address-swap
#      counterfactual intervention targets).
#   2. scale_shift -- the prediction is the right cell's value off by a
#      power of ten (a dropped/extra decimal digit), unrelated to which
#      cell was read.
#   3. other_cell_in_table -- the prediction matches some OTHER real value
#      in the same table (broader address confusion than #1 -- validated
#      against a permutation-test null baseline, see
#      permutation_null_other_cell_match below, so this bucket is only
#      populated when a table_values index is supplied).
#   4. other -- everything else (wrong row, hallucination, arithmetic
#      mistake, ...).

ERROR_MODES = ("captured_by_pair", "scale_shift", "other_cell_in_table", "other")
_SCALE_SHIFT_RATIOS = (Decimal(10), Decimal(100), Decimal(1000), Decimal("0.1"), Decimal("0.01"), Decimal("0.001"))
_SCALE_SHIFT_TOLERANCE = Decimal("0.001")


def is_scale_shift_error(prediction: str | None, target: str) -> bool:
    """True if ``prediction`` is numeric but wrong, and equals ``target``
    scaled by a power of ten -- e.g. ``"24.6"`` (target) vs ``"246"``
    (prediction): the right cell, garbled magnitude."""

    if prediction is None:
        return False
    pred_norm = normalize_numeric_answer(prediction)
    target_norm = normalize_numeric_answer(target)
    if pred_norm is None or target_norm is None:
        return False
    try:
        pred_value, target_value = Decimal(pred_norm), Decimal(target_norm)
    except InvalidOperation:
        return False
    if target_value == 0:
        return False
    ratio = pred_value / target_value
    return any(abs(ratio - k) < _SCALE_SHIFT_TOLERANCE for k in _SCALE_SHIFT_RATIOS)


def classify_wrong_prediction(
    prediction: str | None,
    target: str,
    paired_gold: str,
    table_values: Collection[str] | None = None,
) -> str:
    """Categorize an already-known-WRONG prediction into one of
    ``ERROR_MODES``. Caller is responsible for only calling this on
    predictions that are not numerically correct against ``target``.

    ``table_values`` is optional (defaults to skipping the
    ``other_cell_in_table`` bucket, folding it into ``other`` -- the
    original 3-bucket behavior) since not every caller has a per-table
    value index handy."""

    if is_captured(prediction, paired_gold):
        return "captured_by_pair"
    if is_scale_shift_error(prediction, target):
        return "scale_shift"
    if table_values is not None and prediction is not None:
        pred_norm = normalize_numeric_answer(prediction)
        if pred_norm is not None and pred_norm in table_values:
            return "other_cell_in_table"
    return "other"


def error_taxonomy(
    records: Sequence[Mapping[str, Any]],
    predictions_by_key: Mapping[str, QuestionPrediction],
    table_values_by_table: Mapping[str, set[str]] | None = None,
) -> dict[str, int]:
    """Counts of ``ERROR_MODES`` across every side of every pair whose both
    sides have a cached prediction (same coverage as ``evaluate_pair``).
    Correct sides are not counted. Pass ``table_values_by_table`` (e.g.
    from ``build_table_value_index``) to populate the validated
    ``other_cell_in_table`` bucket; omit it to fall back to the original
    3-bucket taxonomy (that mass folds into ``other``)."""

    counts: dict[str, int] = {mode: 0 for mode in ERROR_MODES}
    for record in records:
        side_a, side_b = pair_sides(record)
        pred_a = predictions_by_key.get(side_a.question_key)
        pred_b = predictions_by_key.get(side_b.question_key)
        if pred_a is None or pred_b is None:
            continue
        table_values = table_values_by_table.get(record["table_id"]) if table_values_by_table is not None else None
        if not pred_a.numeric_correct:
            counts[classify_wrong_prediction(pred_a.prediction, side_a.answer_raw, side_b.answer_raw, table_values)] += 1
        if not pred_b.numeric_correct:
            counts[classify_wrong_prediction(pred_b.prediction, side_b.answer_raw, side_a.answer_raw, table_values)] += 1
    return counts


@dataclass(frozen=True)
class PairOutcome:
    pair_id: str
    table_id: str
    document_id: str
    correct_a: bool
    correct_b: bool
    pair_correct: bool
    a_captured_by_b: bool
    b_captured_by_a: bool
    collapsed: bool


def evaluate_pair(
    record: Mapping[str, Any],
    predictions_by_key: Mapping[str, QuestionPrediction],
) -> PairOutcome | None:
    """Returns ``None`` if either side's question has no cached prediction
    (should not happen once inference has run over the full unique-question
    index, but guarded rather than assumed)."""

    side_a, side_b = pair_sides(record)
    pred_a = predictions_by_key.get(side_a.question_key)
    pred_b = predictions_by_key.get(side_b.question_key)
    if pred_a is None or pred_b is None:
        return None

    correct_a = pred_a.numeric_correct
    correct_b = pred_b.numeric_correct
    return PairOutcome(
        pair_id=record["pair_id"],
        table_id=record["table_id"],
        document_id=record["document_id"],
        correct_a=correct_a,
        correct_b=correct_b,
        pair_correct=correct_a and correct_b,
        a_captured_by_b=is_captured(pred_a.prediction, side_b.answer_raw),
        b_captured_by_a=is_captured(pred_b.prediction, side_a.answer_raw),
        collapsed=is_collapsed(pred_a.prediction, pred_b.prediction),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def individual_accuracy(predictions: Sequence[QuestionPrediction]) -> float | None:
    return _rate(sum(1 for p in predictions if p.numeric_correct), len(predictions))


def micro_pair_accuracy(outcomes: Sequence[PairOutcome]) -> float | None:
    return _rate(sum(1 for o in outcomes if o.pair_correct), len(outcomes))


def table_macro_metric(outcomes: Sequence[PairOutcome], metric_fn: Callable[[Sequence[PairOutcome]], float | None]) -> float | None:
    by_table: dict[str, list[PairOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_table[outcome.table_id].append(outcome)
    per_table = [metric_fn(group) for group in by_table.values()]
    per_table = [v for v in per_table if v is not None]
    return sum(per_table) / len(per_table) if per_table else None


def competing_fact_capture_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    if not outcomes:
        return None
    total = sum((1 if o.a_captured_by_b else 0) + (1 if o.b_captured_by_a else 0) for o in outcomes)
    return total / (2 * len(outcomes))


def collapsed_pair_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    return _rate(sum(1 for o in outcomes if o.collapsed), len(outcomes))


def semantic_switch_success_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    """Treats each pair as two directed transitions (A->B and B->A). A
    transition is *eligible* if its source side is correct, and
    *successful* if the paired target side is also correct. Returns
    successful / eligible across all directed transitions in ``outcomes``
    -- i.e. P(target correct | source correct), pooling both directions.

    A pair with both sides correct contributes 2 eligible+successful
    transitions; a pair with exactly one side correct contributes 1
    eligible, unsuccessful transition (the correct side is the source, the
    wrong side is the target); a pair with both sides wrong contributes 0
    eligible transitions (neither side can be a source)."""

    eligible = 0
    successful = 0
    for o in outcomes:
        if o.correct_a:
            eligible += 1
            if o.correct_b:
                successful += 1
        if o.correct_b:
            eligible += 1
            if o.correct_a:
                successful += 1
    return _rate(successful, eligible)


def conditional_capture_rate(outcomes: Sequence[PairOutcome]) -> float | None:
    """Among individual wrong answers (each side counted separately), what
    fraction were specifically the paired competing fact? Returns ``None``
    (report as NA) if there are zero wrong answers in the subset."""

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


@dataclass(frozen=True)
class RescueRegressionCounts:
    both_correct: int
    baseline_correct_other_wrong: int
    baseline_wrong_other_correct: int
    both_wrong: int

    @property
    def total(self) -> int:
        return self.both_correct + self.baseline_correct_other_wrong + self.baseline_wrong_other_correct + self.both_wrong

    @property
    def rescue_rate(self) -> float | None:
        """P(other correct | baseline wrong)."""

        denom = self.baseline_wrong_other_correct + self.both_wrong
        return _rate(self.baseline_wrong_other_correct, denom)

    @property
    def regression_rate(self) -> float | None:
        """P(other wrong | baseline correct)."""

        denom = self.both_correct + self.baseline_correct_other_wrong
        return _rate(self.baseline_correct_other_wrong, denom)


def rescue_regression(
    baseline_outcomes: Mapping[str, PairOutcome],
    other_outcomes: Mapping[str, PairOutcome],
) -> RescueRegressionCounts:
    """Both mappings keyed by pair_id; only pairs present in both are used
    (matched pair_id set, e.g. same benchmark, two different checkpoints)."""

    both_correct = baseline_correct_other_wrong = baseline_wrong_other_correct = both_wrong = 0
    for pair_id, baseline in baseline_outcomes.items():
        other = other_outcomes.get(pair_id)
        if other is None:
            continue
        if baseline.pair_correct and other.pair_correct:
            both_correct += 1
        elif baseline.pair_correct and not other.pair_correct:
            baseline_correct_other_wrong += 1
        elif not baseline.pair_correct and other.pair_correct:
            baseline_wrong_other_correct += 1
        else:
            both_wrong += 1
    return RescueRegressionCounts(both_correct, baseline_correct_other_wrong, baseline_wrong_other_correct, both_wrong)


def clustered_bootstrap_delta_generic(
    items_a: Sequence[Any],
    items_b: Sequence[Any],
    document_id_fn: Callable[[Any], str],
    metric_fn: Callable[[Sequence[Any]], float | None],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired clustered bootstrap for ``metric(b) - metric(a)``, clustered by
    whatever ``document_id_fn`` returns for each item. Both item sequences
    must cover the same underlying document set (matched benchmark, two
    different model predictions) -- works for ``PairOutcome`` sequences
    (pair-level metrics) or any other per-item sequence (e.g. per-question
    correctness, for Individual Accuracy deltas)."""

    by_doc_a: dict[str, list[Any]] = defaultdict(list)
    by_doc_b: dict[str, list[Any]] = defaultdict(list)
    for item in items_a:
        by_doc_a[document_id_fn(item)].append(item)
    for item in items_b:
        by_doc_b[document_id_fn(item)].append(item)
    documents = sorted(set(by_doc_a) & set(by_doc_b))
    if not documents:
        return {"deltas": [], "ci_low": None, "ci_high": None, "point_estimate": None}

    rng = random.Random(seed)
    point_a = metric_fn(items_a)
    point_b = metric_fn(items_b)
    point_estimate = (point_b - point_a) if (point_a is not None and point_b is not None) else None

    deltas: list[float] = []
    for _ in range(n_resamples):
        sampled_docs = [rng.choice(documents) for _ in documents]
        sample_a: list[Any] = []
        sample_b: list[Any] = []
        for doc in sampled_docs:
            sample_a.extend(by_doc_a[doc])
            sample_b.extend(by_doc_b[doc])
        metric_a = metric_fn(sample_a)
        metric_b = metric_fn(sample_b)
        if metric_a is None or metric_b is None:
            continue
        deltas.append(metric_b - metric_a)

    deltas.sort()
    if not deltas:
        return {"deltas": [], "ci_low": None, "ci_high": None, "point_estimate": point_estimate}
    lo_idx = int(0.025 * len(deltas))
    hi_idx = min(int(0.975 * len(deltas)), len(deltas) - 1)
    return {
        "n_resamples_used": len(deltas),
        "ci_low": deltas[lo_idx],
        "ci_high": deltas[hi_idx],
        "point_estimate": point_estimate,
    }


def clustered_bootstrap_delta(
    outcomes_a: Sequence[PairOutcome],
    outcomes_b: Sequence[PairOutcome],
    metric_fn: Callable[[Sequence[PairOutcome]], float | None],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """``PairOutcome``-specific convenience wrapper around
    ``clustered_bootstrap_delta_generic`` (clusters by ``outcome.document_id``)."""

    return clustered_bootstrap_delta_generic(
        outcomes_a, outcomes_b, lambda o: o.document_id, metric_fn, n_resamples=n_resamples, seed=seed
    )


@dataclass(frozen=True)
class ScoredQuestion:
    """A single unique question's per-checkpoint outcome, for Individual
    Accuracy bootstrap deltas (clustered by document, like the pair-level
    metrics, since a document/table can contribute several unique
    questions)."""

    document_id: str
    correct: bool


def question_accuracy(items: Sequence[ScoredQuestion]) -> float | None:
    return _rate(sum(1 for i in items if i.correct), len(items))


# --- Broader within-table confusion, with a permutation-test null baseline
# ------------------------------------------------------------------------
#
# `classify_wrong_prediction`'s "captured_by_pair" bucket only checks a
# wrong prediction against the ONE specific paired competing fact. A wrong
# prediction can also match some OTHER real cell value in the same table --
# a broader, and possibly more realistic, signal of "the model read the
# wrong address" that the narrow pair-only check misses entirely. But
# financial tables often reuse round or similar-magnitude numbers, so
# "matches some other cell" can happen by pure coincidence even for a
# prediction that has nothing to do with that table. A permutation test
# (shuffle which table each wrong prediction is scored against, keeping
# the real predictions and real per-table value sets fixed) estimates how
# often that coincidence rate would occur under a "no genuine within-table
# confusion" null, so the real observed rate can be judged against it
# rather than taken at face value.


@dataclass(frozen=True)
class UnexplainedWrongItem:
    """A wrong prediction that is neither captured by the paired competing
    fact nor a scale/decimal-point slip -- the residual "other" bucket,
    kept as (table_id, normalized prediction) for the broader-confusion
    check below."""

    table_id: str
    prediction_norm: str | None


def build_table_value_index(records: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """table_id -> the set of normalized gold values that appear anywhere
    among these pair records for that table (both sides of every pair)."""

    index: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for suffix in ("a", "b"):
            norm = normalize_numeric_answer(str(record[f"answer_{suffix}"]))
            if norm is not None:
                index[record["table_id"]].add(norm)
    return index


def unexplained_wrong_items(
    records: Sequence[Mapping[str, Any]],
    predictions_by_key: Mapping[str, QuestionPrediction],
) -> list[UnexplainedWrongItem]:
    """Every wrong-and-not-yet-explained side of every pair (mirrors
    ``error_taxonomy``'s coverage and priority order, stopping short of the
    final "other" classification so the broader within-table check below
    can be applied to exactly that residual set)."""

    items: list[UnexplainedWrongItem] = []
    for record in records:
        side_a, side_b = pair_sides(record)
        pred_a = predictions_by_key.get(side_a.question_key)
        pred_b = predictions_by_key.get(side_b.question_key)
        if pred_a is None or pred_b is None:
            continue
        for side, pred, other_gold in ((side_a, pred_a, side_b.answer_raw), (side_b, pred_b, side_a.answer_raw)):
            if pred.numeric_correct:
                continue
            if is_captured(pred.prediction, other_gold):
                continue
            if is_scale_shift_error(pred.prediction, side.answer_raw):
                continue
            items.append(
                UnexplainedWrongItem(
                    table_id=record["table_id"],
                    prediction_norm=normalize_numeric_answer(pred.prediction) if pred.prediction is not None else None,
                )
            )
    return items


def other_cell_match_rate(
    items: Sequence[UnexplainedWrongItem],
    table_values: Mapping[str, set[str]],
) -> float | None:
    """Fraction of ``items`` whose normalized prediction matches SOME real
    value in its own table (beyond the specific paired competing fact,
    already excluded by construction of ``items``)."""

    if not items:
        return None
    matches = sum(1 for item in items if item.prediction_norm is not None and item.prediction_norm in table_values.get(item.table_id, ()))
    return matches / len(items)


def permutation_null_other_cell_match(
    items: Sequence[UnexplainedWrongItem],
    table_values: Mapping[str, set[str]],
    *,
    n_permutations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Permutation-test null for ``other_cell_match_rate``: shuffle which
    table each item's prediction is scored against (keeping the multiset of
    predictions and the real per-table value sets fixed), estimating how
    often the observed match rate would occur by coincidence alone.

    Returns ``observed_rate``, the null distribution's mean/p05/p95, and a
    one-sided p-value (fraction of null resamples at or above the observed
    rate) -- a small p-value means the observed rate is unlikely to be
    explained by generic financial-number collisions.
    """

    observed = other_cell_match_rate(items, table_values)
    if observed is None:
        return {"observed_rate": None, "null_mean": None, "null_p05": None, "null_p95": None, "p_value_one_sided": None, "n_permutations": 0}

    rng = random.Random(seed)
    table_ids = [item.table_id for item in items]
    preds = [item.prediction_norm for item in items]

    null_rates: list[float] = []
    for _ in range(n_permutations):
        shuffled_tables = table_ids[:]
        rng.shuffle(shuffled_tables)
        shuffled_items = [UnexplainedWrongItem(t, p) for t, p in zip(shuffled_tables, preds)]
        rate = other_cell_match_rate(shuffled_items, table_values)
        if rate is not None:
            null_rates.append(rate)

    null_rates.sort()
    if not null_rates:
        return {"observed_rate": observed, "null_mean": None, "null_p05": None, "null_p95": None, "p_value_one_sided": None, "n_permutations": 0}

    p_value = sum(1 for r in null_rates if r >= observed) / len(null_rates)
    return {
        "observed_rate": observed,
        "null_mean": sum(null_rates) / len(null_rates),
        "null_p05": null_rates[int(0.05 * len(null_rates))],
        "null_p95": null_rates[min(int(0.95 * len(null_rates)), len(null_rates) - 1)],
        "p_value_one_sided": p_value,
        "n_permutations": len(null_rates),
    }
