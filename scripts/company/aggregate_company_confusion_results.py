#!/usr/bin/env python3
"""Aggregate cached predictions from the 10 models (9 trained checkpoints +
frozen base) into the FINAL, frozen `company_confusion_v1` evaluation
report.

Per explicit instruction (chat, 2026-08-13): mirrors the TAT-QA natural-pair
final-results protocol (`docs/tatqa-final-results-report.md`,
`scripts/evaluate_period_primary_reliability_diagnostics.py`) exactly --
reuses `financial_vlm.evaluation.tatqa_natural_pairs_eval.semantic_switch_success_rate`
and `.clustered_bootstrap_delta` DIRECTLY (not a reimplementation; see
`tests/unit/test_company_confusion_eval.py::TatqaEvalFunctionReuseTest` for
confirmation this works by duck-typing against this module's own
`PairOutcome`).

Headline metrics: Individual Accuracy, PairAcc (primary), Semantic Switch
Success, Swap-capture rate (secondary diagnostic). Collapsed Rate is
appendix-only. No HFR/CGS are computed on Company data (never applicable --
those require controlled synthetic interventions, which this benchmark,
built entirely from naturally occurring facts, does not have and must not
introduce -- see AGENTS.md's company-data rules).

Primary statistical comparison: CF augmentation - Compute-matched (same
two-image-per-step training regime, differing only in whether the second
image is a counterfactual -- the more apples-to-apples comparison).
Secondary: CF augmentation - Standard. Both via 10,000-resample paired,
document-clustered bootstrap, 95% CI, reported in percentage points.

Reads only cached prediction files (predictions.jsonl per checkpoint,
written by evaluate_qwen3_vl_natural_pairs.py) and the frozen benchmark --
runs no model, and never re-filters or otherwise modifies the benchmark
based on what these predictions show. Per AGENTS.md's company-data rules,
only aggregate metrics are written/printed -- no raw prediction or gold
text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.evaluation.company_confusion_eval import (  # noqa: E402
    PairOutcome,
    QuestionPrediction,
    ScoredQuestion,
    collapsed_pair_rate,
    competing_fact_capture_rate,
    evaluate_pair,
    micro_pair_accuracy,
    question_accuracy,
)
from financial_vlm.evaluation.tatqa_natural_pairs_eval import (  # noqa: E402
    clustered_bootstrap_delta,
    clustered_bootstrap_delta_generic,
    semantic_switch_success_rate,
)

METHOD_SEEDS: dict[str, list[str]] = {
    "base": ["zero_shot"],
    "standard_lora": ["20260804", "20260806", "20260808"],
    "compute_matched_clean": ["20260804", "20260806", "20260808"],
    "cf_augmentation": ["20260804", "20260806", "20260808"],
}
TRAINED_METHODS = ("standard_lora", "compute_matched_clean", "cf_augmentation")
ALL_METHODS = ("base",) + TRAINED_METHODS
METHOD_LABELS = {
    "base": "Base (zero-shot)",
    "standard_lora": "Standard",
    "compute_matched_clean": "Compute-matched",
    "cf_augmentation": "CF augmentation",
}
# name -> (fn over PairOutcome sequences, headline vs appendix)
PAIR_METRICS = {
    "pair_accuracy": micro_pair_accuracy,
    "semantic_switch_success_rate": semantic_switch_success_rate,
    "swap_capture_rate": competing_fact_capture_rate,
    "collapsed_pair_rate": collapsed_pair_rate,  # appendix only
}
# Primary: CF - Compute-matched (same training regime, differ only in CF).
# Secondary: CF - Standard.
COMPARISONS = [
    ("primary", "cf_augmentation", "compute_matched_clean"),
    ("secondary", "cf_augmentation", "standard_lora"),
]


def label_for(method: str, seed: str) -> str:
    return "base" if method == "base" else f"{method}_seed_{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-confusion-v1", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True, help="Dir containing one subdir per checkpoint label, each with predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(predictions_root: Path, label: str) -> dict[str, QuestionPrediction]:
    path = predictions_root / label / "predictions.jsonl"
    predictions: dict[str, QuestionPrediction] = {}
    for record in load_jsonl(path):
        predictions[record["group_id"]] = QuestionPrediction(
            prediction=record.get("prediction"),
            numeric_correct=bool(record.get("numeric_correct")),
        )
    return predictions


def mean_sd(values: list[float]) -> dict[str, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        sd = variance**0.5
    else:
        sd = 0.0
    return {"mean": mean, "sd": sd, "n": len(values)}


def fmt(stat: dict[str, float | None], digits: int = 2) -> str:
    if stat["mean"] is None:
        return "NA"
    if stat["n"] > 1 and stat["sd"] is not None:
        return f"{stat['mean'] * 100:.{digits}f}% ± {stat['sd'] * 100:.{digits}f}"
    return f"{stat['mean'] * 100:.{digits}f}%"


def fmt_delta(b: dict[str, Any]) -> str:
    if b["point_estimate"] is None:
        return "NA"
    return f"{b['point_estimate'] * 100:+.2f} [{b['ci_low'] * 100:+.2f}, {b['ci_high'] * 100:+.2f}]"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_jsonl(args.company_confusion_v1.expanduser().resolve())
    if not pairs:
        raise ValueError(f"No records found in: {args.company_confusion_v1}")

    per_seed_metrics: dict[str, dict[str, Any]] = {m: {} for m in METHOD_SEEDS}
    pair_outcomes_by_label: dict[str, list[PairOutcome]] = {}
    scored_questions_by_label: dict[str, list[ScoredQuestion]] = {}
    missing_prediction_count: dict[str, int] = {}

    for method, seeds in METHOD_SEEDS.items():
        for seed in seeds:
            label = label_for(method, seed)
            preds = load_predictions(args.predictions_root, label)

            scored: list[ScoredQuestion] = []
            for pair in pairs:
                for side in ("A", "B"):
                    pred = preds.get(f"{pair['pair_id']}_{side}")
                    if pred is None:
                        continue
                    scored.append(ScoredQuestion(document_id=pair["document_id"], correct=pred.numeric_correct))
            scored_questions_by_label[label] = scored

            outcomes: list[PairOutcome] = []
            missing = 0
            for pair in pairs:
                outcome = evaluate_pair(pair, preds)
                if outcome is None:
                    missing += 1
                    continue
                outcomes.append(outcome)
            pair_outcomes_by_label[label] = outcomes
            missing_prediction_count[label] = missing

            per_seed_metrics[method][seed] = {
                "individual_accuracy": question_accuracy(scored),
                "num_unique_questions_scored": len(scored),
                "num_pairs_scored": len(outcomes),
                **{name: fn(outcomes) for name, fn in PAIR_METRICS.items()},
            }

    metric_names = ["individual_accuracy", *PAIR_METRICS]
    aggregated: dict[str, dict[str, Any]] = {}
    for method, seeds in METHOD_SEEDS.items():
        aggregated[method] = {name: mean_sd([per_seed_metrics[method][seed].get(name) for seed in seeds]) for name in metric_names}

    # --- bootstrap: primary (CF vs Compute-matched), secondary (CF vs Standard) ---
    bootstrap_results: dict[str, dict[str, Any]] = {}
    for comp_name, other_method, baseline_method in COMPARISONS:
        per_seed_boot: dict[str, Any] = {}
        for seed in METHOD_SEEDS[other_method]:
            baseline_label = label_for(baseline_method, seed)
            other_label = label_for(other_method, seed)
            baseline_outcomes = pair_outcomes_by_label[baseline_label]
            other_outcomes = pair_outcomes_by_label[other_label]
            seed_boot: dict[str, Any] = {
                name: clustered_bootstrap_delta(
                    baseline_outcomes, other_outcomes, fn, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed
                )
                for name, fn in PAIR_METRICS.items()
            }
            seed_boot["individual_accuracy"] = clustered_bootstrap_delta_generic(
                scored_questions_by_label[baseline_label],
                scored_questions_by_label[other_label],
                lambda s: s.document_id,
                question_accuracy,
                n_resamples=args.bootstrap_resamples,
                seed=args.bootstrap_seed,
            )
            per_seed_boot[seed] = seed_boot
        bootstrap_results[comp_name] = {
            "comparison": f"{other_method}_vs_{baseline_method}",
            "per_seed": per_seed_boot,
        }

    (output_dir / "per_seed_metrics.json").write_text(json.dumps(per_seed_metrics, indent=2, sort_keys=True) + "\n")
    (output_dir / "aggregated_metrics.json").write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n")
    (output_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_results, indent=2, sort_keys=True) + "\n")
    (output_dir / "missing_predictions.json").write_text(json.dumps(missing_prediction_count, indent=2, sort_keys=True) + "\n")

    report = render_markdown_report(aggregated, bootstrap_results, len(pairs), missing_prediction_count)
    report_path = output_dir / "company_confusion_v1_final_report.md"
    report_path.write_text(report)

    if any(missing_prediction_count.values()):
        print(f"WARNING missing predictions by checkpoint (should be 0): {missing_prediction_count}")
    print(f"per_seed_metrics={output_dir / 'per_seed_metrics.json'}")
    print(f"aggregated_metrics={output_dir / 'aggregated_metrics.json'}")
    print(f"bootstrap_ci={output_dir / 'bootstrap_ci.json'}")
    print(f"report={report_path}")
    return 0


def render_markdown_report(
    aggregated: dict[str, dict[str, Any]],
    bootstrap_results: dict[str, dict[str, Any]],
    n_pairs: int,
    missing_prediction_count: dict[str, int],
) -> str:
    lines = ["# company_confusion_v1 -- Final Evaluation Report", ""]
    lines.append(
        f"Frozen benchmark: {n_pairs} period/basis confusion pairs (see "
        "docs/company-benchmark-diagnosis-report.md section 16). No benchmark rules were "
        "changed after observing model results."
    )
    lines.append("")
    if any(missing_prediction_count.values()):
        lines.append(f"**Missing predictions by checkpoint (should be 0): {missing_prediction_count}**")
        lines.append("")

    lines.append("## Headline results (mean ± SD across 3 seeds; base is a single zero-shot run)")
    lines.append("")
    lines.append("| Model | Individual Accuracy | PairAcc (primary) | Semantic Switch Success | Swap-capture rate (secondary) |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in ALL_METHODS:
        m = aggregated[method]
        lines.append(
            f"| {METHOD_LABELS[method]} "
            f"| {fmt(m['individual_accuracy'])} "
            f"| {fmt(m['pair_accuracy'])} "
            f"| {fmt(m['semantic_switch_success_rate'])} "
            f"| {fmt(m['swap_capture_rate'])} |"
        )
    lines.append("")

    for comp_name, title in (("primary", "Primary: CF augmentation − Compute-matched"), ("secondary", "Secondary: CF augmentation − Standard")):
        comp = bootstrap_results[comp_name]
        lines.append(f"## {title} (paired, document-clustered bootstrap, 10,000 resamples, 95% CI)")
        lines.append("")
        lines.append("| Metric | seed 20260804 | seed 20260806 | seed 20260808 | Mean Δ (pp) |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, display in (
            ("individual_accuracy", "Individual Accuracy"),
            ("pair_accuracy", "PairAcc"),
            ("semantic_switch_success_rate", "Semantic Switch Success"),
            ("swap_capture_rate", "Swap-capture rate"),
        ):
            per_seed = comp["per_seed"]
            cells = [fmt_delta(per_seed[seed][name]) for seed in ("20260804", "20260806", "20260808")]
            points = [per_seed[seed][name]["point_estimate"] for seed in ("20260804", "20260806", "20260808")]
            points = [p for p in points if p is not None]
            mean_cell = f"{sum(points) / len(points) * 100:+.2f}" if points else "NA"
            lines.append(f"| {display} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_cell} |")
        lines.append("")

    lines.append("## Appendix: Collapsed Rate")
    lines.append("")
    lines.append("| Model | Same-answer Collapse Rate |")
    lines.append("|---|---:|")
    for method in ALL_METHODS:
        lines.append(f"| {METHOD_LABELS[method]} | {fmt(aggregated[method]['collapsed_pair_rate'])} |")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "> No human verification pass was applied to this benchmark's gold answers at any stage "
        "(see docs/company-benchmark-diagnosis-report.md sections 9/14/16) -- gold labels are gated "
        "only by automated, twice-calibrated checks. No HFR or CGS are computed here -- those require "
        "controlled synthetic interventions, which this benchmark (built entirely from naturally "
        "occurring facts) does not have and must not introduce. Per AGENTS.md, this company benchmark "
        "is a blind validation set and must not be used to select prompts, thresholds, hyperparameters, "
        "or checkpoints -- these results are reporting-only, and no benchmark rule was changed after "
        "seeing them."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
