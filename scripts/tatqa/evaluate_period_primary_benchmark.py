#!/usr/bin/env python3
"""Evaluate the 9 trained checkpoints + base on the frozen 500-pair
"primary" period benchmark (tatqa_period_primary_v1.jsonl), reusing cached
predictions -- runs no model. Reports three views: all 500 pairs,
adjacent-only (375), skip-only (125), since the skip/adjacent split is the
whole point of this frozen set.
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

from financial_vlm.data.tatqa_natural_pairs import ADJACENT_RULE, NONADJACENT_RULE  # noqa: E402
from financial_vlm.evaluation.tatqa_natural_pairs_eval import (  # noqa: E402
    ERROR_MODES,
    PairOutcome,
    QuestionPrediction,
    ScoredQuestion,
    build_table_value_index,
    clustered_bootstrap_delta,
    clustered_bootstrap_delta_generic,
    collapsed_pair_rate,
    competing_fact_capture_rate,
    conditional_capture_rate,
    error_taxonomy,
    evaluate_pair,
    micro_pair_accuracy,
    pair_sides,
    question_accuracy,
    rescue_regression,
    table_macro_metric,
)

VIEWS = ("all", "adjacent", "skip")
METHOD_SEEDS: dict[str, list[str]] = {
    "base": ["zero_shot"],
    "standard_lora": ["20260804", "20260806", "20260808"],
    "compute_matched_clean": ["20260804", "20260806", "20260808"],
    "cf_augmentation": ["20260804", "20260806", "20260808"],
}
METHOD_LABELS = {
    "base": "Base (zero-shot)",
    "standard_lora": "Standard",
    "compute_matched_clean": "Compute-matched",
    "cf_augmentation": "CF augmentation",
}
COMPARISONS = [("cf_augmentation", "standard_lora"), ("cf_augmentation", "compute_matched_clean")]


def label_for(method: str, seed: str) -> str:
    return "base" if method == "base" else f"{method}_seed_{seed}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(predictions_root: Path, label: str) -> dict[str, QuestionPrediction]:
    path = predictions_root / label / "predictions.jsonl"
    predictions: dict[str, QuestionPrediction] = {}
    for record in load_jsonl(path):
        predictions[record["group_id"]] = QuestionPrediction(
            prediction=record.get("prediction"), numeric_correct=bool(record.get("numeric_correct"))
        )
    return predictions


def records_for_view(records: list[dict[str, Any]], view: str) -> list[dict[str, Any]]:
    if view == "all":
        return records
    rule = ADJACENT_RULE if view == "adjacent" else NONADJACENT_RULE
    return [r for r in records if r["construction_rule"] == rule]


def unique_questions(records: list[dict[str, Any]]) -> dict[str, str]:
    by_key: dict[str, str] = {}
    for record in records:
        for side in pair_sides(record):
            by_key[side.question_key] = side.document_id
    return by_key


def mean_sd(values: list[float]) -> dict[str, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": len(values)}


def fmt(stat: dict[str, float | None], pct: bool = True, digits: int = 2) -> str:
    if stat["mean"] is None:
        return "NA"
    scale = 100 if pct else 1
    suffix = "%" if pct else ""
    if stat["n"] > 1 and stat["sd"] is not None:
        return f"{stat['mean']*scale:.{digits}f}{suffix} ± {stat['sd']*scale:.{digits}f}"
    return f"{stat['mean']*scale:.{digits}f}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = load_jsonl(args.primary_pairs_jsonl)
    view_records = {view: records_for_view(all_records, view) for view in VIEWS}
    view_table_values = {view: build_table_value_index(recs) for view, recs in view_records.items()}
    view_uniques = {view: unique_questions(recs) for view, recs in view_records.items()}

    per_seed_metrics: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in METHOD_SEEDS}
    pair_outcomes: dict[str, dict[str, list[PairOutcome]]] = {}
    scored_questions: dict[str, dict[str, list[ScoredQuestion]]] = {}
    missing: dict[str, int] = {}

    for method, seeds in METHOD_SEEDS.items():
        for seed in seeds:
            label = label_for(method, seed)
            preds = load_predictions(args.predictions_root, label)
            pair_outcomes[label] = {}
            scored_questions[label] = {}
            seed_metrics: dict[str, Any] = {}
            miss = 0
            for view in VIEWS:
                recs = view_records[view]
                uniques = view_uniques[view]
                scored = []
                for qkey, doc_id in uniques.items():
                    pred = preds.get(qkey)
                    if pred is None:
                        miss += 1
                        continue
                    scored.append(ScoredQuestion(document_id=doc_id, correct=pred.numeric_correct))
                scored_questions[label][view] = scored

                outcomes = [o for r in recs if (o := evaluate_pair(r, preds)) is not None]
                pair_outcomes[label][view] = outcomes

                taxonomy = error_taxonomy(recs, preds, view_table_values[view])
                total_wrong = sum(taxonomy.values())
                taxonomy_pct = {m: (taxonomy[m] / total_wrong if total_wrong else None) for m in ERROR_MODES}

                seed_metrics[view] = {
                    "individual_accuracy": question_accuracy(scored),
                    "micro_pair_accuracy": micro_pair_accuracy(outcomes),
                    "table_macro_pair_accuracy": table_macro_metric(outcomes, micro_pair_accuracy),
                    "num_pairs": len(outcomes),
                    "competing_fact_capture_rate": competing_fact_capture_rate(outcomes),
                    "conditional_capture_rate": conditional_capture_rate(outcomes),
                    "collapsed_pair_rate": collapsed_pair_rate(outcomes),
                    "error_taxonomy_pct": taxonomy_pct,
                    "error_taxonomy_total_wrong": total_wrong,
                }
            per_seed_metrics[method][seed] = seed_metrics
            missing[label] = miss

    aggregated: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in METHOD_SEEDS}
    metric_names = [
        "individual_accuracy", "micro_pair_accuracy", "table_macro_pair_accuracy",
        "competing_fact_capture_rate", "conditional_capture_rate", "collapsed_pair_rate",
    ]
    for method, seeds in METHOD_SEEDS.items():
        for view in VIEWS:
            aggregated[method][view] = {
                name: mean_sd([per_seed_metrics[method][seed][view][name] for seed in seeds]) for name in metric_names
            }
            aggregated[method][view]["error_taxonomy_pct"] = {
                mode: mean_sd([per_seed_metrics[method][seed][view]["error_taxonomy_pct"][mode] for seed in seeds])
                for mode in ERROR_MODES
            }

    rescue_regression_results: dict[str, dict[str, Any]] = {}
    for other_method, baseline_method in COMPARISONS:
        comp_key = f"{other_method}_vs_{baseline_method}"
        rescue_regression_results[comp_key] = {}
        for view in VIEWS:
            per_seed = {}
            for seed in METHOD_SEEDS[other_method]:
                baseline_by_pair = {o.pair_id: o for o in pair_outcomes[label_for(baseline_method, seed)][view]}
                other_by_pair = {o.pair_id: o for o in pair_outcomes[label_for(other_method, seed)][view]}
                counts = rescue_regression(baseline_by_pair, other_by_pair)
                per_seed[seed] = {"rescue_rate": counts.rescue_rate, "regression_rate": counts.regression_rate}
            rescue_regression_results[comp_key][view] = {
                "per_seed": per_seed,
                "rescue_rate": mean_sd([v["rescue_rate"] for v in per_seed.values()]),
                "regression_rate": mean_sd([v["regression_rate"] for v in per_seed.values()]),
            }

    bootstrap_results: dict[str, dict[str, Any]] = {}
    for other_method, baseline_method in COMPARISONS:
        comp_key = f"{other_method}_vs_{baseline_method}"
        bootstrap_results[comp_key] = {}
        for view in VIEWS:
            per_seed_boot = {}
            for seed in METHOD_SEEDS[other_method]:
                baseline_outcomes = pair_outcomes[label_for(baseline_method, seed)][view]
                other_outcomes = pair_outcomes[label_for(other_method, seed)][view]
                seed_boot = {
                    "pair_accuracy_micro": clustered_bootstrap_delta(
                        baseline_outcomes, other_outcomes, micro_pair_accuracy,
                        n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
                    ),
                    "competing_fact_capture_rate": clustered_bootstrap_delta(
                        baseline_outcomes, other_outcomes, competing_fact_capture_rate,
                        n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
                    ),
                }
                baseline_scored = scored_questions[label_for(baseline_method, seed)][view]
                other_scored = scored_questions[label_for(other_method, seed)][view]
                seed_boot["individual_accuracy"] = clustered_bootstrap_delta_generic(
                    baseline_scored, other_scored, lambda s: s.document_id, question_accuracy,
                    n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed,
                )
                per_seed_boot[seed] = seed_boot
            bootstrap_results[comp_key][view] = per_seed_boot

    (output_dir / "per_seed_metrics.json").write_text(json.dumps(per_seed_metrics, indent=2, sort_keys=True) + "\n")
    (output_dir / "aggregated_metrics.json").write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n")
    (output_dir / "rescue_regression.json").write_text(json.dumps(rescue_regression_results, indent=2, sort_keys=True) + "\n")
    (output_dir / "bootstrap_ci.json").write_text(json.dumps(bootstrap_results, indent=2, sort_keys=True) + "\n")
    (output_dir / "missing_predictions.json").write_text(json.dumps(missing, indent=2, sort_keys=True) + "\n")

    report = render_report(aggregated, rescue_regression_results, bootstrap_results, view_records, missing)
    report_path = output_dir / "period_primary_v1_report.md"
    report_path.write_text(report)

    print(f"pairs: all={len(view_records['all'])} adjacent={len(view_records['adjacent'])} skip={len(view_records['skip'])}")
    print(f"missing predictions (should be all-0): {missing}")
    print(f"report={report_path}")
    return 0


def render_table(aggregated: dict[str, dict[str, dict[str, Any]]], view: str) -> str:
    header = "| Model | Individual Acc | PairAcc (micro) | Table-Macro PairAcc | Capture | Conditional Capture | Collapse |\n"
    header += "|---|---:|---:|---:|---:|---:|---:|\n"
    rows = []
    for method in ("base", "standard_lora", "compute_matched_clean", "cf_augmentation"):
        m = aggregated[method][view]
        rows.append(
            f"| {METHOD_LABELS[method]} | {fmt(m['individual_accuracy'])} | {fmt(m['micro_pair_accuracy'])} "
            f"| {fmt(m['table_macro_pair_accuracy'])} | {fmt(m['competing_fact_capture_rate'])} "
            f"| {fmt(m['conditional_capture_rate'])} | {fmt(m['collapsed_pair_rate'])} |"
        )
    return header + "\n".join(rows) + "\n"


def render_taxonomy_table(aggregated: dict[str, dict[str, dict[str, Any]]], view: str) -> str:
    header = "| Model | Captured by pair | Scale/decimal slip | Other cell in table | Other |\n"
    header += "|---|---:|---:|---:|---:|\n"
    rows = []
    for method in ("base", "standard_lora", "compute_matched_clean", "cf_augmentation"):
        tax = aggregated[method][view]["error_taxonomy_pct"]
        rows.append(
            f"| {METHOD_LABELS[method]} | {fmt(tax['captured_by_pair'])} | {fmt(tax['scale_shift'])} "
            f"| {fmt(tax['other_cell_in_table'])} | {fmt(tax['other'])} |"
        )
    return header + "\n".join(rows) + "\n"


def render_bootstrap_lines(bootstrap_results: dict[str, Any], view: str) -> str:
    lines = []
    for comp_key, per_view in bootstrap_results.items():
        for metric_key, metric_label in [("pair_accuracy_micro", "PairAcc"), ("competing_fact_capture_rate", "Capture")]:
            parts = []
            for seed, seed_boot in per_view[view].items():
                b = seed_boot[metric_key]
                if b["point_estimate"] is None:
                    parts.append(f"seed {seed}: NA")
                else:
                    parts.append(f"seed {seed}: {b['point_estimate']*100:+.2f} [{b['ci_low']*100:+.2f}, {b['ci_high']*100:+.2f}]")
            lines.append(f"- **{comp_key}** ({metric_label}): " + "; ".join(parts))
    return "\n".join(lines)


def render_report(aggregated, rescue_regression_results, bootstrap_results, view_records, missing) -> str:
    lines = ["# TAT-QA Frozen Primary Period Benchmark (v1) -- Evaluation Report", ""]
    lines.append(f"500 pairs total: {len(view_records['adjacent'])} adjacent + {len(view_records['skip'])} skip-period. Reused cached predictions, no model was run by this script.")
    lines.append("")
    if any(missing.values()):
        lines.append(f"Missing predictions by checkpoint (should be 0): {missing}")
        lines.append("")

    view_titles = {"all": "All 500 pairs", "adjacent": f"Adjacent-only ({len(view_records['adjacent'])} pairs)", "skip": f"Skip-period-only ({len(view_records['skip'])} pairs)"}
    for view in VIEWS:
        lines.append(f"## {view_titles[view]}")
        lines.append("")
        lines.append(render_table(aggregated, view))
        lines.append("")
        lines.append("Error-mode taxonomy (share of wrong predictions):")
        lines.append("")
        lines.append(render_taxonomy_table(aggregated, view))
        lines.append("")
        lines.append("Bootstrap deltas (other - baseline, pp, 95% CI, per matched seed):")
        lines.append("")
        lines.append(render_bootstrap_lines(bootstrap_results, view))
        lines.append("")
        lines.append("Rescue / regression (mean ± SD across 3 seeds):")
        lines.append("")
        lines.append("| Comparison | Rescue rate | Regression rate |")
        lines.append("|---|---:|---:|")
        for comp_key, per_view in rescue_regression_results.items():
            d = per_view[view]
            lines.append(f"| {comp_key} | {fmt(d['rescue_rate'])} | {fmt(d['regression_rate'])} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
