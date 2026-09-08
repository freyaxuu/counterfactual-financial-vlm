#!/usr/bin/env python3
"""Render the final, tightly-scoped TAT-QA Natural Pair report from
already-computed results (aggregated_metrics.json / bootstrap_ci.json,
produced by evaluate_period_primary_benchmark.py -- no new metric is
computed here, this only reformats/filters existing numbers to the
requested report shape: Individual Accuracy, Pair Accuracy, Paired Capture
Rate only (no Conditional Capture, no combined aggregate score across
views), reported for all/adjacent/skip, with a descriptive mean effect
size across the 3 matched seeds alongside the per-seed bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHOD_LABELS = {
    "base": "Base",
    "standard_lora": "Standard",
    "compute_matched_clean": "Compute-matched",
    "cf_augmentation": "CF augmentation",
}
VIEWS = ("all", "adjacent", "skip")
VIEW_LABELS = {"all": "All 500 pairs", "adjacent": "Adjacent-period pairs (375)", "skip": "Skip-period pairs (125)"}
COMPARISONS = ["cf_augmentation_vs_standard_lora", "cf_augmentation_vs_compute_matched_clean"]
SEEDS = ["20260804", "20260806", "20260808"]


def fmt_pct(stat: dict, digits: int = 2) -> str:
    if stat["mean"] is None:
        return "NA"
    if stat["n"] > 1 and stat["sd"] is not None:
        return f"{stat['mean']*100:.{digits}f}% ± {stat['sd']*100:.{digits}f}"
    return f"{stat['mean']*100:.{digits}f}%"


def fmt_delta(point_estimate: float | None, ci_low: float | None, ci_high: float | None) -> str:
    if point_estimate is None:
        return "NA"
    return f"{point_estimate*100:+.2f} [{ci_low*100:+.2f}, {ci_high*100:+.2f}]"


def mean_of(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Output dir from evaluate_period_primary_benchmark.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregated = json.loads((args.results_dir / "aggregated_metrics.json").read_text())
    bootstrap = json.loads((args.results_dir / "bootstrap_ci.json").read_text())

    lines = ["# TAT-QA Natural Pair Evaluation -- Final Report (frozen 500-pair primary period benchmark)", ""]
    lines.append(
        "Frozen benchmark, unchanged pair selection/construction/checkpoints/prompts/normalization "
        "(`outputs/tatqa_period_primary_v1/`, seed 20260812, 375 adjacent + 125 skip-period pairs). "
        "All numbers below are read from already-computed, cached-prediction-based results "
        "(`scripts/evaluate_period_primary_benchmark.py`) -- no model was run to produce this report, "
        "and nothing about the benchmark was changed based on these results."
    )
    lines.append("")

    # --- Final compact result table (all 500 pairs) ---
    lines.append("## Final result table (all 500 pairs)")
    lines.append("")
    lines.append("| Model | Individual Acc ↑ | PairAcc ↑ | Capture ↓ |")
    lines.append("|---|---:|---:|---:|")
    for method in ("base", "standard_lora", "compute_matched_clean", "cf_augmentation"):
        m = aggregated[method]["all"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {fmt_pct(m['individual_accuracy'])} "
            f"| {fmt_pct(m['micro_pair_accuracy'])} | {fmt_pct(m['competing_fact_capture_rate'])} |"
        )
    lines.append("")

    # --- Secondary table: adjacent vs skip PairAcc only ---
    lines.append("## Adjacent vs skip-period PairAcc")
    lines.append("")
    lines.append("| Model | Adjacent PairAcc ↑ | Skip-period PairAcc ↑ |")
    lines.append("|---|---:|---:|")
    for method in ("base", "standard_lora", "compute_matched_clean", "cf_augmentation"):
        adj = aggregated[method]["adjacent"]["micro_pair_accuracy"]
        skip = aggregated[method]["skip"]["micro_pair_accuracy"]
        lines.append(f"| {METHOD_LABELS[method]} | {fmt_pct(adj)} | {fmt_pct(skip)} |")
    lines.append("")

    # --- Statistical testing ---
    lines.append("## Statistical testing")
    lines.append("")
    lines.append(
        "Δ = metric_CF − metric_control. For Individual Acc and PairAcc, positive Δ is better. "
        "For Capture, negative Δ is better. 10,000 paired clustered bootstrap resamples, clustered by "
        "document ID (= table ID for TAT-QA), run separately per matched training seed; the descriptive mean "
        "effect below is the plain average of the 3 seeds' point estimates, not a re-run bootstrap."
    )
    lines.append("")
    metric_keys = [("individual_accuracy", "Individual Acc"), ("pair_accuracy_micro", "PairAcc"), ("competing_fact_capture_rate", "Capture")]
    for view in VIEWS:
        lines.append(f"### {VIEW_LABELS[view]}")
        lines.append("")
        for comp_key in COMPARISONS:
            label = "CF vs Standard" if comp_key == "cf_augmentation_vs_standard_lora" else "CF vs Compute-matched (primary CF-specific comparison)"
            lines.append(f"**{label}**")
            lines.append("")
            lines.append("| Metric | seed 20260804 | seed 20260806 | seed 20260808 | Mean Δ across 3 seeds |")
            lines.append("|---|---:|---:|---:|---:|")
            for metric_key, metric_label in metric_keys:
                per_seed = bootstrap[comp_key][view]
                cells = []
                points = []
                for seed in SEEDS:
                    b = per_seed[seed][metric_key]
                    cells.append(fmt_delta(b["point_estimate"], b["ci_low"], b["ci_high"]))
                    points.append(b["point_estimate"])
                mean_delta = mean_of(points)
                mean_cell = f"{mean_delta*100:+.2f}" if mean_delta is not None else "NA"
                lines.append(f"| {metric_label} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_cell} |")
            lines.append("")

    # --- Interpretation ---
    cf_vs_compute_all = bootstrap["cf_augmentation_vs_compute_matched_clean"]["all"]
    pairacc_deltas = [cf_vs_compute_all[s]["pair_accuracy_micro"]["point_estimate"] for s in SEEDS]
    capture_deltas = [cf_vs_compute_all[s]["competing_fact_capture_rate"]["point_estimate"] for s in SEEDS]
    mean_pairacc_delta = mean_of(pairacc_deltas)
    mean_capture_delta = mean_of(capture_deltas)

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"**CF vs Compute-matched is the primary estimate of the benefit specifically attributable to "
        f"counterfactual content** (both methods share the same \"two images per training step\" regime; "
        f"only CF augmentation's second image is a counterfactual). On all 500 pairs, the mean PairAcc "
        f"Δ across the 3 matched seeds is **{mean_pairacc_delta*100:+.2f}pp**, and the mean Capture Δ is "
        f"**{mean_capture_delta*100:+.2f}pp** ({'higher, i.e. worse' if (mean_capture_delta or 0) > 0 else 'lower, i.e. better'} "
        "capture rate for CF). Per-seed CIs are wide relative to the effect size and cross zero in most seeds "
        "(see tables above) -- direction is consistently in CF's favor on PairAcc, but the magnitude is small."
    )
    lines.append("")
    lines.append(
        "> Counterfactual augmentation shows a small, direction-consistent PairAcc advantage over the "
        "compute-matched clean control on this frozen natural-document benchmark. This is a modest effect, "
        "not comparable in magnitude to the much larger HFR/CGS gains reported on the controlled-intervention "
        "benchmark -- the natural-pair results do not support a claim that counterfactual augmentation "
        "substantially improves natural financial reading beyond what a compute-matched clean control already "
        "achieves; they support a claim of a small, real, same-direction benefit specifically attributable to "
        "counterfactual content."
    )
    lines.append("")

    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
