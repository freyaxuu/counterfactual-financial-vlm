#!/usr/bin/env python3
"""Aggregate cached predictions for `company_representative_v1` into the
minimal ordinary-reading / domain-transfer report: Individual Accuracy,
3-seed mean +- SD, and ONE bootstrap comparison (CF augmentation - Compute-
matched, document-clustered, 10,000 resamples, 95% CI). No pairs, no
additional metrics -- per explicit instruction (chat, 2026-08-13), this is
deliberately minimal.

Reuses `financial_vlm.evaluation.tatqa_natural_pairs_eval.question_accuracy`
and `.clustered_bootstrap_delta_generic` directly (same functions
`company_confusion_v1`'s aggregation reuses), not a reimplementation.

Reads only cached prediction files -- runs no model, never modifies the
frozen benchmark. Per AGENTS.md's company-data rules, only aggregate
metrics are written/printed.
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

from financial_vlm.evaluation.tatqa_natural_pairs_eval import (  # noqa: E402
    ScoredQuestion,
    clustered_bootstrap_delta_generic,
    question_accuracy,
)

METHOD_SEEDS: dict[str, list[str]] = {
    "base": ["zero_shot"],
    "standard_lora": ["20260804", "20260806", "20260808"],
    "compute_matched_clean": ["20260804", "20260806", "20260808"],
    "cf_augmentation": ["20260804", "20260806", "20260808"],
}
ALL_METHODS = ("base", "standard_lora", "compute_matched_clean", "cf_augmentation")
METHOD_LABELS = {
    "base": "Base (zero-shot)",
    "standard_lora": "Standard",
    "compute_matched_clean": "Compute-matched",
    "cf_augmentation": "CF augmentation",
}


def label_for(method: str, seed: str) -> str:
    return "base" if method == "base" else f"{method}_seed_{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scored(predictions_root: Path, label: str) -> list[ScoredQuestion]:
    path = predictions_root / label / "predictions.jsonl"
    scored = []
    missing_document_id = 0
    for record in load_jsonl(path):
        doc_id = record.get("document_id")
        if doc_id is None:
            missing_document_id += 1
            continue
        scored.append(ScoredQuestion(document_id=doc_id, correct=bool(record.get("numeric_correct"))))
    return scored


def mean_sd(values: list[float]) -> dict[str, float | None]:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
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

    scored_by_label: dict[str, list[ScoredQuestion]] = {}
    per_seed_accuracy: dict[str, dict[str, Any]] = {m: {} for m in METHOD_SEEDS}
    n_scored: dict[str, int] = {}

    for method, seeds in METHOD_SEEDS.items():
        for seed in seeds:
            label = label_for(method, seed)
            scored = load_scored(args.predictions_root, label)
            scored_by_label[label] = scored
            n_scored[label] = len(scored)
            per_seed_accuracy[method][seed] = question_accuracy(scored)

    aggregated = {
        method: mean_sd([per_seed_accuracy[method][seed] for seed in seeds]) for method, seeds in METHOD_SEEDS.items()
    }

    baseline_method, other_method = "compute_matched_clean", "cf_augmentation"
    per_seed_boot: dict[str, Any] = {}
    for seed in METHOD_SEEDS[other_method]:
        baseline_label = label_for(baseline_method, seed)
        other_label = label_for(other_method, seed)
        per_seed_boot[seed] = clustered_bootstrap_delta_generic(
            scored_by_label[baseline_label],
            scored_by_label[other_label],
            lambda s: s.document_id,
            question_accuracy,
            n_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )

    (output_dir / "per_seed_accuracy.json").write_text(json.dumps(per_seed_accuracy, indent=2, sort_keys=True) + "\n")
    (output_dir / "aggregated_accuracy.json").write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n")
    (output_dir / "bootstrap_ci.json").write_text(json.dumps(per_seed_boot, indent=2, sort_keys=True) + "\n")
    (output_dir / "n_scored.json").write_text(json.dumps(n_scored, indent=2, sort_keys=True) + "\n")

    report = render_markdown_report(aggregated, per_seed_boot, n_scored)
    report_path = output_dir / "company_representative_v1_report.md"
    report_path.write_text(report)

    print(f"n_scored={n_scored}")
    print(f"aggregated_accuracy={ {m: fmt(aggregated[m]) for m in ALL_METHODS} }")
    print(f"report={report_path}")
    return 0


def render_markdown_report(aggregated: dict[str, Any], per_seed_boot: dict[str, Any], n_scored: dict[str, int]) -> str:
    lines = ["# company_representative_v1 -- Evaluation Report", ""]
    lines.append(
        "Ordinary-reading / domain-transfer diagnostic: individual (non-paired) verified facts "
        "from the same KPIs surface as company_confusion_v1, disjoint from it. Individual Accuracy "
        "only -- no pairs, no additional metrics, per instruction."
    )
    lines.append("")
    lines.append(f"n scored per model: {n_scored}")
    lines.append("")
    lines.append("## Individual Accuracy (mean ± SD across 3 seeds; base is a single zero-shot run)")
    lines.append("")
    lines.append("| Model | Individual Accuracy |")
    lines.append("|---|---:|")
    for method in ALL_METHODS:
        lines.append(f"| {METHOD_LABELS[method]} | {fmt(aggregated[method])} |")
    lines.append("")
    lines.append("## CF augmentation − Compute-matched (paired, document-clustered bootstrap, 10,000 resamples, 95% CI)")
    lines.append("")
    lines.append("| seed 20260804 | seed 20260806 | seed 20260808 | Mean Δ (pp) |")
    lines.append("|---:|---:|---:|---:|")
    cells = [fmt_delta(per_seed_boot[seed]) for seed in ("20260804", "20260806", "20260808")]
    points = [per_seed_boot[seed]["point_estimate"] for seed in ("20260804", "20260806", "20260808")]
    points = [p for p in points if p is not None]
    mean_cell = f"{sum(points) / len(points) * 100:+.2f}" if points else "NA"
    lines.append(f"| {cells[0]} | {cells[1]} | {cells[2]} | {mean_cell} |")
    lines.append("")
    lines.append(
        "> Compare against company_confusion_v1's PairAcc/Semantic Switch Success deltas to distinguish "
        "broad domain-transfer degradation (visible here) from confusion-specific degradation (visible "
        "there). No human verification pass was applied to this benchmark's gold answers at any stage."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
