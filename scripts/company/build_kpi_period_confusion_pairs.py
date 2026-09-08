#!/usr/bin/env python3
"""Build the period/basis confusion-pair question set from
`grouped_fields.KPIs`, gated by automated checks only (no human
verification pass) -- see `financial_vlm.evaluation.kpi_question_pairs`
module docstring for the scope and risk of that tradeoff, and
`docs/company-benchmark-diagnosis-report.md` sections 0.1/0.2 for the
calibration work behind the checks used here.

Pipeline:

1. Load all `KPIs` instances (with bboxes).
2. Construct period/basis-pair candidates
   (`financial_vlm.evaluation.kpi_confusion_pairs`), same pseudo-period
   exclusion rules and column-position disambiguation as the diagnosis
   report.
3. Drop any pair touching an instance that belongs to a flagged
   disagreeing same-column duplicate group
   (`find_disagreeing_duplicate_instances`).
4. For each surviving pair, for each metric field populated on both sides,
   run the (twice-calibrated) OCR-proximity proxy check against both
   sides' bounding boxes; only emit a question pair if both sides land in
   `kpi_question_pairs.ACCEPTED_OCR_OUTCOMES`.

Scope: period and basis pairs only (report section 5A/5C) -- metric pairs
(section 5B) need human-adjudicated pair-type selection and are not built
here.

Requires the private `evolution_ai_datasets` package and read access to the
dataset directory (on the training server: the dataset access group before running).

Output split, same convention as every other script in this repo:
- `*_summary.json` -- aggregate counts only, safe to share.
- `*_question_pairs.jsonl` -- contains real company/metric/period/value
  data. Never committed to Git, never printed to a chat transcript (see
  AGENTS.md's private-data rules) -- stays under `--output-dir` on the
  server.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.evaluation.company_profile_audit import normalize_company_name  # noqa: E402
from financial_vlm.evaluation.kpi_confusion_pairs import (  # noqa: E402
    KPIRecord,
    column_x_from_bboxes,
    find_disagreeing_duplicate_instances,
    find_period_and_basis_pairs,
    group_by_company,
)
from financial_vlm.evaluation.kpi_question_pairs import (  # noqa: E402
    QuestionPairRecord,
    build_question_text,
    metric_pair_automated_pass,
    period_label_text,
)
from financial_vlm.evaluation.ocr_quality_proxy import check_value_against_ocr, tokens_near_bbox  # noqa: E402
from financial_vlm.integrations.evolution_ai_datasets_adapter import (  # noqa: E402
    KPI_METRIC_FIELDS,
    load_grouped_field_instances,
    load_ocr_tokens,
)

# `da` (data_type) per KPI metric field, from document_types_schema.json.
_DATA_TYPES: dict[str, str] = {
    "sales": "monetary",
    "EBITDA": "monetary",
    "net_debt": "monetary",
    "EV": "monetary",
    "valuation_multiple": "numerical",
    "net_leverage_multiple": "numerical",
    "Cash and Cash Equivalents": "monetary",
    "total_debt": "monetary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/private-dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def ensure_empty_or_create(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def hash_company(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def pair_id_for(document_id: str, page_a: str, instance_a: int, page_b: str, instance_b: int, metric: str) -> str:
    raw = f"{document_id}|{page_a}|{instance_a}|{page_b}|{instance_b}|{metric}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = ensure_empty_or_create(args.output_dir)

    instances = load_grouped_field_instances(dataset_root, "KPIs")
    instance_by_loc = {(i.document_id, i.page_id, i.instance_index): i for i in instances}

    records: list[KPIRecord] = []
    for instance in instances:
        raw_name = instance.field_values.get("company_name")
        if not raw_name:
            continue
        metric_values = {name: value for name, value in instance.field_values.items() if name in KPI_METRIC_FIELDS}
        records.append(
            KPIRecord(
                document_id=instance.document_id,
                page_id=instance.page_id,
                instance_index=instance.instance_index,
                company_key=hash_company(normalize_company_name(raw_name)),
                period=instance.field_values.get("period"),
                year=instance.field_values.get("year"),
                month=instance.field_values.get("month"),
                metric_values=metric_values,
                column_x=column_x_from_bboxes(instance.field_bboxes, KPI_METRIC_FIELDS),
            )
        )
    record_by_loc = {(r.document_id, r.page_id, r.instance_index): r for r in records}

    groups = group_by_company(records)
    pair_summary = find_period_and_basis_pairs(groups)
    flagged_instances = find_disagreeing_duplicate_instances(groups)

    ocr_cache: dict[tuple[str, str], tuple] = {}

    def ocr_tokens_for(document_id: str, page_id: str):
        key = (document_id, page_id)
        if key not in ocr_cache:
            ocr_cache[key] = load_ocr_tokens(dataset_root, document_id, page_id)
        return ocr_cache[key]

    question_pairs: list[QuestionPairRecord] = []
    pairs_dropped_flagged_duplicate = 0
    pairs_with_no_common_metric = 0
    metric_checks_total = 0
    metric_checks_passed = 0
    ocr_outcome_counts: dict[str, int] = {}

    all_candidate_pairs = (
        *((p, "period") for p in pair_summary.valid_period_pairs),
        *((p, "basis") for p in pair_summary.basis_pairs),
    )

    for pair, pair_type in all_candidate_pairs:
        loc_a = (pair.document_id, pair.page_a, pair.instance_a)
        loc_b = (pair.document_id, pair.page_b, pair.instance_b)
        if loc_a in flagged_instances or loc_b in flagged_instances:
            pairs_dropped_flagged_duplicate += 1
            continue

        instance_a, instance_b = instance_by_loc.get(loc_a), instance_by_loc.get(loc_b)
        record_a, record_b = record_by_loc.get(loc_a), record_by_loc.get(loc_b)
        if not (instance_a and instance_b and record_a and record_b):
            continue

        common_metrics = sorted(set(record_a.metric_values) & set(record_b.metric_values))
        if not common_metrics:
            pairs_with_no_common_metric += 1
            continue

        tokens_a = ocr_tokens_for(pair.document_id, pair.page_a)
        tokens_b = ocr_tokens_for(pair.document_id, pair.page_b)

        for metric in common_metrics:
            bbox_a = instance_a.field_bboxes.get(metric)
            bbox_b = instance_b.field_bboxes.get(metric)
            if bbox_a is None or bbox_b is None:
                continue

            metric_checks_total += 1
            value_a, value_b = record_a.metric_values[metric], record_b.metric_values[metric]
            data_type = _DATA_TYPES[metric]
            outcome_a = check_value_against_ocr(value_a, data_type, tokens_near_bbox(tokens_a, bbox_a))
            outcome_b = check_value_against_ocr(value_b, data_type, tokens_near_bbox(tokens_b, bbox_b))
            ocr_outcome_counts[outcome_a] = ocr_outcome_counts.get(outcome_a, 0) + 1
            ocr_outcome_counts[outcome_b] = ocr_outcome_counts.get(outcome_b, 0) + 1

            if not metric_pair_automated_pass(outcome_a, outcome_b):
                continue
            metric_checks_passed += 1

            period_text_a = period_label_text(record_a.period, record_a.month, record_a.year)
            period_text_b = period_label_text(record_b.period, record_b.month, record_b.year)
            question_pairs.append(
                QuestionPairRecord(
                    pair_id=pair_id_for(pair.document_id, pair.page_a, pair.instance_a, pair.page_b, pair.instance_b, metric),
                    pair_type=pair_type,
                    document_id=pair.document_id,
                    company_key=pair.company_key,
                    metric=metric,
                    page_a=pair.page_a,
                    page_b=pair.page_b,
                    instance_a=pair.instance_a,
                    instance_b=pair.instance_b,
                    bbox_a=bbox_a,
                    bbox_b=bbox_b,
                    period_text_a=period_text_a,
                    period_text_b=period_text_b,
                    question_a=build_question_text(metric, period_text_a),
                    question_b=build_question_text(metric, period_text_b),
                    gold_answer_a=value_a,
                    gold_answer_b=value_b,
                    ocr_outcome_a=outcome_a,
                    ocr_outcome_b=outcome_b,
                    verification_status="automated_pass",
                )
            )

    summary = {
        "dataset_root": str(dataset_root),
        "git_commit": git_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_method": (
            "automated_only (OCR-proximity proxy, calibrated across two rounds -- "
            "see report sections 0.1/0.2 -- plus duplicate-period-key disagreement "
            "filter; no human visual review)"
        ),
        "period_pair_candidates": len(pair_summary.valid_period_pairs),
        "basis_pair_candidates": len(pair_summary.basis_pairs),
        "pairs_dropped_flagged_duplicate": pairs_dropped_flagged_duplicate,
        "pairs_with_no_common_metric": pairs_with_no_common_metric,
        "metric_level_checks_run": metric_checks_total,
        "metric_level_checks_passed": metric_checks_passed,
        "metric_level_pass_rate": (metric_checks_passed / metric_checks_total) if metric_checks_total else None,
        "ocr_outcome_counts_all_sides_checked": ocr_outcome_counts,
        "question_pairs_emitted": len(question_pairs),
        "question_pairs_by_type": {
            pair_type: sum(1 for qp in question_pairs if qp.pair_type == pair_type)
            for pair_type in ("period", "basis")
        },
    }

    summary_path = output_dir / "kpi_period_question_pairs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    pairs_path = output_dir / "kpi_period_question_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for record in question_pairs:
            handle.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")

    print(f"period_pair_candidates={summary['period_pair_candidates']}")
    print(f"basis_pair_candidates={summary['basis_pair_candidates']}")
    print(f"pairs_dropped_flagged_duplicate={summary['pairs_dropped_flagged_duplicate']}")
    print(f"metric_level_checks_run={summary['metric_level_checks_run']}")
    print(f"metric_level_checks_passed={summary['metric_level_checks_passed']} (rate={summary['metric_level_pass_rate']})")
    print(f"question_pairs_emitted={summary['question_pairs_emitted']} {summary['question_pairs_by_type']}")
    print(f"summary_json={summary_path}")
    print(f"question_pairs_jsonl={pairs_path} (REAL VALUES -- private, keep off Git, do not paste into chat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
