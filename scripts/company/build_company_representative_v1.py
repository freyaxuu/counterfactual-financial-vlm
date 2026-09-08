#!/usr/bin/env python3
"""Build and freeze `company_representative_v1`: ~100-150 individual
(non-paired) verified facts from the `KPIs` surface, for the ordinary-
reading / domain-transfer diagnostic (chat, 2026-08-13) -- distinct from
`company_confusion_v1`'s confusion-pair reliability diagnostic.

Model-blind construction, same as `company_confusion_v1`:

1. Load all `KPIs` grouped-field instances (with bboxes).
2. Exclude any (document_id, page_id, instance_index) already used by
   `company_confusion_v1` -- the two frozen sets stay disjoint, per
   `docs/company-benchmark-diagnosis-report.md` section 4's own principle
   that representative sampling stays distinct from challenge-set
   construction.
3. For each remaining instance, for each populated metric field, build a
   candidate fact (question via the same `period_label_text`/
   `build_question_text` used for confusion pairs) and gate it through the
   same calibrated OCR-proximity proxy check (`ACCEPTED_OCR_OUTCOMES`).
4. Apply stratified per-page/per-company/per-metric caps
   (`kpi_representative_facts.stratified_cap_facts`) so the sample doesn't
   just reproduce the raw surface's page/company/metric concentration.

Requires the private `evolution_ai_datasets` package and read access to the
dataset directory (on the training server: the dataset access group before running).

Output split, same convention as every other script in this repo:
- `company_representative_v1_manifest.json` -- aggregate counts/parameters
  only, safe to share.
- `company_representative_v1.jsonl` -- contains real question/answer data.
  Never committed to Git, never printed to a chat transcript.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.evaluation.company_profile_audit import normalize_company_name  # noqa: E402
from financial_vlm.evaluation.kpi_question_pairs import ACCEPTED_OCR_OUTCOMES, build_question_text, period_label_text  # noqa: E402
from financial_vlm.evaluation.kpi_representative_facts import RepresentativeFactRecord, stratified_cap_facts  # noqa: E402
from financial_vlm.evaluation.ocr_quality_proxy import check_value_against_ocr, tokens_near_bbox  # noqa: E402
from financial_vlm.integrations.evolution_ai_datasets_adapter import (  # noqa: E402
    KPI_METRIC_FIELDS,
    load_grouped_field_instances,
    load_ocr_tokens,
)

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
    parser.add_argument("--company-confusion-v1", type=Path, required=True, help="Frozen company_confusion_v1.jsonl, to exclude its instances")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-page", type=int, default=3)
    parser.add_argument("--max-per-company", type=int, default=3)
    parser.add_argument("--max-per-metric", type=int, default=25)
    parser.add_argument("--shuffle-seed", type=int, default=20260813)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
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


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fact_id_for(document_id: str, page_id: str, instance_id: int, metric: str) -> str:
    raw = f"{document_id}|{page_id}|{instance_id}|{metric}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = ensure_empty_or_create(args.output_dir)

    confusion_pairs = load_jsonl(args.company_confusion_v1.expanduser().resolve())
    excluded_instances: set[tuple[str, str, int]] = set()
    for pair in confusion_pairs:
        excluded_instances.add((pair["document_id"], pair["page_a"], pair["instance_a"]))
        excluded_instances.add((pair["document_id"], pair["page_b"], pair["instance_b"]))

    instances = load_grouped_field_instances(dataset_root, "KPIs")

    ocr_cache: dict[tuple[str, str], tuple] = {}

    def ocr_tokens_for(document_id: str, page_id: str):
        key = (document_id, page_id)
        if key not in ocr_cache:
            ocr_cache[key] = load_ocr_tokens(dataset_root, document_id, page_id)
        return ocr_cache[key]

    candidates: list[RepresentativeFactRecord] = []
    skipped_excluded_instance = 0
    skipped_no_company_name = 0
    metric_checks_total = 0
    metric_checks_passed = 0
    ocr_outcome_counts: Counter[str] = Counter()

    for instance in instances:
        loc = (instance.document_id, instance.page_id, instance.instance_index)
        if loc in excluded_instances:
            skipped_excluded_instance += 1
            continue

        raw_name = instance.field_values.get("company_name")
        if not raw_name:
            skipped_no_company_name += 1
            continue
        company_key = hash_company(normalize_company_name(raw_name))

        period_text = period_label_text(
            instance.field_values.get("period"), instance.field_values.get("month"), instance.field_values.get("year")
        )

        tokens = ocr_tokens_for(instance.document_id, instance.page_id)

        for metric in KPI_METRIC_FIELDS:
            value = instance.field_values.get(metric)
            bbox = instance.field_bboxes.get(metric)
            if value is None or bbox is None:
                continue

            metric_checks_total += 1
            outcome = check_value_against_ocr(value, _DATA_TYPES[metric], tokens_near_bbox(tokens, bbox))
            ocr_outcome_counts[outcome] += 1
            if outcome not in ACCEPTED_OCR_OUTCOMES:
                continue
            metric_checks_passed += 1

            candidates.append(
                RepresentativeFactRecord(
                    fact_id=fact_id_for(instance.document_id, instance.page_id, instance.instance_index, metric),
                    document_id=instance.document_id,
                    page_id=instance.page_id,
                    instance_id=instance.instance_index,
                    company_key=company_key,
                    metric=metric,
                    bbox=bbox,
                    period_text=period_text,
                    question=build_question_text(metric, period_text),
                    gold_answer=value,
                    ocr_outcome=outcome,
                    verification_status="automated_pass",
                )
            )

    frozen = stratified_cap_facts(
        candidates,
        max_per_page=args.max_per_page,
        max_per_company=args.max_per_company,
        max_per_metric=args.max_per_metric,
        shuffle_seed=args.shuffle_seed,
    )

    manifest = {
        "version": "company_representative_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_root": str(dataset_root),
        "company_confusion_v1_path": str(args.company_confusion_v1),
        "cap_parameters": {
            "max_per_page": args.max_per_page,
            "max_per_company": args.max_per_company,
            "max_per_metric": args.max_per_metric,
            "shuffle_seed": args.shuffle_seed,
        },
        "skipped_excluded_instance": skipped_excluded_instance,
        "skipped_no_company_name": skipped_no_company_name,
        "metric_checks_total": metric_checks_total,
        "metric_checks_passed": metric_checks_passed,
        "ocr_outcome_counts": dict(ocr_outcome_counts),
        "candidate_facts_after_verification": len(candidates),
        "frozen_fact_count": len(frozen),
        "frozen_metric_counts": dict(Counter(f.metric for f in frozen)),
        "frozen_distinct_pages": len({(f.document_id, f.page_id) for f in frozen}),
        "frozen_distinct_companies": len({f.company_key for f in frozen}),
    }

    manifest_path = output_dir / "company_representative_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    frozen_path = output_dir / "company_representative_v1.jsonl"
    with frozen_path.open("w", encoding="utf-8") as handle:
        for record in frozen:
            handle.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")

    print(f"skipped_excluded_instance={skipped_excluded_instance}")
    print(f"candidate_facts_after_verification={len(candidates)}")
    print(f"metric_checks_passed={metric_checks_passed}/{metric_checks_total}")
    print(f"frozen_fact_count={len(frozen)}")
    print(f"frozen_metric_counts={manifest['frozen_metric_counts']}")
    print(f"frozen_distinct_pages={manifest['frozen_distinct_pages']}")
    print(f"frozen_distinct_companies={manifest['frozen_distinct_companies']}")
    print(f"manifest_json={manifest_path}")
    print(f"frozen_jsonl={frozen_path} (REAL VALUES -- private, keep off Git, do not paste into chat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
