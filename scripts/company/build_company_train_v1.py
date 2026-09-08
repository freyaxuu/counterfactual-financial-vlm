#!/usr/bin/env python3
"""Build `company_train_v1`: an exploratory Standard-LoRA training manifest
from a narrowly-scoped, verified-disjoint slice of company data.

Per the one-time exception documented in `AGENTS.md`'s "Company data"
section (added 2026-08-17), this is restricted to 5 documents already
verified (read-only, by normalized company name) to share zero companies
with `company_representative_v1`/`company_confusion_v1` -- those two
frozen benchmarks are never read/written/modified by this script and must
remain byte-identical afterward.

Pipeline, reusing the same already-tested functions
`build_company_representative_v1.py` uses (OCR-proxy gate, question
templating) -- this script adds only: (a) a document-id allowlist, (b) a
canonical_clean_v1 schema gate (`normalise_numeric_value` must parse the
raw answer -- some company values like "8.5x"/"$1.2B" pass the OCR-proxy's
own smarter normalizer but not canonical_schema's plainer numeric regex,
so this is a real, separate, additive filter, not a duplicate of the OCR
gate), and (c) a per-document cap (requested explicitly, since the raw
5-document pool is heavily concentrated: 2 of 5 documents otherwise supply
71% of all accepted facts).

Requires the private `evolution_ai_datasets` package and read access to the
dataset directory (on the training server: the dataset access group/`bash -c` before running).

Output split, same convention as every other script in this repo:
- `construction_summary.json` -- aggregate counts/parameters only, safe to
  share, safe to commit.
- `manifest_train.jsonl` / `manifest_dev.jsonl` -- contain real question/
  answer data. Never committed to Git, never printed to a chat transcript.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.canonical_schema import normalise_numeric_value  # noqa: E402
from financial_vlm.data.company_loader import CompanyTrainCandidate, build_company_clean_record, cap_per_document  # noqa: E402
from financial_vlm.evaluation.kpi_question_pairs import ACCEPTED_OCR_OUTCOMES, build_question_text, period_label_text  # noqa: E402
from financial_vlm.evaluation.ocr_quality_proxy import check_value_against_ocr, tokens_near_bbox  # noqa: E402
from financial_vlm.integrations.evolution_ai_datasets_adapter import (  # noqa: E402
    KPI_METRIC_FIELDS,
    load_grouped_field_instances,
    load_ocr_tokens,
)

# The 5 documents verified disjoint (by normalized company name) from
# company_representative_v1/company_confusion_v1 -- see AGENTS.md's
# "Company data" exception. Not a CLI default to override casually: the
# exception is scoped to exactly these documents.
EXCEPTION_DOCUMENT_IDS: tuple[str, ...] = (
    "8WBh5X4suJ89eLs29KEKGc",
    "L3V6UTr7kNWNK3TtWT2EK3",
    "QqnoV27mD2XDWhXqkivjVS",
    "bpgbScHakK5pmDDMc8Fsag",
    "kp53BRGzTicjrSKACNkmKG",
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
    parser.add_argument("--company-representative-v1", type=Path, required=True, help="Frozen company_representative_v1.jsonl, to exclude its instances (defense in depth)")
    parser.add_argument("--company-confusion-v1", type=Path, required=True, help="Frozen company_confusion_v1.jsonl, to exclude its instances (defense in depth)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-document", type=int, default=200)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--shuffle-seed", type=int, default=20260817)
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


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_path_for(dataset_root: Path, document_id: str, page_id: str) -> str:
    return str(dataset_root / "files" / document_id / "pages" / page_id / "image.png")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = ensure_empty_or_create(args.output_dir)

    excluded_instances: set[tuple[str, str, int]] = set()
    for path in (args.company_representative_v1, args.company_confusion_v1):
        for record in load_jsonl(path.expanduser().resolve()):
            if "page_id" in record and "instance_id" in record:
                excluded_instances.add((record["document_id"], record["page_id"], record["instance_id"]))
            else:
                excluded_instances.add((record["document_id"], record["page_a"], record["instance_a"]))
                excluded_instances.add((record["document_id"], record["page_b"], record["instance_b"]))

    instances = load_grouped_field_instances(dataset_root, "KPIs")

    ocr_cache: dict[tuple[str, str], tuple] = {}

    def ocr_tokens_for(document_id: str, page_id: str):
        key = (document_id, page_id)
        if key not in ocr_cache:
            ocr_cache[key] = load_ocr_tokens(dataset_root, document_id, page_id)
        return ocr_cache[key]

    candidates: list[CompanyTrainCandidate] = []
    skipped_wrong_document = 0
    skipped_excluded_instance = 0
    metric_checks_total = 0
    metric_checks_ocr_passed = 0
    rejected_unparseable_numeric_value = 0
    ocr_outcome_counts: Counter[str] = Counter()

    for instance in instances:
        if instance.document_id not in EXCEPTION_DOCUMENT_IDS:
            skipped_wrong_document += 1
            continue

        loc = (instance.document_id, instance.page_id, instance.instance_index)
        if loc in excluded_instances:
            skipped_excluded_instance += 1
            continue

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
            metric_checks_ocr_passed += 1

            if normalise_numeric_value(value) is None:
                rejected_unparseable_numeric_value += 1
                continue

            candidates.append(
                CompanyTrainCandidate(
                    document_id=instance.document_id,
                    page_id=instance.page_id,
                    instance_index=instance.instance_index,
                    metric=metric,
                    period_text=period_text,
                    gold_answer=value,
                    question=build_question_text(metric, period_text),
                    bbox=bbox,
                )
            )

    capped = cap_per_document(candidates, max_per_document=args.max_per_document, shuffle_seed=args.shuffle_seed)

    # Deterministic 90/10 record-level train/dev split. With only 5
    # documents, a document-level split would be far too coarse; dev is
    # not used for model/checkpoint selection here regardless (AGENTS.md
    # forbids that for company data), matching every other config in this
    # repo's eval_strategy: "no" convention -- this split exists only to
    # match the manifest_train/manifest_dev shape the trainer expects.
    split_order = list(capped)
    random.Random(args.shuffle_seed + 1).shuffle(split_order)
    n_dev = round(len(split_order) * args.dev_fraction)
    dev_candidates = split_order[:n_dev]
    train_candidates = split_order[n_dev:]

    from PIL import Image  # deferred: only needed once real candidates exist

    image_size_cache: dict[str, tuple[int, int]] = {}

    def image_size_for(path: str) -> tuple[int, int]:
        if path not in image_size_cache:
            with Image.open(path) as img:
                image_size_cache[path] = img.size
        return image_size_cache[path]

    rejected_image_open_failed = 0
    train_records: list[dict] = []
    dev_records: list[dict] = []
    for split_name, split_candidates, out_list in (("train", train_candidates, train_records), ("dev", dev_candidates, dev_records)):
        for group_index, candidate in enumerate(split_candidates):
            image_path = image_path_for(dataset_root, candidate.document_id, candidate.page_id)
            try:
                image_size = image_size_for(image_path)
            except Exception:
                rejected_image_open_failed += 1
                continue
            record = build_company_clean_record(
                document_id=candidate.document_id,
                page_id=candidate.page_id,
                instance_index=candidate.instance_index,
                split=split_name,
                image_path=image_path,
                image_size=image_size,
                metric=candidate.metric,
                period_text=candidate.period_text,
                gold_answer=candidate.gold_answer,
                question=candidate.question,
                bbox=candidate.bbox,
                group_index=group_index,
            )
            out_list.append(record)

    train_path = output_dir / "manifest_train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for record in train_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    dev_path = output_dir / "manifest_dev.jsonl"
    with dev_path.open("w", encoding="utf-8") as handle:
        for record in dev_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    per_document_final_counts = Counter(c.document_id for c in capped)
    summary = {
        "version": "company_train_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_root": str(dataset_root),
        "exception_document_ids": list(EXCEPTION_DOCUMENT_IDS),
        "cap_parameters": {
            "max_per_document": args.max_per_document,
            "dev_fraction": args.dev_fraction,
            "shuffle_seed": args.shuffle_seed,
        },
        "skipped_wrong_document": skipped_wrong_document,
        "skipped_excluded_instance": skipped_excluded_instance,
        "metric_checks_total": metric_checks_total,
        "metric_checks_ocr_passed": metric_checks_ocr_passed,
        "rejected_unparseable_numeric_value": rejected_unparseable_numeric_value,
        "ocr_outcome_counts": dict(ocr_outcome_counts),
        "candidate_facts_after_all_gates": len(candidates),
        "capped_fact_count": len(capped),
        "rejected_image_open_failed": rejected_image_open_failed,
        "train_record_count": len(train_records),
        "dev_record_count": len(dev_records),
        "per_document_final_counts": dict(per_document_final_counts),
        "per_document_metric_counts": {
            doc: dict(Counter(c.metric for c in capped if c.document_id == doc)) for doc in EXCEPTION_DOCUMENT_IDS
        },
    }
    summary_path = output_dir / "construction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"skipped_wrong_document={skipped_wrong_document}")
    print(f"skipped_excluded_instance={skipped_excluded_instance}")
    print(f"metric_checks_ocr_passed={metric_checks_ocr_passed}/{metric_checks_total}")
    print(f"rejected_unparseable_numeric_value={rejected_unparseable_numeric_value}")
    print(f"candidate_facts_after_all_gates={len(candidates)}")
    print(f"capped_fact_count={len(capped)}")
    print(f"per_document_final_counts={dict(per_document_final_counts)}")
    print(f"train_record_count={len(train_records)}")
    print(f"dev_record_count={len(dev_records)}")
    print(f"summary_json={summary_path}")
    print(f"manifest_train={train_path} (REAL VALUES -- private, keep off Git, do not paste into chat)")
    print(f"manifest_dev={dev_path} (REAL VALUES -- private, keep off Git, do not paste into chat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
