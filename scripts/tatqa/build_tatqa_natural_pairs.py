#!/usr/bin/env python3
"""Build and freeze the TAT-QA Natural Financial Confusion Pair Challenge.

Frozen Strategy C (hybrid) protocol -- see
docs/tatqa-natural-pairs-construction-diagnosis.md for the diagnosis this
implements. Produces THREE separately labeled subsets from the same run:

1. ``human_authored``  -- existing TAT-QA questions, paired and classified
   exactly as Phase 1 did (``extract_eligible_question`` / ``classify_pair``
   / ``build_pair_record``), now also carrying a lexical-relatedness score
   for metric-type pairs so low-relatedness ones can be filtered downstream
   without being discarded here.
2. ``period_derived``  -- table-derived (Strategy B) period pairs: same
   table, same normalized metric, same unit class, and the two period
   strings must not differ by anything other than their temporal component
   (this is what excludes same-year scenario columns like "2019 actual" vs
   "2019 threshold" -- see ``temporal_remainder``). v4: any two periods
   within a metric group may pair, not only chronologically adjacent ones
   -- capped per (table, metric) group and per table instead, with the
   smallest year-gap pairs kept first (deterministic). See
   ``METRIC_GROUP_PERIOD_PAIR_CAP`` / ``TABLE_PERIOD_PAIR_CAP`` in
   ``tatqa_natural_pairs.py``.
3. ``metric_candidate`` -- table-derived (Strategy B) metric pairs, accepted
   entirely by an automatic threshold gate (same table, same period,
   compatible unit class, resolved scope, and jaccard >=
   ``METRIC_RELATEDNESS_JACCARD_THRESHOLD`` or a basis-qualifier match;
   footnote-suffix duplicates excluded). v1 planned a per-candidate
   model-blind human adjudication pass on top of this; the user reviewed a
   spot-check sample instead and opted to raise the threshold
   (0.34 -> 0.50 in v2) rather than label all 882 candidates individually --
   see the PROTOCOL_VERSION changelog in
   ``financial_vlm.data.tatqa_natural_pairs`` for the evidence. No model
   prediction is inspected anywhere in this script.

This script does not run model inference, does not train anything, and
does not touch or generate any images -- construction is entirely text/
table-driven. Per AGENTS.md dataset roles, TAT-QA is public real-world
data: eval-only, never for training or hyperparameter selection.

Construction rules are versioned (``financial_vlm.data.tatqa_natural_pairs.
PROTOCOL_VERSION``) and must not change based on later model results -- a
rule change is a new protocol version, not a silent rewrite of this one.

Input: local TAT-QA JSON files (e.g. `tatqa_dataset_test_gold.json`,
`tatqa_dataset_dev.json`) as distributed at
https://huggingface.co/datasets/next-tat/TAT-QA. Download them yourself
first; this script does not fetch data over the network. Do not pass a
train-split file here -- train was used for gradient updates on the models
this benchmark is meant to evaluate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.tatqa_natural_pairs import (  # noqa: E402
    METRIC_RELATEDNESS_JACCARD_THRESHOLD,
    PAIR_TYPES,
    PROTOCOL_VERSION,
    EligibleQuestion,
    TableFact,
    build_derived_pair_record,
    build_metric_candidates_for_table,
    build_pair_record,
    build_period_pairs_for_table,
    classify_pair,
    extract_eligible_question,
    extract_table_facts,
    validate_natural_pairs,
)
from financial_vlm.data.tatqa_source_rerender import flatten_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, required=True, nargs="+", help="One or more tatqa_dataset_*.json files.")
    parser.add_argument(
        "--split",
        required=True,
        nargs="+",
        help="Split label for each --qa-path, same order/length (e.g. test_gold dev).",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh output directory for the frozen benchmark.")
    parser.add_argument("--max-documents", type=int, default=0, help="0 = no limit, applied per input file.")
    parser.add_argument("--audit-sample-seed", type=int, default=20260810)
    parser.add_argument("--audit-sample-size-period", type=int, default=25)
    parser.add_argument("--audit-sample-size-metric-candidate", type=int, default=25)
    return parser.parse_args()


def prepare_output_dir(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_root}. "
            "Use a fresh run directory for reproducibility."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def load_tatqa_json(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of TAT-QA documents in {path}")
    return data


def collect_eligible_questions(
    documents: list[dict[str, Any]],
    split: str,
    max_documents: int,
    stats: Counter[str],
) -> dict[str, list[EligibleQuestion]]:
    """Existing-question (human-authored) eligibility pool, unchanged from Phase 1."""

    eligible_by_table: dict[str, list[EligibleQuestion]] = {}
    for doc_index, doc in enumerate(documents):
        if max_documents and doc_index >= max_documents:
            stats[f"{split}_stopped_at_max_documents"] += 1
            break
        table_uid = str((doc.get("table") or {}).get("uid") or f"doc_{doc_index:06d}")
        grid = (doc.get("table") or {}).get("table") or []
        cells = flatten_grid(grid)
        for question_index, question in enumerate(doc.get("questions") or []):
            stats["total_questions"] += 1
            stats[f"{split}_total_questions"] += 1
            eligible, reason = extract_eligible_question(doc, question, question_index, cells, split)
            stats[reason] += 1
            stats[f"{split}_{reason}"] += 1
            if eligible is None:
                continue
            eligible_by_table.setdefault(table_uid, []).append(eligible)
    return eligible_by_table


def build_human_authored_pairs(
    eligible_by_table: dict[str, list[EligibleQuestion]],
    pair_index_by_type: Counter[str],
    stats: Counter[str],
    split: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for table_uid, questions in sorted(eligible_by_table.items()):
        if len(questions) < 2:
            continue
        ordered = sorted(questions, key=lambda q: q.question_uid)
        for qa, qb in itertools.combinations(ordered, 2):
            stats["human_authored_candidate_pairs"] += 1
            pair_type, rule_or_reason = classify_pair(qa, qb)
            if pair_type is None:
                stats[rule_or_reason] += 1
                stats[f"{split}_{rule_or_reason}"] += 1
                continue
            pair_index_by_type[f"human_authored_{pair_type}"] += 1
            record = build_pair_record(qa, qb, pair_type, rule_or_reason, pair_index_by_type[f"human_authored_{pair_type}"])
            records.append(record)
    return records


def collect_table_facts(
    documents: list[dict[str, Any]],
    split: str,
    max_documents: int,
    stats: Counter[str],
) -> dict[str, list[TableFact]]:
    facts_by_table: dict[str, list[TableFact]] = {}
    for doc_index, doc in enumerate(documents):
        if max_documents and doc_index >= max_documents:
            break
        table_uid = str((doc.get("table") or {}).get("uid") or f"doc_{doc_index:06d}")
        grid = (doc.get("table") or {}).get("table") or []
        cells = flatten_grid(grid)
        facts = extract_table_facts(doc, cells, split, stats=stats)
        if facts:
            facts_by_table[table_uid] = facts
    return facts_by_table


def build_period_derived_records(
    facts_by_table: dict[str, list[TableFact]],
    pair_index: Counter[str],
) -> list[dict[str, Any]]:
    records = []
    for table_uid in sorted(facts_by_table):
        for fact_a, fact_b, rule in build_period_pairs_for_table(facts_by_table[table_uid]):
            pair_index["period_derived"] += 1
            records.append(
                build_derived_pair_record(
                    fact_a, fact_b, "period", rule, pair_index["period_derived"],
                    "period_derived", "tatqa_period_derived",
                )
            )
    return records


def build_metric_candidate_records(
    facts_by_table: dict[str, list[TableFact]],
    pair_index: Counter[str],
    stats: Counter[str],
) -> list[dict[str, Any]]:
    records = []
    for table_uid in sorted(facts_by_table):
        for fact_a, fact_b, matched_via, score in build_metric_candidates_for_table(facts_by_table[table_uid], stats=stats):
            pair_index["metric_candidate"] += 1
            records.append(
                build_derived_pair_record(
                    fact_a, fact_b, "metric", matched_via, pair_index["metric_candidate"],
                    "metric_candidate", "tatqa_metric_candidate",
                    extra={
                        "jaccard_score": score,
                        "matched_via": matched_via,
                        "inclusion_basis": "auto_threshold_v2",
                    },
                )
            )
    return records


def pairs_per_document(records: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter(r["document_id"] for r in records)
    counts = sorted(counter.values())
    return {
        "unique_documents": len(counter),
        "min": counts[0] if counts else 0,
        "max": counts[-1] if counts else 0,
        "mean": round(sum(counts) / len(counts), 3) if counts else None,
        "median": counts[len(counts) // 2] if counts else None,
        "histogram": dict(sorted(Counter(counts).items())),
        "top_contributing_tables": counter.most_common(10),
    }


def sample_for_audit(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if not records:
        return []
    rng = random.Random(seed)
    return rng.sample(records, min(n, len(records)))


def main() -> int:
    args = parse_args()
    if len(args.qa_path) != len(args.split):
        raise ValueError(f"--qa-path ({len(args.qa_path)}) and --split ({len(args.split)}) must have the same length")

    output_root = prepare_output_dir(args.output_root)
    stats: Counter[str] = Counter()
    pair_index: Counter[str] = Counter()

    human_authored_records: list[dict[str, Any]] = []
    period_derived_records: list[dict[str, Any]] = []
    metric_candidate_records: list[dict[str, Any]] = []
    by_split_summary: dict[str, Any] = {}

    for qa_path, split in zip(args.qa_path, args.split):
        documents = load_tatqa_json(qa_path)

        eligible_by_table = collect_eligible_questions(documents, split, args.max_documents, stats)
        split_human_records = build_human_authored_pairs(eligible_by_table, pair_index, stats, split)
        human_authored_records.extend(split_human_records)

        facts_by_table = collect_table_facts(documents, split, args.max_documents, stats)
        split_period_records = build_period_derived_records(facts_by_table, pair_index)
        split_metric_candidates = build_metric_candidate_records(facts_by_table, pair_index, stats)
        period_derived_records.extend(split_period_records)
        metric_candidate_records.extend(split_metric_candidates)

        by_split_summary[split] = {
            "qa_path": str(qa_path),
            "total_questions": stats[f"{split}_total_questions"],
            "total_eligible_questions": sum(len(v) for v in eligible_by_table.values()),
            "human_authored_pairs": len(split_human_records),
            "fact_accepted": sum(len(v) for v in facts_by_table.values()),
            "unique_tables_with_facts": len(facts_by_table),
            "period_derived_pairs": len(split_period_records),
            "metric_candidate_pairs": len(split_metric_candidates),
        }

    # Validate each subset independently (pair_id namespaces are disjoint by
    # construction -- distinct id prefixes per subset -- but validating
    # separately keeps each subset's report self-contained).
    validation_reports = {
        "human_authored": validate_natural_pairs(human_authored_records),
        "period_derived": validate_natural_pairs(period_derived_records),
        "metric_candidate": validate_natural_pairs(metric_candidate_records),
    }

    # --- write the three subsets ---
    human_authored_path = output_root / f"tatqa_natural_pairs_human_authored_{PROTOCOL_VERSION}.jsonl"
    with human_authored_path.open("w", encoding="utf-8") as handle:
        for record in human_authored_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    period_derived_path = output_root / f"tatqa_natural_pairs_period_derived_{PROTOCOL_VERSION}.jsonl"
    with period_derived_path.open("w", encoding="utf-8") as handle:
        for record in period_derived_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metric_candidate_path = output_root / f"tatqa_natural_pairs_metric_candidates_{PROTOCOL_VERSION}.jsonl"
    with metric_candidate_path.open("w", encoding="utf-8") as handle:
        for record in metric_candidate_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    # --- audit sample (for manual/visual review, drawn before any labeling) ---
    audit_sample = {
        "period_derived": sample_for_audit(period_derived_records, args.audit_sample_size_period, args.audit_sample_seed),
        "metric_candidate": sample_for_audit(
            metric_candidate_records, args.audit_sample_size_metric_candidate, args.audit_sample_seed
        ),
    }
    audit_sample_path = output_root / f"manual_audit_sample_{PROTOCOL_VERSION}.json"
    audit_sample_path.write_text(json.dumps(audit_sample, indent=2, sort_keys=True) + "\n")

    # --- construction summary ---
    rejected_by_reason = {
        key: value
        for key, value in sorted(stats.items())
        if (key.startswith("rejected_") or key.startswith("fact_rejected_"))
        and not any(key.startswith(f"{split}_") for split in args.split)
    }

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "qa_paths": [str(p) for p in args.qa_path],
        "splits": list(args.split),
        "subsets": {
            "human_authored": {
                "total_pairs": len(human_authored_records),
                "by_pair_type": {
                    pt: sum(1 for r in human_authored_records if r["pair_type"] == pt) for pt in PAIR_TYPES
                },
                "pairs_per_document": pairs_per_document(human_authored_records),
                "path": str(human_authored_path),
            },
            "period_derived": {
                "total_pairs": len(period_derived_records),
                "pairs_per_document": pairs_per_document(period_derived_records),
                "path": str(period_derived_path),
            },
            "metric_candidate": {
                "total_pairs": len(metric_candidate_records),
                "note": (
                    "Accepted by the v2 automatic threshold gate (jaccard >= "
                    f"{METRIC_RELATEDNESS_JACCARD_THRESHOLD} or basis_qualifier match, footnote-suffix "
                    "duplicates excluded). No per-candidate human or model label."
                ),
                "matched_via_breakdown": dict(
                    Counter(r["matched_via"] for r in metric_candidate_records)
                ),
                "pairs_per_document": pairs_per_document(metric_candidate_records),
                "path": str(metric_candidate_path),
            },
        },
        "by_split": by_split_summary,
        "rejected_by_reason": rejected_by_reason,
        "fact_extraction_totals": {
            "fact_candidate_numeric_cells": stats["fact_candidate_numeric_cells"],
            "fact_accepted": stats["fact_accepted"],
            "fact_rejected_no_metric_or_degenerate": stats["fact_rejected_no_metric_or_degenerate"],
            "fact_rejected_bare_total_row": stats["fact_rejected_bare_total_row"],
            "fact_rejected_no_period": stats["fact_rejected_no_period"],
            "rejected_footnote_suffix_duplicate": stats["rejected_footnote_suffix_duplicate"],
        },
        "audit_sample_path": str(audit_sample_path),
        "notes": [
            "Frozen Strategy C (hybrid) protocol -- see",
            "docs/tatqa-natural-pairs-construction-diagnosis.md for the full diagnosis.",
            "human_authored: existing TAT-QA questions; kept in full including low-relatedness metric",
            "pairs (see relatedness_jaccard/relatedness_matched_via per record), never discarded here.",
            "period_derived: v4 -- table-derived, any two periods within a metric group may pair",
            "(not adjacent-only), capped per (table, metric) group and per table (smallest year-gap",
            "kept first, deterministic) -- see METRIC_GROUP_PERIOD_PAIR_CAP / TABLE_PERIOD_PAIR_CAP",
            "in tatqa_natural_pairs.py. construction_rule distinguishes adjacent from non-adjacent",
            "('skip-period') pairs. document_id/table_id preserved for table-clustered bootstrap CIs.",
            "metric_candidate: v2 -- accepted entirely by the automatic threshold gate (see",
            "protocol_version changelog in financial_vlm.data.tatqa_natural_pairs); no per-candidate",
            "human or model label. v1 (jaccard>=0.34, no footnote-suffix-duplicate filter) required a",
            "manual adjudication pass instead; the user chose the stricter-threshold path after",
            "spot-checking the v1 candidate pool.",
            "No model was run and no model prediction was inspected anywhere in this construction.",
            "Per AGENTS.md, TAT-QA is public real-world evaluation data: eval-only, never for training",
            "or hyperparameter selection.",
        ],
    }
    summary_path = output_root / f"construction_summary_{PROTOCOL_VERSION}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    validation_path = output_root / f"validation_report_{PROTOCOL_VERSION}.json"
    validation_path.write_text(json.dumps(validation_reports, indent=2, sort_keys=True) + "\n")

    print(f"human_authored: {len(human_authored_records)} -> {human_authored_path}")
    print(f"period_derived: {len(period_derived_records)} -> {period_derived_path}")
    print(f"metric_candidate (auto-threshold {PROTOCOL_VERSION}): {len(metric_candidate_records)} -> {metric_candidate_path}")
    print(f"summary={summary_path}")
    print(f"validation={validation_path}")
    print(f"audit_sample={audit_sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
