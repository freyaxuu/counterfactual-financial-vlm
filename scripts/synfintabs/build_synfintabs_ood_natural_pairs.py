#!/usr/bin/env python3
"""Build a natural (non-counterfactual) confusion-pair benchmark from
SynFinTabs' held-out OOD template ("5" -- "Company Report" style, never
used in any train/dev split per configs/synfintabs_train_dev_test_ood.yaml).

Each pair is two GENUINELY DIFFERENT, unmodified facts from the SAME raw
table: the same row (metric), two different columns (periods). This reuses
`find_header_swap`'s peer-discovery logic exactly as production training
does (via `build_variant_plans`), but instead of rendering a counterfactual
swapped image, it takes the peer column's TRUE ORIGINAL header/value
(`TextPatch.old_text`, captured before any mutation) and treats it as the
second side of a natural pair -- no synthetic image, no altered cell text,
the same clean render used for training's "clean" variant shows both
periods' genuine values simultaneously.

At most one pair per source table (diversity cap, mirrors
company_confusion_v1's per-document cap). All rejections are counted, never
silently dropped (AGENTS.md).

Output schema matches `company_confusion_v1.jsonl` (pair_id, document_id,
pair_type, gold_answer_a/b, ...) so the existing
`financial_vlm.evaluation.company_confusion_eval` module and the
`aggregate_company_confusion_results_synfintabs.py` pattern can be reused
unchanged, plus a `unique_questions.jsonl` sibling matching
`evaluate_qwen3_vl_natural_pairs.py`'s expected input schema.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.synfintabs_pilot import (  # noqa: E402
    build_evidence_packet,
    build_variant_plans,
    flatten_table,
    locate_answer_cell,
    render_source_table_image,
)
from financial_vlm.data.synfintabs_loader import (  # noqa: E402
    document_id_for_table,
    question_slug,
    template_family_for_table,
    unit_and_scale_from_cells,
)

OOD_TEMPLATE_ID = "5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="ethanbradley/synfintabs")
    parser.add_argument("--revision", default="88dec4fbebe890d66f719eff5964f1c7152a60f5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ood-template-id", default=OOD_TEMPLATE_ID)
    parser.add_argument("--target-pairs", type=int, default=200)
    parser.add_argument("--max-source-tables", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Smoke-test: stop after this many ACCEPTED pairs.")
    return parser.parse_args()


def iter_dataset(dataset_id: str, split: str, revision: str) -> Any:
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split, revision=revision, streaming=True)


def pair_id_for(document_id: str, group_index: int) -> str:
    digest = hashlib.sha256(f"{document_id}:{group_index}".encode("utf-8")).hexdigest()
    return digest[:16]


def substitute_period(question: str, old_period: str, new_period: str) -> str | None:
    if old_period and old_period in question:
        return question.replace(old_period, new_period, 1)
    return None


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    target_pairs = args.limit or args.target_pairs
    rng = random.Random(args.seed)
    stats: Counter[str] = Counter()
    question_b_method: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []
    unique_questions: list[dict[str, Any]] = []

    dataset = iter_dataset(args.dataset_id, args.split, args.revision)

    for source_index, table in enumerate(dataset):
        if len(pairs) >= target_pairs:
            break
        if source_index >= args.max_source_tables:
            stats["stopped_at_max_source_tables"] += 1
            break

        stats["source_tables_seen"] += 1
        if template_family_for_table(table) != args.ood_template_id:
            stats["rejected_wrong_template"] += 1
            continue

        try:
            cells, words = flatten_table(table.get("rows") or [])
        except Exception:
            stats["rejected_malformed_rows"] += 1
            continue

        questions = list(table.get("questions") or [])
        if not questions:
            stats["rejected_no_questions"] += 1
            continue

        accepted_for_table = False
        for question in rng.sample(questions, len(questions)):
            if accepted_for_table:
                break
            stats["questions_seen"] += 1
            located = locate_answer_cell(question, cells, words)
            if located is None:
                stats["rejected_answer_not_single_cell"] += 1
                continue

            plans = build_variant_plans(located, cells, rng)
            if plans is None:
                stats["rejected_no_valid_peer"] += 1
                continue

            clean_plan, _target_value_plan, header_swap_plan, _irrelevant_plan = plans
            target_patch, peer_patch = header_swap_plan.patches
            period_a = target_patch.old_text
            period_b = peer_patch.old_text
            if not period_a or not period_b or period_a == period_b:
                stats["rejected_degenerate_period_pair"] += 1
                continue
            if not any(ch.isdigit() for ch in period_a) or not any(ch.isdigit() for ch in period_b):
                stats["rejected_non_date_period_header"] += 1
                continue

            evidence_packet = build_evidence_packet(cells, located.answer_cell)
            metric = evidence_packet.row_headers[-1].text if evidence_packet.row_headers else None
            if not metric:
                stats["rejected_no_metric_label"] += 1
                continue
            unit, scale = unit_and_scale_from_cells(evidence_packet.unit_cells)

            value_a = located.answer
            value_b = header_swap_plan.answer
            if not value_a or not value_b or value_a == value_b:
                stats["rejected_degenerate_value_pair"] += 1
                continue

            document_id = document_id_for_table(table, source_index)
            group_index = len(pairs)
            pair_id = pair_id_for(document_id, group_index)

            question_b = substitute_period(located.question, period_a, period_b)
            if question_b is not None:
                question_b_source = "substituted"
                question_b_method["substituted"] += 1
            else:
                question_b = f"What was {metric} in {period_b}?"
                question_b_source = "templated"
                question_b_method["templated"] += 1

            try:
                image = render_source_table_image(cells, table["bbox"], clean_plan)
            except Exception:
                stats["rejected_render_failed"] += 1
                continue

            image_name = f"{pair_id}.png"
            image_path = images_dir / image_name
            image.save(image_path)
            image_path_str = str(image_path)

            pairs.append(
                {
                    "pair_id": pair_id,
                    "document_id": document_id,
                    "template_family": template_family_for_table(table),
                    "pair_type": "period",
                    "metric": metric,
                    "unit": unit,
                    "scale": scale,
                    "period_a": period_a,
                    "gold_answer_a": value_a,
                    "bbox_a": list(located.answer_cell.bbox),
                    "question_a": located.question,
                    "period_b": period_b,
                    "gold_answer_b": value_b,
                    "bbox_b": list(header_swap_plan.evidence_cell.bbox),
                    "question_b": question_b,
                    "question_b_source": question_b_source,
                    "image_path": image_path_str,
                }
            )
            for side, question_text, target_value in (("A", located.question, value_a), ("B", question_b, value_b)):
                group_id = f"{pair_id}_{side}"
                unique_questions.append(
                    {
                        "group_id": group_id,
                        "question_key": group_id,
                        "document_id": document_id,
                        "image_path": image_path_str,
                        "question": question_text,
                        "variant": "natural_pair",
                        "answer": {
                            "raw": target_value,
                            "metric": metric,
                            "period": period_a if side == "A" else period_b,
                            "unit": unit,
                            "scale": scale,
                        },
                    }
                )

            accepted_for_table = True
            stats["accepted_pairs"] = len(pairs)

        if not accepted_for_table:
            stats["rejected_no_accepted_pair_for_table"] += 1

    (output_dir / "synfintabs_ood_natural_pairs_v1.jsonl").write_text(
        "\n".join(json.dumps(p, sort_keys=True) for p in pairs) + ("\n" if pairs else "")
    )
    (output_dir / "unique_questions.jsonl").write_text(
        "\n".join(json.dumps(q, sort_keys=True) for q in unique_questions) + ("\n" if unique_questions else "")
    )
    summary = {
        "target_pairs": target_pairs,
        "accepted_pairs": len(pairs),
        "unique_questions_written": len(unique_questions),
        "question_b_construction_method": dict(question_b_method),
        "rejection_and_progress_counts": dict(stats),
        "dataset_id": args.dataset_id,
        "revision": args.revision,
        "ood_template_id": args.ood_template_id,
        "seed": args.seed,
    }
    (output_dir / "construction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
