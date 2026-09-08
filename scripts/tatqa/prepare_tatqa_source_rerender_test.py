#!/usr/bin/env python3
"""Build a canonical-schema TAT-QA source-rerender CF test manifest.

Produces the same 4-variant group-record schema as SynFinTabs
(``validate_group_record``) directly from TAT-QA's structured table grid --
see ``docs/tatqa-source-rerender-cf-feasibility.md`` for the feasibility
check this is built on. Per AGENTS.md dataset roles, this is public
real-world evaluation data: it must never be used for training,
hyperparameter selection, or checkpoint selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.canonical_schema import validate_group_record, write_jsonl_record  # noqa: E402
from financial_vlm.data.tatdqa_pilot import is_direct_extraction_question, prepare_output_dir  # noqa: E402
from financial_vlm.data.tatqa_loader import build_tatqa_group_record, safe_slug  # noqa: E402
from financial_vlm.data.tatqa_source_rerender import (  # noqa: E402
    build_variant_plans,
    flatten_grid,
    locate_answer_cell,
    render_variant_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "tatqa" / "tatqa_source_rerender_test.yaml"
    )
    parser.add_argument("--qa-path", type=Path, help="Override dataset.qa_path.")
    parser.add_argument("--split", type=str, help="Override dataset.split (also used as the split label in records).")
    parser.add_argument("--output-root", type=Path, help="Override output.root_dir.")
    parser.add_argument("--max-questions", type=int, help="Override test_data.max_questions (0 = no limit).")
    parser.add_argument("--max-documents", type=int, help="Override test_data.max_documents (0 = no limit).")
    parser.add_argument("--limit", type=int, help="Smoke-test alias for --max-questions.")
    parser.add_argument(
        "--no-highlight-changed-cells",
        action="store_true",
        help="Do not tint intervened cells in counterfactual variant images. The tint is a "
        "visible cue a clean image lacks (and SynFinTabs' renderer omits); use this for a "
        "render surface free of that confound.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install project dependencies first.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return loaded


def load_tatqa_json(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of TAT-QA documents in {path}")
    return data


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_cfg = config["dataset"]
    test_cfg = config["test_data"]
    output_cfg = config["output"]
    seed = int(test_cfg["seed"])
    split = str(args.split or dataset_cfg.get("split", "dev"))
    max_questions = int(args.limit or args.max_questions or test_cfg.get("max_questions") or 0)
    max_documents = int(args.max_documents or test_cfg.get("max_documents") or 0)

    qa_path = Path(args.qa_path or dataset_cfg["qa_path"]).expanduser().resolve()
    output_root = Path(args.output_root or output_cfg["root_dir"])
    output_root, images_dir = prepare_output_dir(output_root, args.config)
    manifest_path = output_root / output_cfg.get("manifest_file", "manifest.jsonl")
    summary_path = output_root / output_cfg.get("summary_file", "summary.json")

    documents = load_tatqa_json(qa_path)
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    accepted = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for doc_index, doc in enumerate(documents):
            if max_documents and doc_index >= max_documents:
                stats["stopped_at_max_documents"] += 1
                break

            table_uid = str((doc.get("table") or {}).get("uid") or f"doc_{doc_index:06d}")
            grid = (doc.get("table") or {}).get("table") or []
            cells = flatten_grid(grid)
            nrows = len(grid)
            ncols = max((len(row) for row in grid), default=0)

            for question_index, question in enumerate(doc.get("questions") or []):
                if max_questions and accepted >= max_questions:
                    break

                stats["total_questions"] += 1
                answer_from = question.get("answer_from")
                if answer_from not in ("table", "table-text"):
                    stats["rejected_not_table_grounded"] += 1
                    continue

                ok, reason = is_direct_extraction_question(question)
                if not ok:
                    stats[f"rejected_{reason}"] += 1
                    continue

                located = locate_answer_cell(question, cells)
                if located is None:
                    stats["rejected_answer_cell_not_located"] += 1
                    continue

                plans = build_variant_plans(located, cells, rng)
                if plans is None:
                    stats["rejected_no_valid_counterfactual_set"] += 1
                    continue

                group_slug = safe_slug(table_uid)
                question_slug = safe_slug(located.question_id) if located.question_id else f"q{question_index:04d}"

                try:
                    variant_rendered_cells = {}
                    variant_bboxes = {}
                    variant_image_sizes = {}
                    variant_image_paths = {}
                    for plan in plans:
                        image, bboxes, rendered_cells = render_variant_image(
                            cells, nrows, ncols, plan, highlight_changed=not args.no_highlight_changed_cells
                        )
                        image_name = f"{group_slug}_{question_slug}_{plan.variant}.png"
                        image_path = images_dir / image_name
                        image.save(image_path)
                        variant_rendered_cells[plan.variant] = rendered_cells
                        variant_bboxes[plan.variant] = bboxes
                        variant_image_sizes[plan.variant] = image.size
                        variant_image_paths[plan.variant] = str(image_path.relative_to(output_root))

                    record = build_tatqa_group_record(
                        table_uid=table_uid,
                        question_uid=located.question_id,
                        question_index=question_index,
                        question=located.question,
                        split=split,
                        plans=plans,
                        variant_rendered_cells=variant_rendered_cells,
                        variant_bboxes=variant_bboxes,
                        variant_image_sizes=variant_image_sizes,
                        variant_image_paths=variant_image_paths,
                        seed=seed,
                    )
                except Exception:
                    stats["rejected_render_or_validation_failed"] += 1
                    continue

                write_jsonl_record(manifest, record, validator=validate_group_record)
                accepted += 1
                stats["accepted_groups"] = accepted
                if accepted % 25 == 0:
                    print(f"accepted={accepted} questions_seen={stats['total_questions']}", flush=True)

            if max_questions and accepted >= max_questions:
                break

    summary = {
        "qa_path": str(qa_path),
        "seed": seed,
        "split": split,
        "accepted_groups": accepted,
        "manifest_path": str(manifest_path),
        "images_dir": str(images_dir),
        "config_path": str(output_root / "config.yaml"),
        "highlight_changed_cells": not args.no_highlight_changed_cells,
        "stats": dict(sorted(stats.items())),
        "notes": [
            "Render method: source_table_rerender (auto-layout grid, no bbox or original image required).",
            (
                "Intervened cells are tinted in counterfactual variant images."
                if not args.no_highlight_changed_cells
                else "Intervened cells are NOT tinted (--no-highlight-changed-cells)."
            ),
            "Public real-world evaluation data: never use for training, hyperparameter selection, or",
            "checkpoint selection (AGENTS.md dataset roles).",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"summary={summary_path}")
    print(f"accepted_groups={accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
