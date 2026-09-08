#!/usr/bin/env python3
"""Resolve one clean, unedited table image per unique table_uid referenced by
the frozen TAT-QA Natural Pair benchmark.

Reuses a pre-rendered "clean" PNG from the existing CF group manifest
(canonical_group_cf_v1, e.g. tatqa_source_rerender_test_test_gold/
manifest.jsonl) where one exists for a given table, since that PNG is what
the main clean/TVFR/HFR/ISR/CGS evaluation already used. For every other
table (most of them -- the CF-eligible pool is a narrow subset of all
tables with natural-pair facts), a fresh image is rendered with
``render_grid_image`` using the exact same call signature `render_variant_image`
uses for its "clean" plan (no patches, no highlighted cells, default
col_padding/row_padding) -- verified by inspecting
``financial_vlm.data.tatqa_source_rerender.render_variant_image``: for the
clean plan, ``patches=()`` so ``rendered_cells_for_plan`` is a no-op and
``highlighted_cell_ids`` is empty, making it byte-identical to
``render_grid_image(cells, nrows, ncols)``.

No table content is edited. This script only decides which image path to
use for inference -- it does not touch benchmark construction.
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

from financial_vlm.data.tatqa_natural_pairs import PROTOCOL_VERSION  # noqa: E402
from financial_vlm.data.tatqa_source_rerender import flatten_grid, render_grid_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-pairs-root", type=Path, required=True, help="outputs/tatqa_natural_pairs_v2_test_gold_only")
    parser.add_argument("--tatqa-json", type=Path, required=True, help="tatqa_dataset_test_gold.json")
    parser.add_argument("--split", required=True, help="Split label matching the natural-pairs records (e.g. test_gold).")
    parser.add_argument(
        "--existing-cf-manifest",
        type=Path,
        required=True,
        help="canonical_group_cf_v1 manifest.jsonl (e.g. tatqa_source_rerender_test_test_gold/manifest.jsonl) to reuse clean images from.",
    )
    parser.add_argument(
        "--existing-cf-data-root",
        type=Path,
        required=True,
        help="Root the existing manifest's image_path values are relative to (so paths resolve to real files).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory: rendered/ + image_manifest.jsonl.")
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION, help="Frozen benchmark filename suffix (default: current library PROTOCOL_VERSION).")
    return parser.parse_args()


def collect_table_uids(natural_pairs_root: Path, split: str, protocol_version: str) -> set[str]:
    table_uids: set[str] = set()
    for name in (
        f"tatqa_natural_pairs_human_authored_{protocol_version}.jsonl",
        f"tatqa_natural_pairs_period_derived_{protocol_version}.jsonl",
        f"tatqa_natural_pairs_metric_candidates_{protocol_version}.jsonl",
    ):
        path = natural_pairs_root / name
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["split"] != split:
                    continue
                table_uids.add(record["table_id"])
    return table_uids


def existing_clean_image_by_document_id(manifest_path: Path, data_root: Path) -> dict[str, Path]:
    by_document_id: dict[str, Path] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            document_id = record.get("document_id")
            clean = (record.get("variants") or {}).get("clean")
            if not document_id or not clean:
                continue
            if document_id in by_document_id:
                continue
            by_document_id[document_id] = data_root / clean["image_path"]
    return by_document_id


def load_tatqa_json(path: Path) -> dict[str, dict[str, Any]]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        docs = json.load(handle)
    by_table_uid = {}
    for doc in docs:
        table_uid = str((doc.get("table") or {}).get("uid") or "")
        if table_uid:
            by_table_uid[table_uid] = doc
    return by_table_uid


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    rendered_dir = output_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)

    table_uids = collect_table_uids(args.natural_pairs_root, args.split, args.protocol_version)
    existing = existing_clean_image_by_document_id(args.existing_cf_manifest, args.existing_cf_data_root)
    docs_by_table_uid = load_tatqa_json(args.tatqa_json)

    manifest_path = output_dir / "image_manifest.jsonl"
    reused = 0
    rendered_count = 0
    missing = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for table_uid in sorted(table_uids):
            # document_id in the existing CF manifest is safe_slug(table_uid);
            # for these TAT-QA uids (already alnum/hex) that equals table_uid.
            existing_path = existing.get(table_uid)
            if existing_path is not None and existing_path.is_file():
                handle.write(json.dumps({"table_uid": table_uid, "image_path": str(existing_path), "source": "existing_cf_manifest"}) + "\n")
                reused += 1
                continue

            doc = docs_by_table_uid.get(table_uid)
            if doc is None:
                handle.write(json.dumps({"table_uid": table_uid, "image_path": None, "source": "missing_from_tatqa_json"}) + "\n")
                missing += 1
                continue

            grid = (doc.get("table") or {}).get("table") or []
            cells = flatten_grid(grid)
            nrows = len(grid)
            ncols = max((len(row) for row in grid), default=0)
            image, _bboxes = render_grid_image(cells, nrows, ncols)
            image_path = rendered_dir / f"{table_uid}.png"
            image.save(image_path)
            handle.write(json.dumps({"table_uid": table_uid, "image_path": str(image_path), "source": "rendered_fresh"}) + "\n")
            rendered_count += 1

    print(f"unique tables: {len(table_uids)}")
    print(f"reused from existing CF manifest: {reused}")
    print(f"rendered fresh: {rendered_count}")
    print(f"missing (not found in --tatqa-json): {missing}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
