from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from financial_vlm.training.compute_matched_clean_sampler import (
    load_clean_pair_groups,
    materialize_compute_matched_pairs,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def make_train_manifest(group_ids: list[str]) -> list[dict]:
    return [
        {
            "group_id": group_id,
            "question": f"What was revenue for {group_id}?",
            "variants": {
                "clean": {
                    "image_id": f"{group_id}__clean",
                    "image_path": f"images/train/{group_id}__clean.png",
                    "answer": {"raw": "100"},
                },
            },
        }
        for group_id in group_ids
    ]


def make_clean_b_manifest(group_ids: list[str]) -> list[dict]:
    return [
        {"group_id": group_id, "image_path": f"clean_b/{group_id}__clean_b.png"}
        for group_id in group_ids
    ]


class ComputeMatchedCleanSamplerTests(unittest.TestCase):
    def test_load_clean_pair_groups_joins_train_and_clean_b_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            write_jsonl(train_path, make_train_manifest(["g1", "g2"]))
            write_jsonl(clean_b_path, make_clean_b_manifest(["g1", "g2"]))

            groups = load_clean_pair_groups(train_path, clean_b_path)

        self.assertEqual(set(groups), {"g1", "g2"})
        self.assertEqual(groups["g1"].clean_a["image"], "images/train/g1__clean.png")
        self.assertEqual(groups["g1"].clean_b["image"], "clean_b/g1__clean_b.png")
        self.assertEqual(groups["g1"].clean_a["gold_answer"], "100")
        self.assertEqual(groups["g1"].clean_b["gold_answer"], "100")
        self.assertEqual(groups["g1"].clean_a["question"], groups["g1"].clean_b["question"])

    def test_load_clean_pair_groups_raises_for_missing_clean_b(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            write_jsonl(train_path, make_train_manifest(["g1", "g2"]))
            write_jsonl(clean_b_path, make_clean_b_manifest(["g1"]))  # g2 missing

            with self.assertRaises(ValueError) as ctx:
                load_clean_pair_groups(train_path, clean_b_path)
            self.assertIn("g2", str(ctx.exception))

    def test_materialize_visits_each_group_once_per_cycle_with_fixed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            group_ids = ["g1", "g2", "g3"]
            write_jsonl(train_path, make_train_manifest(group_ids))
            write_jsonl(clean_b_path, make_clean_b_manifest(group_ids))
            groups = load_clean_pair_groups(train_path, clean_b_path)

        pairs = materialize_compute_matched_pairs(groups, global_seed=7, num_cycles=1)

        self.assertEqual(len(pairs), 3)
        seen = {pair["group_id"] for pair in pairs}
        self.assertEqual(seen, set(group_ids))
        for pair in pairs:
            group_id = pair["group_id"]
            self.assertEqual(pair["clean_a"]["image"], f"images/train/{group_id}__clean.png")
            self.assertEqual(pair["clean_b"]["image"], f"clean_b/{group_id}__clean_b.png")

    def test_materialize_two_cycles_gives_two_visits_per_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            group_ids = ["g1", "g2"]
            write_jsonl(train_path, make_train_manifest(group_ids))
            write_jsonl(clean_b_path, make_clean_b_manifest(group_ids))
            groups = load_clean_pair_groups(train_path, clean_b_path)

        pairs = materialize_compute_matched_pairs(groups, global_seed=7, num_cycles=2)

        self.assertEqual(len(pairs), 4)
        by_group: dict[str, int] = {}
        for pair in pairs:
            by_group[pair["group_id"]] = by_group.get(pair["group_id"], 0) + 1
        self.assertEqual(by_group, {"g1": 2, "g2": 2})

    def test_materialize_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            group_ids = [f"g{i}" for i in range(10)]
            write_jsonl(train_path, make_train_manifest(group_ids))
            write_jsonl(clean_b_path, make_clean_b_manifest(group_ids))
            groups = load_clean_pair_groups(train_path, clean_b_path)

        pairs_a = materialize_compute_matched_pairs(groups, global_seed=42, num_cycles=1)
        pairs_b = materialize_compute_matched_pairs(groups, global_seed=42, num_cycles=1)

        self.assertEqual([p["group_id"] for p in pairs_a], [p["group_id"] for p in pairs_b])

    def test_materialize_shuffles_group_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            group_ids = [f"g{i}" for i in range(10)]
            write_jsonl(train_path, make_train_manifest(group_ids))
            write_jsonl(clean_b_path, make_clean_b_manifest(group_ids))
            groups = load_clean_pair_groups(train_path, clean_b_path)

        pairs = materialize_compute_matched_pairs(groups, global_seed=42, num_cycles=1)

        self.assertNotEqual([p["group_id"] for p in pairs], sorted(group_ids))

    def test_materialize_rejects_non_positive_num_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "manifest_train.jsonl"
            clean_b_path = Path(tmp) / "clean_b_manifest.jsonl"
            write_jsonl(train_path, make_train_manifest(["g1"]))
            write_jsonl(clean_b_path, make_clean_b_manifest(["g1"]))
            groups = load_clean_pair_groups(train_path, clean_b_path)

        with self.assertRaises(ValueError):
            materialize_compute_matched_pairs(groups, global_seed=1, num_cycles=0)


if __name__ == "__main__":
    unittest.main()
