from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import random

from financial_vlm.data.synfintabs_training import (
    choose_split,
    choose_split_with_template_pool,
    flatten_group_records,
    read_excluded_table_keys,
    sample_epoch_records,
    select_records_by_group_prefix,
    source_table_key,
    split_dev_records_by_group,
)


def group_records(group_id: str) -> list[dict[str, str]]:
    return [
        {"group_id": group_id, "variant": "clean", "answer": "100"},
        {"group_id": group_id, "variant": "target_value", "answer": "111"},
        {"group_id": group_id, "variant": "header_swap", "answer": "90"},
        {"group_id": group_id, "variant": "irrelevant_cell", "answer": "100"},
    ]


class SynFinTabsTrainingTests(unittest.TestCase):
    def test_reads_excluded_table_keys_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "pilot.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps({"source_table_id": "table-a", "source_index": 7}),
                        json.dumps({"source_table_id": "", "source_index": 8}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            excluded = read_excluded_table_keys([manifest])

        self.assertIn(source_table_key("table-a", 7), excluded)
        self.assertIn(source_table_key("", 8), excluded)

    def test_flatten_group_records_merges_shared_and_variant_fields(self) -> None:
        group_record = {
            "group_id": "syn_000001_abc",
            "source": "synfintabs",
            "split": "train",
            "document_id": "doc-1",
            "template_family": "0",
            "question": "What was revenue in 2022?",
            "counterfactual_policy": {"allowed_types": ["target_value_replace"]},
            "renderer_seed_base": 42,
            "variants": {
                "clean": {
                    "variant": "clean",
                    "image_path": "images/train/g1__clean.png",
                    "answer": {"raw": "100"},
                },
                "target_value_replacement": {
                    "variant": "target_value_replacement",
                    "image_path": "images/train/g1__target_value_replacement.png",
                    "answer": {"raw": "111"},
                },
            },
        }

        flattened = flatten_group_records([group_record], ["clean"])

        self.assertEqual(len(flattened), 1)
        record = flattened[0]
        self.assertEqual(record["variant"], "clean")
        self.assertEqual(record["group_id"], "syn_000001_abc")
        self.assertEqual(record["question"], "What was revenue in 2022?")
        self.assertEqual(record["image_path"], "images/train/g1__clean.png")
        self.assertEqual(record["answer"], {"raw": "100"})
        self.assertNotIn("variants", record)

    def test_flatten_group_records_selects_multiple_variants(self) -> None:
        group_record = {
            "group_id": "syn_000001_abc",
            "question": "What was revenue in 2022?",
            "variants": {
                "clean": {"variant": "clean", "answer": {"raw": "100"}},
                "target_value_replacement": {"variant": "target_value_replacement", "answer": {"raw": "111"}},
                "header_address_swap": {"variant": "header_address_swap", "answer": {"raw": "90"}},
            },
        }

        flattened = flatten_group_records([group_record], ["clean", "target_value_replacement"])

        self.assertEqual(
            sorted(record["variant"] for record in flattened),
            ["clean", "target_value_replacement"],
        )

    def test_flatten_group_records_passes_through_flat_records_unchanged(self) -> None:
        flat_record = {"group_id": "g1", "variant": "clean", "answer": {"raw": "100"}}

        flattened = flatten_group_records([flat_record], ["clean"])

        self.assertEqual(flattened, [flat_record])

    def test_epoch_sampler_keeps_clean_and_one_counterfactual_per_group(self) -> None:
        records = [*group_records("g1"), *group_records("g2")]

        sampled = sample_epoch_records(records, epoch=0, seed=123)

        by_group: dict[str, list[str]] = {}
        for record in sampled:
            by_group.setdefault(record["group_id"], []).append(record["variant"])

        self.assertEqual(set(by_group), {"g1", "g2"})
        for variants in by_group.values():
            self.assertEqual(len(variants), 2)
            self.assertIn("clean", variants)
            self.assertEqual(len(set(variants) - {"clean"}), 1)

    def test_epoch_sampler_resamples_across_epochs_deterministically(self) -> None:
        records = group_records("g1")

        first_pass = [
            sample_epoch_records(records, epoch=epoch, seed=123)[1]["epoch_sampled_counterfactual"]
            for epoch in range(10)
        ]
        second_pass = [
            sample_epoch_records(records, epoch=epoch, seed=123)[1]["epoch_sampled_counterfactual"]
            for epoch in range(10)
        ]

        self.assertEqual(first_pass, second_pass)
        self.assertGreater(len(set(first_pass)), 1)

    def test_split_dev_records_by_group_is_deterministic(self) -> None:
        records = [
            {"group_id": "g03", "row": 1},
            {"group_id": "g01", "row": 2},
            {"group_id": "g02", "row": 3},
            {"group_id": "g01", "row": 4},
        ]

        quick, select = split_dev_records_by_group(records, quick_groups=1, select_groups=2)

        self.assertEqual([record["row"] for record in quick], [2, 4])
        self.assertEqual([record["row"] for record in select], [1, 3])

    def test_split_dev_records_requires_enough_groups(self) -> None:
        with self.assertRaises(ValueError):
            split_dev_records_by_group([{"group_id": "g01"}], quick_groups=1, select_groups=1)

    def test_select_records_by_group_prefix_keeps_nested_groups(self) -> None:
        records = [
            {"group_id": "g03", "row": 1},
            {"group_id": "g01", "row": 2},
            {"group_id": "g02", "row": 3},
            {"group_id": "g01", "row": 4},
        ]

        selected = select_records_by_group_prefix(records, group_count=2)

        self.assertEqual([record["row"] for record in selected], [2, 3, 4])

    def test_select_records_by_group_prefix_requires_enough_groups(self) -> None:
        with self.assertRaises(ValueError):
            select_records_by_group_prefix([{"group_id": "g01"}], group_count=2)

    def test_choose_split_with_template_pool_reserves_ood_template_for_test_only(self) -> None:
        ood = frozenset({"theme_ood"})
        remaining = {"train": 5, "dev": 5, "test": 3}
        for seed in range(200):
            result = choose_split_with_template_pool(
                "theme_ood", remaining, random.Random(seed), ood_template_ids=ood
            )
            self.assertIn(result, {"test", None})

    def test_choose_split_with_template_pool_blocks_non_ood_from_test(self) -> None:
        ood = frozenset({"theme_ood"})
        remaining = {"train": 0, "dev": 0, "test": 10}
        for seed in range(200):
            result = choose_split_with_template_pool(
                "theme_in_distribution", remaining, random.Random(seed), ood_template_ids=ood
            )
            self.assertIsNone(result)

    def test_choose_split_with_template_pool_isolates_exhausted_pool(self) -> None:
        ood = frozenset({"theme_ood"})
        remaining = {"train": 5, "dev": 5, "test": 0}
        for seed in range(200):
            result = choose_split_with_template_pool(
                "theme_ood", remaining, random.Random(seed), ood_template_ids=ood
            )
            self.assertIsNone(result)

    def test_choose_split_with_template_pool_matches_choose_split_when_unconfigured(self) -> None:
        remaining = {"train": 5, "dev": 5}
        for seed in range(200):
            for family in ("theme_0", "theme_1", "unknown"):
                pooled = choose_split_with_template_pool(
                    family, remaining, random.Random(seed), ood_template_ids=frozenset()
                )
                plain = choose_split(remaining, random.Random(seed))
                self.assertEqual(pooled, plain)


if __name__ == "__main__":
    unittest.main()
