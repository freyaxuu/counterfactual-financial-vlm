from __future__ import annotations

import random
import unittest

from financial_vlm.data.synfintabs_loader import build_synfintabs_clean_record, build_synfintabs_group_record
from financial_vlm.data.synfintabs_pilot import (
    build_evidence_packet,
    build_variant_plans,
    flatten_table,
    locate_answer_cell,
)
from tests.unit.test_synfintabs_pilot import synthetic_rows


class SynFinTabsLoaderTests(unittest.TestCase):
    def test_builds_canonical_clean_record(self) -> None:
        rows = synthetic_rows()
        rows.insert(
            1,
            {
                "bbox": [0, 30, 300, 60],
                "cells": [
                    {"bbox": [0, 30, 100, 60], "text": "", "label": "data", "words": []},
                    {"bbox": [100, 30, 200, 60], "text": "USD millions", "label": "data", "words": []},
                    {"bbox": [200, 30, 300, 60], "text": "USD millions", "label": "data", "words": []},
                ],
            },
        )
        cells, words = flatten_table(rows)
        located = locate_answer_cell(
            {
                "id": "q04",
                "question": "What was Adjusted EBITDA in 2022?",
                "answer": "159",
                "answer_span": {"start": 4, "end": 5},
            },
            cells,
            words,
        )
        assert located is not None
        packet = build_evidence_packet(cells, located.answer_cell)

        record = build_synfintabs_clean_record(
            table={"id": "table-1", "theme": "theme_03"},
            source_index=123,
            question_index=4,
            split="train",
            image_path="images/syn_000123.png",
            image_size=(300, 120),
            located=located,
            evidence_packet=packet,
        )

        self.assertEqual(record["group_id"], "syn_000123_q04")
        self.assertEqual(record["document_id"], "table-1")
        self.assertEqual(record["template_family"], "theme_03")
        self.assertEqual(record["answer"]["normalised_value"], 159.0)
        self.assertEqual(record["answer"]["metric"], "Adjusted EBITDA")
        self.assertEqual(record["answer"]["period"], "2022")
        self.assertEqual(record["answer"]["unit"], "USD")
        self.assertEqual(record["answer"]["scale"], "million")
        target_id = record["evidence"]["target_id"]
        unit_ids = {unit["id"] for unit in record["evidence_units"]}
        self.assertIn(target_id, unit_ids)
        self.assertEqual(record["evidence"]["row_header_ids"], ["cell_2_0"])
        self.assertEqual(record["evidence"]["column_header_ids"], ["cell_0_2"])
        self.assertEqual(record["evidence"]["unit_region_ids"], ["cell_1_2"])

    def test_builds_group_record_with_all_variants(self) -> None:
        cells, words = flatten_table(synthetic_rows())
        located = locate_answer_cell(
            {
                "id": "q1",
                "question": "What was Adjusted EBITDA in 2022?",
                "answer": "159",
                "answer_span": {"start": 4, "end": 5},
            },
            cells,
            words,
        )
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(7))
        assert plans is not None

        record = build_synfintabs_group_record(
            table={"id": "table-1", "theme": "5"},
            source_index=0,
            question_index=0,
            split="test",
            located=located,
            cells=cells,
            plans=plans,
            variant_image_paths={plan.variant: f"images/test/{plan.variant}.png" for plan in plans},
            variant_image_sizes={plan.variant: (300, 90) for plan in plans},
            seed=20260804,
        )

        self.assertEqual(record["template_family"], "5")
        variants = record["variants"]
        self.assertEqual(
            set(variants),
            {"clean", "target_value_replacement", "header_address_swap", "irrelevant_value_replacement"},
        )

        clean = variants["clean"]
        header_swap = variants["header_address_swap"]
        irrelevant = variants["irrelevant_value_replacement"]

        self.assertNotEqual(header_swap["evidence"]["target_id"], clean["evidence"]["target_id"])
        self.assertEqual(irrelevant["evidence"], clean["evidence"])

        self.assertEqual(clean["changed_cell_ids"], [])
        for variant_name in ("target_value_replacement", "header_address_swap", "irrelevant_value_replacement"):
            self.assertTrue(variants[variant_name]["changed_cell_ids"])

        seeds = {payload["renderer_seed"] for payload in variants.values()}
        self.assertEqual(len(seeds), 4)
        image_ids = {payload["image_id"] for payload in variants.values()}
        self.assertEqual(len(image_ids), 4)

        self.assertEqual(clean["expected_behavior"], "baseline")
        self.assertEqual(header_swap["expected_behavior"], "answer_changes")
        self.assertEqual(irrelevant["expected_behavior"], "answer_invariant")


if __name__ == "__main__":
    unittest.main()
