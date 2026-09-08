from __future__ import annotations

import random
import unittest

from financial_vlm.data.tatqa_loader import build_tatqa_group_record
from financial_vlm.data.tatqa_source_rerender import (
    build_variant_plans,
    flatten_grid,
    locate_answer_cell,
    render_variant_image,
)


def synthetic_grid():
    return [
        ["", "Years Ended December 31,", ""],
        ["", "2019", "2018"],
        ["Revenue", "1,234", "1,100"],
        ["Operating costs", "500", "480"],
    ]


def build_plans_for_synthetic_grid():
    grid = synthetic_grid()
    cells = flatten_grid(grid)
    question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
    located = locate_answer_cell(question, cells)
    assert located is not None
    plans = build_variant_plans(located, cells, random.Random(0))
    assert plans is not None
    return grid, cells, located, plans


class TatQaLoaderTests(unittest.TestCase):
    def build_record(self):
        grid, cells, located, plans = build_plans_for_synthetic_grid()
        nrows, ncols = len(grid), len(grid[0])

        variant_rendered_cells = {}
        variant_bboxes = {}
        variant_image_sizes = {}
        variant_image_paths = {}
        for plan in plans:
            image, bboxes, rendered_cells = render_variant_image(cells, nrows, ncols, plan)
            variant_rendered_cells[plan.variant] = rendered_cells
            variant_bboxes[plan.variant] = bboxes
            variant_image_sizes[plan.variant] = image.size
            variant_image_paths[plan.variant] = f"images/tbl1_q1_{plan.variant}.png"

        return build_tatqa_group_record(
            table_uid="tbl1",
            question_uid=located.question_id,
            question_index=0,
            question=located.question,
            split="test",
            plans=plans,
            variant_rendered_cells=variant_rendered_cells,
            variant_bboxes=variant_bboxes,
            variant_image_sizes=variant_image_sizes,
            variant_image_paths=variant_image_paths,
            seed=20260807,
        )

    def test_build_group_record_passes_canonical_validation(self) -> None:
        # build_tatqa_group_record calls validate_group_record internally;
        # this just confirms it doesn't raise.
        record = self.build_record()
        self.assertEqual(record["source"], "tatqa")
        self.assertEqual(set(record["variants"]), {
            "clean", "target_value_replacement", "header_address_swap", "irrelevant_value_replacement",
        })

    def test_clean_variant_has_no_changed_cells(self) -> None:
        record = self.build_record()
        clean = record["variants"]["clean"]
        self.assertEqual(clean["changed_cell_ids"], [])
        self.assertEqual(clean["answer"]["raw"], "1,234")
        self.assertEqual(clean["evidence"]["target_id"], "r02c01")

    def test_target_value_replacement_changes_answer_only(self) -> None:
        record = self.build_record()
        variant = record["variants"]["target_value_replacement"]
        self.assertEqual(variant["changed_cell_ids"], ["r02c01"])
        self.assertNotEqual(variant["answer"]["raw"], "1,234")
        self.assertEqual(variant["evidence"]["target_id"], "r02c01")

    def test_header_address_swap_points_at_peer_cell_with_swapped_header(self) -> None:
        record = self.build_record()
        variant = record["variants"]["header_address_swap"]
        self.assertEqual(variant["answer"]["raw"], "1,100")
        self.assertEqual(variant["evidence"]["target_id"], "r02c02")
        column_header_ids = variant["evidence"]["column_header_ids"]
        self.assertEqual(len(column_header_ids), 1)
        header_unit = next(unit for unit in variant["evidence_units"] if unit["id"] == column_header_ids[0])
        self.assertEqual(header_unit["text"], "2019")

    def test_irrelevant_value_replacement_keeps_answer_and_evidence(self) -> None:
        record = self.build_record()
        variant = record["variants"]["irrelevant_value_replacement"]
        self.assertEqual(variant["answer"]["raw"], "1,234")
        self.assertEqual(variant["evidence"]["target_id"], "r02c01")
        self.assertEqual(len(variant["changed_cell_ids"]), 1)
        self.assertNotEqual(variant["changed_cell_ids"][0], "r02c01")

    def test_renderer_seed_base_is_deterministic(self) -> None:
        first = self.build_record()
        second = self.build_record()
        self.assertEqual(first["renderer_seed_base"], second["renderer_seed_base"])
        self.assertEqual(
            first["variants"]["clean"]["renderer_seed"],
            second["variants"]["clean"]["renderer_seed"],
        )


if __name__ == "__main__":
    unittest.main()
