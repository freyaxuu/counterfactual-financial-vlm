from __future__ import annotations

import random
import unittest

from financial_vlm.data.tatqa_source_rerender import (
    build_evidence_packet,
    build_variant_plans,
    find_header_swap,
    find_irrelevant_cell,
    flatten_grid,
    locate_answer_cell,
    patched_text_by_cell_id,
    render_grid_image,
    render_variant_image,
    rendered_cells_for_plan,
)


def synthetic_grid():
    return [
        ["", "Years Ended December 31,", ""],
        ["", "2019", "2018"],
        ["Revenue", "1,234", "1,100"],
        ["Operating costs", "500", "480"],
    ]


class TatQaSourceRerenderTests(unittest.TestCase):
    def test_flatten_grid_assigns_row_col_and_cell_ids(self) -> None:
        cells = flatten_grid(synthetic_grid())

        self.assertEqual(len(cells), 12)
        revenue_cell = next(cell for cell in cells if cell.text == "Revenue")
        self.assertEqual((revenue_cell.row_idx, revenue_cell.col_idx), (2, 0))
        self.assertEqual(revenue_cell.cell_id, "r02c00")

    def test_locate_answer_cell_unique_match(self) -> None:
        cells = flatten_grid(synthetic_grid())
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}

        located = locate_answer_cell(question, cells)

        self.assertIsNotNone(located)
        assert located is not None
        self.assertEqual((located.answer_cell.row_idx, located.answer_cell.col_idx), (2, 1))

    def test_locate_answer_cell_returns_none_when_missing(self) -> None:
        cells = flatten_grid(synthetic_grid())
        question = {"uid": "q1", "question": "What was the 2019 margin?", "answer": "999,999"}

        self.assertIsNone(locate_answer_cell(question, cells))

    def test_locate_answer_cell_returns_none_when_ambiguous(self) -> None:
        grid = synthetic_grid()
        grid.append(["Duplicate", "1,234", "0"])
        cells = flatten_grid(grid)
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}

        self.assertIsNone(locate_answer_cell(question, cells))

    def test_find_header_swap_locates_peer_year_column(self) -> None:
        cells = flatten_grid(synthetic_grid())
        answer_cell = next(cell for cell in cells if cell.text == "1,234")

        result = find_header_swap(cells, answer_cell)

        self.assertIsNotNone(result)
        assert result is not None
        header, peer_header, peer_value = result
        self.assertEqual(header.text, "2019")
        self.assertEqual(peer_header.text, "2018")
        self.assertEqual(peer_value.text, "1,100")

    def test_find_irrelevant_cell_excludes_forbidden_and_prefers_nearest(self) -> None:
        cells = flatten_grid(synthetic_grid())
        answer_cell = next(cell for cell in cells if cell.text == "1,234")

        irrelevant = find_irrelevant_cell(cells, answer_cell, forbidden={(answer_cell.row_idx, answer_cell.col_idx)})

        self.assertIsNotNone(irrelevant)
        assert irrelevant is not None
        self.assertEqual(irrelevant.text, "500")

    def test_build_variant_plans_produces_four_variants(self) -> None:
        cells = flatten_grid(synthetic_grid())
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None

        plans = build_variant_plans(located, cells, random.Random(0))

        self.assertIsNotNone(plans)
        assert plans is not None
        variant_names = {plan.variant for plan in plans}
        self.assertEqual(
            variant_names,
            {"clean", "target_value_replacement", "header_address_swap", "irrelevant_value_replacement"},
        )
        header_swap_plan = next(plan for plan in plans if plan.variant == "header_address_swap")
        self.assertEqual(header_swap_plan.answer, "1,100")
        target_plan = next(plan for plan in plans if plan.variant == "target_value_replacement")
        self.assertNotEqual(target_plan.answer, "1,234")

    def test_build_variant_plans_returns_none_without_year_header(self) -> None:
        grid = [["Revenue", "1,234"]]
        cells = flatten_grid(grid)
        question = {"uid": "q1", "question": "What was revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None

        self.assertIsNone(build_variant_plans(located, cells, random.Random(0)))

    def test_render_variant_image_matches_grid_dimensions_and_reflects_patches(self) -> None:
        grid = synthetic_grid()
        cells = flatten_grid(grid)
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(0))
        assert plans is not None

        clean_plan = plans[0]
        target_plan = next(plan for plan in plans if plan.variant == "target_value_replacement")

        clean_image, clean_bboxes, _ = render_variant_image(cells, len(grid), len(grid[0]), clean_plan)
        cf_image, cf_bboxes, cf_cells = render_variant_image(cells, len(grid), len(grid[0]), target_plan)

        self.assertEqual(clean_image.size, cf_image.size)
        self.assertGreater(clean_image.width, 0)
        self.assertGreater(clean_image.height, 0)
        self.assertNotEqual(list(clean_image.getdata()), list(cf_image.getdata()))
        self.assertEqual(set(clean_bboxes), set(cf_bboxes))
        patched_cell = next(cell for cell in cf_cells if cell.cell_id == "r02c01")
        self.assertEqual(patched_cell.text, target_plan.answer)

    def test_render_variant_image_highlight_changed_toggle(self) -> None:
        grid = synthetic_grid()
        cells = flatten_grid(grid)
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(0))
        assert plans is not None
        swap_plan = next(plan for plan in plans if plan.variant == "header_address_swap")

        highlight_rgb = (255, 246, 214)
        highlighted, _, _ = render_variant_image(cells, len(grid), len(grid[0]), swap_plan)
        plain, _, _ = render_variant_image(
            cells, len(grid), len(grid[0]), swap_plan, highlight_changed=False
        )

        self.assertIn(highlight_rgb, set(highlighted.getdata()))
        self.assertNotIn(highlight_rgb, set(plain.getdata()))

    def test_patched_text_by_cell_id_reflects_patches(self) -> None:
        cells = flatten_grid(synthetic_grid())
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(0))
        assert plans is not None

        target_plan = next(plan for plan in plans if plan.variant == "target_value_replacement")
        patched = patched_text_by_cell_id(target_plan)

        self.assertEqual(patched, {"r02c01": target_plan.answer})

    def test_render_grid_image_is_deterministic_for_same_inputs(self) -> None:
        cells = flatten_grid(synthetic_grid())
        first, first_bboxes = render_grid_image(cells, 4, 3)
        second, second_bboxes = render_grid_image(cells, 4, 3)

        self.assertEqual(list(first.getdata()), list(second.getdata()))
        self.assertEqual(first_bboxes, second_bboxes)

    def test_rendered_cells_for_plan_applies_header_swap_text(self) -> None:
        cells = flatten_grid(synthetic_grid())
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(0))
        assert plans is not None
        swap_plan = next(plan for plan in plans if plan.variant == "header_address_swap")

        rendered = rendered_cells_for_plan(cells, swap_plan)

        header_cell = next(cell for cell in rendered if cell.cell_id == "r01c01")
        peer_cell = next(cell for cell in rendered if cell.cell_id == "r01c02")
        self.assertEqual(header_cell.text, "2018")
        self.assertEqual(peer_cell.text, "2019")

    def test_build_evidence_packet_classifies_header_context(self) -> None:
        cells = flatten_grid(synthetic_grid())
        value_cell = next(cell for cell in cells if cell.text == "1,234")

        packet = build_evidence_packet(cells, value_cell)

        self.assertEqual([cell.text for cell in packet.row_headers], ["Revenue"])
        self.assertEqual([cell.text for cell in packet.column_headers], ["2019"])
        self.assertEqual([cell.text for cell in packet.spanning_headers], ["Years Ended December 31,"])
        self.assertEqual(packet.unit_cells, ())

    def test_build_evidence_packet_after_header_swap_reflects_swapped_header(self) -> None:
        cells = flatten_grid(synthetic_grid())
        question = {"uid": "q1", "question": "What was the 2019 revenue?", "answer": "1,234"}
        located = locate_answer_cell(question, cells)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(0))
        assert plans is not None
        swap_plan = next(plan for plan in plans if plan.variant == "header_address_swap")

        rendered = rendered_cells_for_plan(cells, swap_plan)
        packet = build_evidence_packet(rendered, swap_plan.evidence_cell)

        self.assertEqual([cell.text for cell in packet.column_headers], ["2019"])
        self.assertEqual(packet.value_cell.text, "1,100")


if __name__ == "__main__":
    unittest.main()
