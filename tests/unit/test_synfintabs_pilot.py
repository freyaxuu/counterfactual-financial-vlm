from __future__ import annotations

import random
import unittest

from financial_vlm.data.synfintabs_pilot import (
    build_evidence_packet,
    build_variant_plans,
    find_header_swap,
    find_irrelevant_cell,
    flatten_table,
    locate_answer_cell,
    make_counterfactual_number,
    plan_to_json,
    rendered_cells_for_plan,
    render_source_table_image,
)


def synthetic_rows():
    return [
        {
            "bbox": [0, 0, 300, 30],
            "cells": [
                {"bbox": [0, 0, 100, 30], "text": "Metric", "label": "stub_header", "words": []},
                {
                    "bbox": [100, 0, 200, 30],
                    "text": "2021",
                    "label": "column_header",
                    "words": [{"bbox": [110, 8, 150, 20], "text": "2021"}],
                },
                {
                    "bbox": [200, 0, 300, 30],
                    "text": "2022",
                    "label": "column_header",
                    "words": [{"bbox": [210, 8, 250, 20], "text": "2022"}],
                },
            ],
        },
        {
            "bbox": [0, 30, 300, 60],
            "cells": [
                {
                    "bbox": [0, 30, 100, 60],
                    "text": "Adjusted EBITDA",
                    "label": "row_header",
                    "words": [{"bbox": [5, 38, 80, 50], "text": "Adjusted"}],
                },
                {
                    "bbox": [100, 30, 200, 60],
                    "text": "149",
                    "label": "data",
                    "words": [{"bbox": [120, 38, 145, 50], "text": "149"}],
                },
                {
                    "bbox": [200, 30, 300, 60],
                    "text": "159",
                    "label": "data",
                    "words": [{"bbox": [220, 38, 245, 50], "text": "159"}],
                },
            ],
        },
        {
            "bbox": [0, 60, 300, 90],
            "cells": [
                {
                    "bbox": [0, 60, 100, 90],
                    "text": "Revenue",
                    "label": "row_header",
                    "words": [{"bbox": [5, 68, 50, 80], "text": "Revenue"}],
                },
                {
                    "bbox": [100, 60, 200, 90],
                    "text": "818",
                    "label": "data",
                    "words": [{"bbox": [120, 68, 145, 80], "text": "818"}],
                },
                {
                    "bbox": [200, 60, 300, 90],
                    "text": "918",
                    "label": "data",
                    "words": [{"bbox": [220, 68, 245, 80], "text": "918"}],
                },
            ],
        },
    ]


class SynFinTabsPilotTests(unittest.TestCase):
    def test_locates_single_cell_answer_span(self) -> None:
        cells, words = flatten_table(synthetic_rows())
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }

        located = locate_answer_cell(question, cells, words)

        self.assertIsNotNone(located)
        assert located is not None
        self.assertEqual(located.answer_cell.cell_id, "r01c02")
        self.assertEqual(located.answer, "159")

    def test_builds_complete_counterfactual_set(self) -> None:
        cells, words = flatten_table(synthetic_rows())
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }
        located = locate_answer_cell(question, cells, words)
        assert located is not None

        plans = build_variant_plans(located, cells, random.Random(7))

        self.assertIsNotNone(plans)
        assert plans is not None
        by_variant = {plan.variant: plan for plan in plans}
        self.assertEqual(
            set(by_variant),
            {"clean", "target_value_replacement", "header_address_swap", "irrelevant_value_replacement"},
        )
        self.assertEqual(by_variant["clean"].answer, "159")
        self.assertEqual(by_variant["target_value_replacement"].evidence_cell.cell_id, "r01c02")
        self.assertNotEqual(by_variant["target_value_replacement"].answer, "159")
        self.assertEqual(
            by_variant["target_value_replacement"].evidence_cell.text,
            by_variant["target_value_replacement"].answer,
        )
        self.assertEqual(by_variant["header_address_swap"].answer, "149")
        self.assertEqual(by_variant["header_address_swap"].evidence_cell.cell_id, "r01c01")
        self.assertEqual(by_variant["irrelevant_value_replacement"].answer, "159")
        self.assertEqual(by_variant["irrelevant_value_replacement"].evidence_cell.cell_id, "r01c02")

    def test_plan_json_records_gold_evidence_and_box_mapping(self) -> None:
        cells, words = flatten_table(synthetic_rows())
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }
        located = locate_answer_cell(question, cells, words)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(7))
        assert plans is not None
        header_swap = {plan.variant: plan for plan in plans}["header_address_swap"]

        payload = plan_to_json(header_swap)

        self.assertNotIn("gold_evidence_id", payload)
        self.assertNotIn("gold_evidence_bbox", payload)
        self.assertEqual(payload["gold_evidence_ids"], ["r01c01"])
        self.assertEqual(payload["source_evidence_cell"]["cell_id"], "r01c01")
        self.assertEqual(
            payload["box_mapping"]["source_to_rendered"]["r01c01"]["rendered_cell_id"],
            "r01c01",
        )
        self.assertEqual(payload["box_mapping"]["source_to_rendered"]["r01c01"]["bbox"], [100, 30, 200, 60])
        self.assertEqual(payload["box_mapping"]["source_to_rendered"]["r01c01"]["rendered_text"], "149")

    def test_rendered_cells_apply_source_level_text_replacement(self) -> None:
        cells, words = flatten_table(synthetic_rows())
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }
        located = locate_answer_cell(question, cells, words)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(7))
        assert plans is not None
        target_value = {plan.variant: plan for plan in plans}["target_value_replacement"]

        rendered = {
            cell.cell_id: cell.text
            for cell in rendered_cells_for_plan(cells, target_value)
        }

        self.assertEqual(rendered["r01c02"], target_value.answer)
        self.assertEqual(rendered["r01c01"], "149")

    def test_counterfactual_number_preserves_basic_format(self) -> None:
        rng = random.Random(3)

        self.assertRegex(make_counterfactual_number("1,590", rng), r"^\d,\d{3}$")
        self.assertRegex(make_counterfactual_number("$159.00", rng), r"^\$\d+\.\d{2}$")

    def test_builds_evidence_packet(self) -> None:
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
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }
        located = locate_answer_cell(question, cells, words)
        assert located is not None

        packet = build_evidence_packet(cells, located.answer_cell)

        self.assertEqual(packet.value_cell.cell_id, "r02c02")
        self.assertEqual([cell.text for cell in packet.row_headers], ["Adjusted EBITDA"])
        self.assertEqual([cell.text for cell in packet.column_headers], ["2022"])
        self.assertIn("USD millions", [cell.text for cell in packet.unit_cells])

    def test_header_swap_skips_unit_row(self) -> None:
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
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }
        located = locate_answer_cell(question, cells, words)
        assert located is not None

        swap = find_header_swap(cells, located.answer_cell)

        self.assertIsNotNone(swap)
        assert swap is not None
        target_header, peer_header, peer_answer_cell = swap
        self.assertEqual(target_header.text, "2022")
        self.assertEqual(peer_header.text, "2021")
        self.assertEqual(peer_answer_cell.text, "149")

    def test_counterfactual_number_delta_is_magnitude_relative_and_signed(self) -> None:
        deltas = []
        signs = set()
        for seed in range(300):
            rng = random.Random(seed)
            result = make_counterfactual_number("1000", rng)
            new_value = float(result)
            delta = new_value - 1000.0
            self.assertNotEqual(delta, 0.0)
            deltas.append(abs(delta))
            signs.add(delta > 0)

        self.assertEqual(signs, {True, False})
        for delta in deltas:
            self.assertGreaterEqual(delta, 1000.0 * 0.05 - 1)
            self.assertLessEqual(delta, 1000.0 * 0.35 + 1)

    def test_counterfactual_number_zero_magnitude_stays_positive(self) -> None:
        for seed in range(20):
            result = make_counterfactual_number("0", random.Random(seed))
            self.assertGreater(float(result), 0.0)

    def test_counterfactual_number_raises_on_unparsable_input(self) -> None:
        with self.assertRaises(ValueError):
            make_counterfactual_number("not a number", random.Random(1))

    def test_find_irrelevant_cell_stratifies_near_and_far(self) -> None:
        from financial_vlm.data.synfintabs_pilot import CellRef

        answer_cell = CellRef(row_idx=5, col_idx=0, text="500", bbox=(0, 0, 10, 10), label="data", word_indices=())
        candidates = [
            CellRef(row_idx=row_idx, col_idx=0, text=str(100 + row_idx), bbox=(0, 0, 10, 10), label="data", word_indices=())
            for row_idx in (0, 1, 2, 8, 9, 10)
        ]
        chosen_row_indices = {
            find_irrelevant_cell(candidates + [answer_cell], answer_cell, forbidden=set(), rng=random.Random(seed)).row_idx
            for seed in range(200)
        }

        # Sorted by |row - 5|: row2/row8 (d=3), row1/row9 (d=4), row0/row10 (d=5);
        # split at the midpoint into near=[row2,row8,row1] / far=[row9,row0,row10].
        near_rows = {1, 2, 8}
        far_rows = {0, 9, 10}
        self.assertTrue(chosen_row_indices & near_rows, "near candidates were never chosen")
        self.assertTrue(chosen_row_indices & far_rows, "far candidates were never chosen")

    def test_find_header_swap_requires_matching_spanning_header(self) -> None:
        rows = [
            {
                "bbox": [0, 0, 400, 20],
                "cells": [
                    {"bbox": [0, 0, 100, 20], "text": "", "label": "data", "words": []},
                    {"bbox": [100, 0, 200, 20], "text": "Q1", "label": "column_header", "words": []},
                    {"bbox": [200, 0, 300, 20], "text": "Q2", "label": "column_header", "words": []},
                    {"bbox": [300, 0, 400, 20], "text": "Q1", "label": "column_header", "words": []},
                ],
            },
            {
                "bbox": [0, 20, 400, 40],
                "cells": [
                    {"bbox": [0, 20, 100, 40], "text": "Metric", "label": "stub_header", "words": []},
                    {"bbox": [100, 20, 200, 40], "text": "2021", "label": "column_header", "words": []},
                    {"bbox": [200, 20, 300, 40], "text": "2022", "label": "column_header", "words": []},
                    {"bbox": [300, 20, 400, 40], "text": "2023", "label": "column_header", "words": []},
                ],
            },
            {
                "bbox": [0, 40, 400, 60],
                "cells": [
                    {"bbox": [0, 40, 100, 60], "text": "Metric", "label": "row_header", "words": []},
                    {"bbox": [100, 40, 200, 60], "text": "100", "label": "data", "words": []},
                    {"bbox": [200, 40, 300, 60], "text": "200", "label": "data", "words": []},
                    {"bbox": [300, 40, 400, 60], "text": "300", "label": "data", "words": []},
                ],
            },
        ]
        cells, words = flatten_table(rows)
        answer_cell = next(cell for cell in cells if cell.cell_id == "r02c01")

        swap = find_header_swap(cells, answer_cell)

        self.assertIsNotNone(swap)
        assert swap is not None
        target_header, peer_header, peer_answer_cell = swap
        self.assertEqual(target_header.text, "2021")
        # col_idx 2 ("2022") shares no parent with col_idx 1 ("Q1" vs "Q2") and must be skipped;
        # col_idx 3 ("2023") shares the same "Q1" parent and is the only valid peer.
        self.assertEqual(peer_header.text, "2023")
        self.assertEqual(peer_answer_cell.text, "300")

    def test_source_table_rerender_preserves_size_and_applies_patch(self) -> None:
        try:
            from PIL import ImageChops
        except ImportError:
            self.skipTest("Pillow is not installed")

        cells, words = flatten_table(synthetic_rows())
        question = {
            "id": "q1",
            "question": "What was Adjusted EBITDA in 2022?",
            "answer": "159",
            "answer_span": {"start": 4, "end": 5},
        }
        located = locate_answer_cell(question, cells, words)
        assert located is not None
        plans = build_variant_plans(located, cells, random.Random(7))
        assert plans is not None
        by_variant = {plan.variant: plan for plan in plans}

        clean = render_source_table_image(cells, [0, 0, 300, 90], by_variant["clean"])
        target_value = render_source_table_image(
            cells,
            [0, 0, 300, 90],
            by_variant["target_value_replacement"],
        )

        self.assertEqual(clean.size, (300, 90))
        self.assertEqual(target_value.size, (300, 90))
        self.assertIsNotNone(ImageChops.difference(clean, target_value).getbbox())


if __name__ == "__main__":
    unittest.main()
