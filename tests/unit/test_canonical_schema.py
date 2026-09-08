from __future__ import annotations

import unittest

from financial_vlm.data.canonical_schema import (
    normalise_bbox,
    normalise_numeric_value,
    validate_clean_record,
    validate_group_record,
)


def valid_record() -> dict:
    return {
        "group_id": "syn_000123_q04",
        "source": "synfintabs",
        "split": "train",
        "document_id": "syn_000123",
        "template_family": "theme_03",
        "image_path": "images/syn_000123.png",
        "question": "What was adjusted EBITDA in 2022?",
        "answer": {
            "raw": "159",
            "normalised_value": 159.0,
            "unit": "USD",
            "scale": "million",
            "metric": "Adjusted EBITDA",
            "period": "2022",
        },
        "evidence": {
            "type": "cell",
            "target_id": "cell_1_2",
            "bbox_normalised": [0.51, 0.42, 0.61, 0.48],
            "row_header_ids": ["cell_1_0"],
            "column_header_ids": ["cell_0_2"],
            "unit_region_ids": ["cell_0_1"],
        },
        "evidence_units": [
            {
                "id": "cell_1_2",
                "type": "cell",
                "text": "159",
                "bbox_normalised": [0.51, 0.42, 0.61, 0.48],
                "role": "target",
                "metadata": {},
            },
            {
                "id": "cell_1_0",
                "type": "cell",
                "text": "Adjusted EBITDA",
                "bbox_normalised": [0.0, 0.42, 0.3, 0.48],
                "role": "row_header",
                "metadata": {},
            },
            {
                "id": "cell_0_2",
                "type": "cell",
                "text": "2022",
                "bbox_normalised": [0.51, 0.0, 0.61, 0.1],
                "role": "column_header",
                "metadata": {},
            },
            {
                "id": "cell_0_1",
                "type": "cell",
                "text": "USD millions",
                "bbox_normalised": [0.31, 0.0, 0.5, 0.1],
                "role": "unit_region",
                "metadata": {},
            },
        ],
        "counterfactual_policy": {
            "allowed_types": ["target_value_replace", "header_swap", "irrelevant_cell_replace"],
        },
    }


def valid_variant_payload(variant: str, expected_behavior: str, changed_cell_ids: list) -> dict:
    return {
        "variant": variant,
        "intervention_type": "none" if variant == "clean" else "some intervention",
        "image_id": f"g1__{variant}",
        "image_path": f"images/g1__{variant}.png",
        "renderer_seed": 12345,
        "answer": {
            "raw": "159",
            "normalised_value": 159.0,
            "unit": "USD",
            "scale": "million",
            "metric": "Adjusted EBITDA",
            "period": "2022",
        },
        "evidence": {
            "type": "cell",
            "target_id": "cell_1_2",
            "bbox_normalised": [0.51, 0.42, 0.61, 0.48],
            "row_header_ids": ["cell_1_0"],
            "column_header_ids": ["cell_0_2"],
            "unit_region_ids": ["cell_0_1"],
        },
        "evidence_units": [
            {
                "id": "cell_1_2",
                "type": "cell",
                "text": "159",
                "bbox_normalised": [0.51, 0.42, 0.61, 0.48],
                "role": "target",
                "metadata": {},
            },
            {
                "id": "cell_1_0",
                "type": "cell",
                "text": "Adjusted EBITDA",
                "bbox_normalised": [0.0, 0.42, 0.3, 0.48],
                "role": "row_header",
                "metadata": {},
            },
            {
                "id": "cell_0_2",
                "type": "cell",
                "text": "2022",
                "bbox_normalised": [0.51, 0.0, 0.61, 0.1],
                "role": "column_header",
                "metadata": {},
            },
            {
                "id": "cell_0_1",
                "type": "cell",
                "text": "USD millions",
                "bbox_normalised": [0.31, 0.0, 0.5, 0.1],
                "role": "unit_region",
                "metadata": {},
            },
        ],
        "gold_evidence_ids": {
            "target_value": ["cell_1_2"],
            "row_headers": ["cell_1_0"],
            "column_headers": ["cell_0_2"],
            "parent_headers": [],
            "unit_cells": ["cell_0_1"],
        },
        "changed_cell_ids": changed_cell_ids,
        "expected_behavior": expected_behavior,
        "validation_status": "passed",
        "validation_notes": ["note"],
    }


def valid_group_record() -> dict:
    return {
        "group_id": "syn_000123_q04",
        "source": "synfintabs",
        "split": "test",
        "document_id": "syn_000123",
        "template_family": "5",
        "question": "What was adjusted EBITDA in 2022?",
        "counterfactual_policy": {
            "allowed_types": ["target_value_replace", "header_swap", "irrelevant_cell_replace"],
        },
        "renderer_seed_base": 999,
        "variants": {
            "clean": valid_variant_payload("clean", "baseline", []),
            "target_value_replacement": valid_variant_payload(
                "target_value_replacement", "answer_changes", ["cell_1_2"]
            ),
            "header_address_swap": valid_variant_payload(
                "header_address_swap", "answer_changes", ["cell_0_1", "cell_0_2"]
            ),
            "irrelevant_value_replacement": valid_variant_payload(
                "irrelevant_value_replacement", "answer_invariant", ["cell_2_1"]
            ),
        },
    }


class CanonicalSchemaTests(unittest.TestCase):
    def test_normalises_numeric_values(self) -> None:
        self.assertEqual(normalise_numeric_value("1,590"), 1590.0)
        self.assertEqual(normalise_numeric_value("($12.50)"), -12.5)
        self.assertIsNone(normalise_numeric_value("not numeric"))

    def test_normalises_bbox(self) -> None:
        self.assertEqual(normalise_bbox([50, 25, 100, 50], 200, 100), [0.25, 0.25, 0.5, 0.5])

    def test_valid_synfintabs_style_record_passes(self) -> None:
        validate_clean_record(valid_record())

    def test_valid_tatdqa_style_record_with_nullable_fields_passes(self) -> None:
        record = valid_record()
        record["source"] = "tatdqa"
        record["template_family"] = "public_page"
        record["answer"]["unit"] = None
        record["answer"]["scale"] = "million"
        record["answer"]["metric"] = None
        record["answer"]["period"] = None
        record["evidence"] = {
            "type": "region",
            "target_id": "region_0_block_char_12_16",
            "bbox_normalised": [0.2, 0.1, 0.3, 0.2],
            "row_header_ids": [],
            "column_header_ids": [],
            "unit_region_ids": [],
        }
        record["evidence_units"] = [
            {
                "id": "region_0_block_char_12_16",
                "type": "region",
                "text": "1,234",
                "bbox_normalised": [0.2, 0.1, 0.3, 0.2],
                "role": "target",
                "metadata": {},
            }
        ]
        record["counterfactual_policy"] = {"allowed_types": []}

        validate_clean_record(record)

    def test_rejects_missing_target_id(self) -> None:
        record = valid_record()
        record["evidence"]["target_id"] = "missing"

        with self.assertRaises(ValueError):
            validate_clean_record(record)

    def test_rejects_unknown_header_id(self) -> None:
        record = valid_record()
        record["evidence"]["row_header_ids"] = ["missing"]

        with self.assertRaises(ValueError):
            validate_clean_record(record)

    def test_rejects_invalid_bbox(self) -> None:
        record = valid_record()
        record["evidence"]["bbox_normalised"] = [0.1, 0.2, 1.5, 0.4]

        with self.assertRaises(ValueError):
            validate_clean_record(record)

    def test_rejects_empty_answer_and_image_path(self) -> None:
        record = valid_record()
        record["answer"]["raw"] = ""
        with self.assertRaises(ValueError):
            validate_clean_record(record)

        record = valid_record()
        record["image_path"] = ""
        with self.assertRaises(ValueError):
            validate_clean_record(record)

    def test_valid_group_record_passes(self) -> None:
        validate_group_record(valid_group_record())

    def test_group_record_rejects_missing_variant(self) -> None:
        record = valid_group_record()
        del record["variants"]["clean"]
        with self.assertRaises(ValueError):
            validate_group_record(record)

    def test_group_record_rejects_variant_missing_renderer_seed(self) -> None:
        record = valid_group_record()
        del record["variants"]["clean"]["renderer_seed"]
        with self.assertRaises(ValueError):
            validate_group_record(record)

    def test_group_record_rejects_variant_missing_image_id(self) -> None:
        record = valid_group_record()
        del record["variants"]["target_value_replacement"]["image_id"]
        with self.assertRaises(ValueError):
            validate_group_record(record)

    def test_group_record_rejects_variant_name_mismatch(self) -> None:
        record = valid_group_record()
        record["variants"]["clean"]["variant"] = "target_value_replacement"
        with self.assertRaises(ValueError):
            validate_group_record(record)

    def test_group_record_rejects_clean_with_changed_cells(self) -> None:
        record = valid_group_record()
        record["variants"]["clean"]["changed_cell_ids"] = ["cell_1_2"]
        with self.assertRaises(ValueError):
            validate_group_record(record)

    def test_group_record_rejects_counterfactual_without_changed_cells(self) -> None:
        record = valid_group_record()
        record["variants"]["target_value_replacement"]["changed_cell_ids"] = []
        with self.assertRaises(ValueError):
            validate_group_record(record)


if __name__ == "__main__":
    unittest.main()
