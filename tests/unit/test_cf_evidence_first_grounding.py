from __future__ import annotations

import unittest

from financial_vlm.data.grounding_target import format_evidence_first_target
from financial_vlm.training.cf_cycle_sampler import CF_TYPES, GroupRecord, VARIANT_BY_CF_TYPE
from financial_vlm.training.cf_evidence_first_grounding import (
    build_evidence_first_cf_pair,
    materialize_evidence_first_cf_pairs,
    pair_to_log_record,
    validate_and_filter_groups,
)

from tests.unit.test_cf_cycle_sampler import make_groups
from tests.unit.test_grounding_target import make_payload


def make_group_record(group_id: str, *, ambiguous_variant: str | None = None) -> GroupRecord:
    # cf_cycle_sampler.EpochCycleSampler's internal (unused-here) plain-pair
    # scheduling also runs over these payloads, so they need the fields
    # _example_view reads (image_id, gold_evidence_ids, expected_behavior on
    # CF sides) even though this module never reads those fields itself.
    clean = make_payload(image_path=f"images/train/{group_id}__clean.png")
    clean["image_id"] = f"{group_id}__clean"
    clean["gold_evidence_ids"] = ["cell_12_0", "cell_0_2", "cell_12_2"]

    counterfactuals = {}
    for cf_type in CF_TYPES:
        variant_name = VARIANT_BY_CF_TYPE[cf_type]
        payload = make_payload(image_path=f"images/train/{group_id}__{variant_name}.png")
        payload["image_id"] = f"{group_id}__{variant_name}"
        payload["gold_evidence_ids"] = ["cell_12_0", "cell_0_2", "cell_12_2"]
        payload["expected_behavior"] = "follow_edit"
        if variant_name == ambiguous_variant:
            payload["evidence"] = dict(payload["evidence"])
            payload["evidence"]["row_header_ids"] = ["cell_12_0", "cell_13_0"]
        counterfactuals[cf_type] = payload

    if ambiguous_variant == "clean":
        clean = dict(clean)
        clean["evidence"] = dict(clean["evidence"])
        clean["evidence"]["row_header_ids"] = ["cell_12_0", "cell_13_0"]

    return GroupRecord(group_id=group_id, question="What was revenue?", clean=clean, counterfactuals=counterfactuals)


class ValidateAndFilterGroupsTests(unittest.TestCase):
    def test_all_valid_groups_kept(self) -> None:
        groups = {f"g{i}": make_group_record(f"g{i}") for i in range(5)}
        valid, stats = validate_and_filter_groups(groups)
        self.assertEqual(set(valid), set(groups))
        self.assertEqual(stats["total_groups"], 5)
        self.assertEqual(stats["valid_groups"], 5)

    def test_group_invalid_on_clean_is_excluded_and_counted(self) -> None:
        groups = {
            "g0": make_group_record("g0"),
            "g1": make_group_record("g1", ambiguous_variant="clean"),
        }
        valid, stats = validate_and_filter_groups(groups)
        self.assertEqual(set(valid), {"g0"})
        self.assertEqual(stats["valid_groups"], 1)
        self.assertEqual(stats.get("invalid__clean__ambiguous_row_header"), 1)

    def test_group_invalid_on_a_single_cf_variant_is_excluded_even_though_clean_is_fine(self) -> None:
        groups = {
            "g0": make_group_record("g0"),
            "g1": make_group_record("g1", ambiguous_variant="header_address_swap"),
        }
        valid, stats = validate_and_filter_groups(groups)
        self.assertEqual(set(valid), {"g0"})
        self.assertEqual(stats.get("invalid__header_address__ambiguous_row_header"), 1)

    def test_never_silently_drops_total_accounts_for_every_group(self) -> None:
        groups = {
            "g0": make_group_record("g0"),
            "g1": make_group_record("g1", ambiguous_variant="clean"),
            "g2": make_group_record("g2", ambiguous_variant="target_value_replacement"),
        }
        valid, stats = validate_and_filter_groups(groups)
        accounted = stats["valid_groups"] + sum(
            value for key, value in stats.items() if key.startswith("invalid__")
        )
        self.assertEqual(accounted, stats["total_groups"])


class BuildEvidenceFirstCfPairTests(unittest.TestCase):
    def test_pair_has_independent_evidence_first_targets_on_each_side(self) -> None:
        group = make_group_record("g0")
        pair = build_evidence_first_cf_pair(group, "header_address", cycle_index=0, position_in_cycle=0)

        self.assertEqual(pair["group_id"], "g0")
        self.assertEqual(pair["cf_type"], "header_address")
        clean_example = pair["clean"]
        cf_example = pair["counterfactual"]
        self.assertEqual(clean_example.variant, "clean")
        self.assertEqual(cf_example.variant, "header_address_swap")

        expected_clean_target = format_evidence_first_target(clean_example.answer_raw, clean_example.bbox_1000)
        expected_cf_target = format_evidence_first_target(cf_example.answer_raw, cf_example.bbox_1000)
        # target_text on GroundingExample is the answer-first (Grounding-only) format;
        # the evidence-first target must be rebuilt from answer_raw/bbox_1000, not reused.
        self.assertNotEqual(clean_example.target_text, expected_clean_target)
        self.assertTrue(expected_clean_target.startswith("<bbox>"))
        self.assertTrue(expected_cf_target.startswith("<bbox>"))

    def test_rejects_unknown_cf_type(self) -> None:
        group = make_group_record("g0")
        with self.assertRaises(ValueError):
            build_evidence_first_cf_pair(group, "not_a_real_type", cycle_index=0, position_in_cycle=0)

    def test_raises_if_a_side_fails_grounding_validation(self) -> None:
        group = make_group_record("g0", ambiguous_variant="header_address_swap")
        with self.assertRaises(ValueError):
            build_evidence_first_cf_pair(group, "header_address", cycle_index=0, position_in_cycle=0)


class MaterializeEvidenceFirstCfPairsTests(unittest.TestCase):
    def test_num_cycles_one_visits_each_group_three_times_covering_all_cf_types(self) -> None:
        groups = {f"g{i}": make_group_record(f"g{i}") for i in range(6)}
        pairs = materialize_evidence_first_cf_pairs(groups, global_seed=42, num_cycles=1)

        self.assertEqual(len(pairs), 3 * len(groups))
        cf_types_by_group: dict[str, list[str]] = {}
        for pair in pairs:
            cf_types_by_group.setdefault(pair["group_id"], []).append(pair["cf_type"])
        for group_id, types in cf_types_by_group.items():
            self.assertEqual(sorted(types), sorted(CF_TYPES), group_id)

    def test_matches_cf_cycle_sampler_group_and_cf_type_schedule(self) -> None:
        from financial_vlm.training.cf_cycle_sampler import materialize_cycles

        groups = {f"g{i}": make_group_record(f"g{i}") for i in range(6)}
        plain_pairs = materialize_cycles(groups, global_seed=7, num_cycles=1)
        ef_pairs = materialize_evidence_first_cf_pairs(groups, global_seed=7, num_cycles=1)

        plain_schedule = [(p["group_id"], p["counterfactual"]["cf_type"]) for p in plain_pairs]
        ef_schedule = [(p["group_id"], p["cf_type"]) for p in ef_pairs]
        self.assertEqual(plain_schedule, ef_schedule)

    def test_same_seed_is_reproducible(self) -> None:
        groups = {f"g{i}": make_group_record(f"g{i}") for i in range(4)}
        pairs_a = materialize_evidence_first_cf_pairs(groups, global_seed=99, num_cycles=1)
        pairs_b = materialize_evidence_first_cf_pairs(groups, global_seed=99, num_cycles=1)
        schedule_a = [(p["group_id"], p["cf_type"]) for p in pairs_a]
        schedule_b = [(p["group_id"], p["cf_type"]) for p in pairs_b]
        self.assertEqual(schedule_a, schedule_b)

    def test_rejects_num_cycles_below_one(self) -> None:
        groups = {"g0": make_group_record("g0")}
        with self.assertRaises(ValueError):
            materialize_evidence_first_cf_pairs(groups, global_seed=1, num_cycles=0)

    def test_end_to_end_with_realistic_group_records_from_synfintabs_fixtures(self) -> None:
        groups = make_groups(4)
        valid, stats = validate_and_filter_groups(groups)
        self.assertEqual(stats["total_groups"], 4)
        pairs = materialize_evidence_first_cf_pairs(valid, global_seed=20260804, num_cycles=1)
        self.assertEqual(len(pairs), 3 * len(valid))
        for pair in pairs:
            self.assertTrue(pair["clean"].target_text.startswith("<answer>"))


class PairToLogRecordTests(unittest.TestCase):
    def test_log_record_captures_both_sides(self) -> None:
        group = make_group_record("g0")
        pair = build_evidence_first_cf_pair(group, "target_value", cycle_index=1, position_in_cycle=2)
        record = pair_to_log_record(pair, run_id="run1", epoch=1, global_step=42)

        self.assertEqual(record["run_id"], "run1")
        self.assertEqual(record["global_step"], 42)
        self.assertEqual(record["group_id"], "g0")
        self.assertEqual(record["cf_type"], "target_value")
        self.assertEqual(record["clean_target_cell_id"], pair["clean"].target_cell_id)
        self.assertEqual(record["cf_target_cell_id"], pair["counterfactual"].target_cell_id)
        self.assertEqual(record["clean_bbox_1000"], list(pair["clean"].bbox_1000))
        self.assertEqual(record["cf_bbox_1000"], list(pair["counterfactual"].bbox_1000))


if __name__ == "__main__":
    unittest.main()
