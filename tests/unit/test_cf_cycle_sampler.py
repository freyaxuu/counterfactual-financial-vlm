from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from financial_vlm.data.synfintabs_loader import build_synfintabs_group_record
from financial_vlm.data.synfintabs_pilot import build_variant_plans, flatten_table, locate_answer_cell
from financial_vlm.training.cf_cycle_sampler import (
    CF_TYPES,
    EpochCycleSampler,
    GroupCycleSampler,
    build_training_pair,
    load_group_records,
    materialize_cycles,
    pair_to_log_record,
    parse_group_records,
    summarize_epoch,
    verify_cycle_coverage,
)
from tests.unit.test_synfintabs_pilot import synthetic_rows


def make_raw_group_record(source_index: int, *, table_id: str | None = None, seed: int = 1) -> dict:
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

    tid = table_id or f"table-{source_index}"
    return build_synfintabs_group_record(
        table={"id": tid, "theme": "0"},
        source_index=source_index,
        question_index=0,
        split="train",
        located=located,
        cells=cells,
        plans=plans,
        variant_image_paths={plan.variant: f"images/train/{tid}_{plan.variant}.png" for plan in plans},
        variant_image_sizes={plan.variant: (300, 90) for plan in plans},
        seed=seed,
    )


def make_groups(n: int, *, seed: int = 1) -> dict:
    raw_records = [make_raw_group_record(i, seed=seed) for i in range(n)]
    return parse_group_records(raw_records)


class GroupCycleSamplerTests(unittest.TestCase):
    def test_cycle_covers_all_three_types_without_early_repeat(self) -> None:
        groups = make_groups(1)
        group_id = next(iter(groups))
        sampler = GroupCycleSampler(groups, global_seed=42)

        cf_types = [sampler.visit(group_id)["counterfactual"]["cf_type"] for _ in range(9)]

        for cycle in range(3):
            window = cf_types[cycle * 3 : (cycle + 1) * 3]
            self.assertEqual(sorted(window), sorted(CF_TYPES))
            self.assertEqual(len(set(window)), 3, f"cycle {cycle} repeated a type before exhausting the others")

    def test_different_groups_receive_different_orders(self) -> None:
        groups = make_groups(20)
        sampler = GroupCycleSampler(groups, global_seed=42)
        orders = {
            group_id: tuple(sampler.visit(group_id)["counterfactual"]["cf_type"] for _ in range(3))
            for group_id in groups
        }
        self.assertGreater(len(set(orders.values())), 1, "all groups received the identical CF order")

    def test_same_seed_reproduces_sequence(self) -> None:
        groups = make_groups(5)
        sampler_a = GroupCycleSampler(groups, global_seed=99)
        sampler_b = GroupCycleSampler(groups, global_seed=99)

        seq_a = [sampler_a.visit(group_id) for group_id in groups for _ in range(3)]
        seq_b = [sampler_b.visit(group_id) for group_id in groups for _ in range(3)]

        self.assertEqual(seq_a, seq_b)

    def test_different_seed_changes_some_sequences(self) -> None:
        groups = make_groups(10)
        sampler_a = GroupCycleSampler(groups, global_seed=1)
        sampler_b = GroupCycleSampler(groups, global_seed=2)

        seq_a = {
            group_id: tuple(sampler_a.visit(group_id)["counterfactual"]["cf_type"] for _ in range(3))
            for group_id in groups
        }
        seq_b = {
            group_id: tuple(sampler_b.visit(group_id)["counterfactual"]["cf_type"] for _ in range(3))
            for group_id in groups
        }

        self.assertNotEqual(seq_a, seq_b)

    def test_checkpoint_resume_reproduces_uninterrupted_sequence(self) -> None:
        groups = make_groups(3)
        group_id = next(iter(groups))
        total_visits = 7

        ground_truth_sampler = GroupCycleSampler(groups, global_seed=7)
        ground_truth = [ground_truth_sampler.visit(group_id) for _ in range(total_visits)]

        first_sampler = GroupCycleSampler(groups, global_seed=7)
        first_half = [first_sampler.visit(group_id) for _ in range(4)]
        checkpoint = first_sampler.state_dict()

        resumed_sampler = GroupCycleSampler(groups, global_seed=7)
        resumed_sampler.load_state_dict(checkpoint)
        second_half = [resumed_sampler.visit(group_id) for _ in range(total_visits - 4)]

        self.assertEqual(first_half + second_half, ground_truth)

    def test_missing_variant_detected_before_training(self) -> None:
        raw = make_raw_group_record(0)
        del raw["variants"]["irrelevant_value_replacement"]

        with self.assertRaises(ValueError) as ctx:
            parse_group_records([raw])

        self.assertIn(raw["group_id"], str(ctx.exception))

    def test_header_address_uses_its_own_target_cell_and_answer(self) -> None:
        groups = make_groups(1)
        group = next(iter(groups.values()))

        clean_pair = build_training_pair(group, "target_value", cycle_index=0, position_in_cycle=0)
        header_pair = build_training_pair(group, "header_address", cycle_index=0, position_in_cycle=0)

        self.assertNotEqual(
            header_pair["counterfactual"]["target_cell_id"], clean_pair["clean"]["target_cell_id"]
        )
        self.assertEqual(
            header_pair["counterfactual"]["gold_answer"],
            group.counterfactuals["header_address"]["answer"]["raw"],
        )
        self.assertNotEqual(header_pair["counterfactual"]["gold_answer"], clean_pair["clean"]["gold_answer"])

    def test_irrelevant_value_never_touches_gold_evidence(self) -> None:
        raw = make_raw_group_record(0)

        groups = parse_group_records([raw])
        self.assertEqual(len(groups), 1)

        corrupted = copy.deepcopy(raw)
        clean_target_id = corrupted["variants"]["clean"]["evidence"]["target_id"]
        corrupted["variants"]["irrelevant_value_replacement"]["changed_cell_ids"] = [clean_target_id]

        with self.assertRaises(ValueError):
            parse_group_records([corrupted])

    def test_two_methods_consume_identical_cf_sequences(self) -> None:
        # Stands in for "counterfactual_augmentation" and "full method" both
        # instantiating a sampler over the same (groups, global_seed) --
        # confirms they'd see byte-identical CF sequences.
        groups = make_groups(8)
        cf_augmentation_sampler = GroupCycleSampler(groups, global_seed=555)
        full_method_sampler = GroupCycleSampler(groups, global_seed=555)

        seq_a = [cf_augmentation_sampler.visit(group_id) for group_id in groups for _ in range(3)]
        seq_b = [full_method_sampler.visit(group_id) for group_id in groups for _ in range(3)]

        self.assertEqual(seq_a, seq_b)

    def test_visit_unknown_group_raises(self) -> None:
        groups = make_groups(1)
        sampler = GroupCycleSampler(groups, global_seed=1)
        with self.assertRaises(KeyError):
            sampler.visit("does-not-exist")

    def test_load_state_dict_rejects_unknown_group(self) -> None:
        groups = make_groups(1)
        sampler = GroupCycleSampler(groups, global_seed=1)
        with self.assertRaises(KeyError):
            sampler.load_state_dict({"does-not-exist": 1})

    def test_load_group_records_from_file(self) -> None:
        raw = make_raw_group_record(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            groups = load_group_records(path)
        self.assertEqual(len(groups), 1)


class EpochCycleSamplerTests(unittest.TestCase):
    def test_epoch_visits_every_group_exactly_once(self) -> None:
        groups = make_groups(6)
        sampler = EpochCycleSampler(groups, global_seed=21)

        pairs = sampler.epoch(0)

        self.assertEqual({pair["group_id"] for pair in pairs}, set(groups))
        self.assertEqual(len(pairs), len(groups))

    def test_epoch_cycle_sampler_covers_full_cycle_across_three_epochs(self) -> None:
        groups = make_groups(6)
        sampler = EpochCycleSampler(groups, global_seed=21)

        pairs = sampler.epoch(0) + sampler.epoch(1) + sampler.epoch(2)

        verify_cycle_coverage(pairs)

    def test_epoch_order_is_shuffled_and_reproducible(self) -> None:
        groups = make_groups(10)
        sampler_a = EpochCycleSampler(groups, global_seed=8)
        sampler_b = EpochCycleSampler(groups, global_seed=8)

        order_a = [pair["group_id"] for pair in sampler_a.epoch(0)]
        order_b = [pair["group_id"] for pair in sampler_b.epoch(0)]

        self.assertEqual(order_a, order_b)
        self.assertNotEqual(order_a, sorted(order_a), "epoch order was not shuffled")


class CoverageAndLoggingHelperTests(unittest.TestCase):
    def test_verify_cycle_coverage_raises_for_incomplete_cycle(self) -> None:
        groups = make_groups(2)
        sampler = GroupCycleSampler(groups, global_seed=3)
        pairs = [sampler.visit(group_id) for group_id in groups for _ in range(2)]

        with self.assertRaises(ValueError):
            verify_cycle_coverage(pairs)

    def test_summarize_epoch_counts(self) -> None:
        groups = make_groups(5)
        sampler = EpochCycleSampler(groups, global_seed=11)
        pairs = sampler.epoch(0)

        summary = summarize_epoch(pairs, all_group_ids=set(groups) | {"missing-group"})

        self.assertEqual(summary["groups_visited"], 5)
        self.assertEqual(summary["missing_groups"], 1)
        self.assertEqual(summary["duplicate_visits"], 0)
        self.assertEqual(
            summary["target_value_count"] + summary["header_address_count"] + summary["irrelevant_value_count"],
            5,
        )

    def test_materialize_cycles_visits_each_group_three_times_per_cycle(self) -> None:
        groups = make_groups(6)

        pairs = materialize_cycles(groups, global_seed=13, num_cycles=1)

        self.assertEqual(len(pairs), 3 * len(groups))
        by_group: dict[str, list[str]] = {}
        for pair in pairs:
            by_group.setdefault(pair["group_id"], []).append(pair["counterfactual"]["cf_type"])
        self.assertEqual(set(by_group), set(groups))
        for cf_types in by_group.values():
            self.assertEqual(sorted(cf_types), sorted(CF_TYPES))

    def test_materialize_cycles_two_cycles_is_two_independent_full_cycles(self) -> None:
        groups = make_groups(4)

        pairs = materialize_cycles(groups, global_seed=13, num_cycles=2)

        self.assertEqual(len(pairs), 2 * 3 * len(groups))
        first_cycle, second_cycle = pairs[: 3 * len(groups)], pairs[3 * len(groups) :]
        verify_cycle_coverage(first_cycle)
        verify_cycle_coverage(second_cycle)

    def test_materialize_cycles_is_deterministic_for_same_seed(self) -> None:
        groups = make_groups(5)

        pairs_a = materialize_cycles(groups, global_seed=41, num_cycles=1)
        pairs_b = materialize_cycles(groups, global_seed=41, num_cycles=1)

        self.assertEqual(pairs_a, pairs_b)

    def test_materialize_cycles_rejects_non_positive_num_cycles(self) -> None:
        groups = make_groups(2)
        with self.assertRaises(ValueError):
            materialize_cycles(groups, global_seed=1, num_cycles=0)

    def test_pair_to_log_record_shape(self) -> None:
        groups = make_groups(1)
        group = next(iter(groups.values()))
        pair = build_training_pair(group, "target_value", cycle_index=0, position_in_cycle=0)

        log_record = pair_to_log_record(pair, run_id="run-1", epoch=0, global_step=42)

        self.assertEqual(
            set(log_record),
            {
                "run_id",
                "epoch",
                "global_step",
                "group_id",
                "cycle_index",
                "position_in_cycle",
                "cf_type",
                "clean_image_id",
                "cf_image_id",
                "clean_gold_answer",
                "cf_gold_answer",
                "clean_target_cell_id",
                "cf_target_cell_id",
            },
        )


if __name__ == "__main__":
    unittest.main()
