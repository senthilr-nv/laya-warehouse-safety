"""Tests for leak-free warehouse policy datasets."""

from __future__ import annotations

import unittest

from laya_warehouse.dataset import (
    add_horizontal_mirrors,
    dataset_summary,
    generate_cases,
    mirror_case,
)
from laya_warehouse.model import World
from laya_warehouse.oracle import oracle_labels


class DatasetTests(unittest.TestCase):
    def test_dataset_is_deterministic(self) -> None:
        first = generate_cases(split="validation")
        second = generate_cases(split="validation")

        self.assertEqual(first, second)
        self.assertEqual(dataset_summary(first), dataset_summary(second))

    def test_training_and_compositional_ood_families_are_disjoint(self) -> None:
        train = generate_cases(split="train")
        ood = generate_cases(split="ood_test")

        self.assertEqual(
            {case["family"] for case in train},
            {"stationary_pallet", "crossing_worker"},
        )
        self.assertEqual(
            {case["family"] for case in ood},
            {"combined_pallet_worker"},
        )
        self.assertTrue(
            {case["scenario_id"] for case in train}.isdisjoint(
                {case["scenario_id"] for case in ood}
            )
        )

    def test_observation_contains_no_policy_or_safety_answers(self) -> None:
        cases = generate_cases(split="validation")
        prohibited_keys = {
            "answer",
            "candidate_moves",
            "label",
            "oracle_action",
            "path_blocked",
            "risk",
            "safe",
            "safety",
            "shield",
            "unsafe",
            "visible_hazards",
        }

        def assert_clean(value) -> None:
            if isinstance(value, dict):
                self.assertTrue(prohibited_keys.isdisjoint(value))
                for child in value.values():
                    assert_clean(child)
            elif isinstance(value, list):
                for child in value:
                    assert_clean(child)

        for case in cases:
            assert_clean(case["state"])
            self.assertEqual(set(case["labels"]), {"action", "path_blocked"})

    def test_labels_match_the_oracle_for_every_validation_state(self) -> None:
        for case in generate_cases(split="validation"):
            expected = oracle_labels(World.from_observation(case["state"]))
            self.assertEqual(case["labels"]["action"], expected["action"])
            self.assertEqual(case["labels"]["path_blocked"], expected["path_blocked"])

    def test_horizontal_mirror_relabels_reflected_lateral_examples(self) -> None:
        cases = generate_cases(split="train")
        source = next(
            case for case in cases if case["labels"]["action"] == "shift_left"
        )
        mirrored = mirror_case(source)
        expected = oracle_labels(World.from_observation(mirrored["state"]))

        self.assertEqual(mirrored["labels"]["action"], expected["action"])
        self.assertEqual(mirrored["labels"]["path_blocked"], expected["path_blocked"])

        summary = dataset_summary(add_horizontal_mirrors(cases))
        self.assertEqual(summary["cases"], 760)
        self.assertGreater(summary["labels"]["action"]["shift_left"], 12)
        self.assertGreater(summary["labels"]["action"]["shift_right"], 8)


if __name__ == "__main__":
    unittest.main()
