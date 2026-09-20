"""Tests for the synthetic warehouse policy dataset."""

from __future__ import annotations

import unittest
from collections import Counter

from laya_warehouse.dataset import generate_cases
from laya_warehouse.model import Action


class DatasetTests(unittest.TestCase):
    def test_dataset_is_balanced_and_deterministic(self) -> None:
        first = generate_cases(per_action=3, seed=17)
        second = generate_cases(per_action=3, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(
            Counter(case["label"] for case in first),
            Counter({action.value: 3 for action in Action}),
        )

    def test_wait_examples_describe_temporary_crossing_traffic(self) -> None:
        cases = generate_cases(per_action=2, seed=5)
        wait_states = [case["state"] for case in cases if case["label"] == Action.WAIT.value]

        for state in wait_states:
            self.assertFalse(state["candidate_moves"]["advance"]["safe"])
            self.assertTrue(state["candidate_moves"]["wait"]["safe"])
            self.assertIn(state["visible_hazards"][0]["kind"], ("worker", "forklift"))

    def test_lateral_examples_avoid_a_stationary_pallet_and_align_with_goal(self) -> None:
        cases = generate_cases(per_action=3, seed=9)

        for case in cases:
            if case["label"] not in (Action.SHIFT_LEFT.value, Action.SHIFT_RIGHT.value):
                continue
            state = case["state"]
            pallet = state["visible_hazards"][0]
            self.assertEqual(pallet["kind"], "pallet")
            self.assertEqual(pallet["relative_column"], 0)
            self.assertEqual(pallet["motion"], "stationary")
            if case["label"] == Action.SHIFT_LEFT.value:
                self.assertGreater(state["robot"]["column"], max(state["goal"]["columns"]))
            else:
                self.assertLess(state["robot"]["column"], min(state["goal"]["columns"]))


if __name__ == "__main__":
    unittest.main()
