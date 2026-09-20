"""Tests for immutable benchmark scenario identities and splits."""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import FrozenInstanceError

from laya_warehouse.scenarios import (
    FROZEN_MANIFEST,
    MANIFEST_VERSION,
    ScenarioFamily,
    development_manifest,
    final_benchmark_manifest,
    manifest_digest,
)


class ScenarioManifestTests(unittest.TestCase):
    def test_manifest_has_exactly_three_initial_families(self) -> None:
        self.assertEqual(MANIFEST_VERSION, 2)
        self.assertEqual(
            {spec.family for spec in FROZEN_MANIFEST},
            {
                ScenarioFamily.STATIONARY_PALLET,
                ScenarioFamily.CROSSING_WORKER,
                ScenarioFamily.COMBINED_PALLET_WORKER,
            },
        )
        self.assertEqual(
            Counter(spec.split for spec in FROZEN_MANIFEST),
            Counter(
                {
                    "train": 40,
                    "validation": 12,
                    "iid_test": 16,
                    "ood_test": 16,
                    "final_iid_test": 16,
                    "final_ood_test": 16,
                }
            ),
        )

    def test_scenario_ids_and_splits_are_disjoint(self) -> None:
        ids = [spec.scenario_id for spec in FROZEN_MANIFEST]
        self.assertEqual(len(ids), len(set(ids)))

        family_splits = {
            family: {spec.split for spec in FROZEN_MANIFEST if spec.family is family}
            for family in ScenarioFamily
        }
        self.assertEqual(
            family_splits[ScenarioFamily.COMBINED_PALLET_WORKER],
            {"ood_test", "final_ood_test"},
        )
        self.assertEqual(
            family_splits[ScenarioFamily.STATIONARY_PALLET],
            {"train", "validation", "iid_test", "final_iid_test"},
        )
        self.assertEqual(
            family_splits[ScenarioFamily.CROSSING_WORKER],
            {"train", "validation", "iid_test", "final_iid_test"},
        )

    def test_manifest_digest_pins_version_two_content(self) -> None:
        self.assertEqual(
            manifest_digest(),
            "b8f6069c74abda2282924af8b17c2d290bea332ba6da1b4cb1815608a78c0dfd",
        )

    def test_final_manifest_is_disjoint_and_pinned_before_evaluation(self) -> None:
        development = development_manifest()
        final = final_benchmark_manifest()

        self.assertTrue(
            {spec.scenario_id for spec in development}.isdisjoint(
                {spec.scenario_id for spec in final}
            )
        )
        def physical_key(spec):
            return (
                spec.family,
                spec.width,
                spec.height,
                spec.robot_x,
                spec.robot_y,
                spec.goal_columns,
                tuple((actor.kind, actor.x, actor.y, actor.dx) for actor in spec.actors),
            )

        development_states = {physical_key(spec) for spec in development}
        final_states = {physical_key(spec) for spec in final}
        self.assertEqual(len(development_states), 32)
        self.assertEqual(len(final_states), 32)
        self.assertTrue(development_states.isdisjoint(final_states))
        self.assertEqual(
            manifest_digest(final),
            "98c7ac272381a3f6c5d8db26f90960d84055c7bc84eeb3d23425125520790d6b",
        )

    def test_scenario_specs_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            FROZEN_MANIFEST[0].robot_x = 99  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
