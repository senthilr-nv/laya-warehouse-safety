"""Tests for immutable benchmark scenario identities and splits."""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import FrozenInstanceError

from laya_warehouse.scenarios import (
    FROZEN_MANIFEST,
    MANIFEST_VERSION,
    ScenarioFamily,
    manifest_digest,
)


class ScenarioManifestTests(unittest.TestCase):
    def test_manifest_has_exactly_three_initial_families(self) -> None:
        self.assertEqual(MANIFEST_VERSION, 1)
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
            Counter({"train": 40, "validation": 12, "iid_test": 16, "ood_test": 16}),
        )

    def test_scenario_ids_and_splits_are_disjoint(self) -> None:
        ids = [spec.scenario_id for spec in FROZEN_MANIFEST]
        self.assertEqual(len(ids), len(set(ids)))

        family_splits = {
            family: {spec.split for spec in FROZEN_MANIFEST if spec.family is family}
            for family in ScenarioFamily
        }
        self.assertEqual(
            family_splits[ScenarioFamily.COMBINED_PALLET_WORKER], {"ood_test"}
        )
        self.assertEqual(
            family_splits[ScenarioFamily.STATIONARY_PALLET],
            {"train", "validation", "iid_test"},
        )
        self.assertEqual(
            family_splits[ScenarioFamily.CROSSING_WORKER],
            {"train", "validation", "iid_test"},
        )

    def test_manifest_digest_pins_version_one_content(self) -> None:
        self.assertEqual(
            manifest_digest(),
            "bcbfa32073342508aa089b0f61329592c2d0e0f6fbdefca28f9ae7dabcd61edd",
        )

    def test_scenario_specs_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            FROZEN_MANIFEST[0].robot_x = 99  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
