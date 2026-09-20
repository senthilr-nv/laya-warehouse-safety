"""Tests for the pre-registered compositional-coverage experiment split."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from laya_warehouse.dataset import dataset_summary, generate_cases_for_specs
from laya_warehouse.scenarios import (
    ScenarioFamily,
    final_benchmark_manifest,
    geometry_signature,
    load_versioned_manifest,
    worker_trajectory_signatures,
)


MANIFEST_DIR = Path("benchmarks/compositional-coverage/manifests")
EXPECTED_DIGESTS = {
    "train": "175534e88d06d9fd2a2b97613ea8e07c5256bff50966937660a06a25f4d65430",
    "dev": "162edfce1683c848657271eac2020924e505cd53e50d5c163b2c6f1dbf396293",
    "final": "bb40999de39c6f921c5a960de4e9dee4be9a13aab351f2719441d9d0f1ae4bfe",
}
EXPECTED_GEOMETRY_DIGESTS = {
    "train": "c49f249ecb17fd2c06db6594ae73ab55f96d99e82a2fcdbaffb50a7e5dd7dd40",
    "dev": "4e706d11fab7460e743cddaeeb4d701c56afcb06dbaaf77971a898de146ca77c",
    "final": "ec2746f5ec06c44f96649a463a9fa19811d7be206e2d9675bdf57f710d55b6b8",
}
EXPECTED_TRAJECTORY_DIGESTS = {
    "train": "3429cf915f9f8072e8e355f4327019e22e212c59a33d79a68a0b9afc46459eb8",
    "dev": "3b3f165095c1240cfc888162a5eb35538ddeddaf18fd3c0cc90a57ee9a3b8544",
    "final": "9ab0d9f08b6e21cac3a19e950e31d0761b2c4b9f59025ddb045d1800a7e2b773",
}


def load_manifests():
    return {
        role: load_versioned_manifest(
            MANIFEST_DIR / f"{role}.json",
            expected_role=role,
        )
        for role in ("train", "dev", "final")
    }


class CoverageManifestTests(unittest.TestCase):
    def test_manifests_have_pinned_digests_and_family_counts(self) -> None:
        manifests = load_manifests()
        expected_counts = {
            "train": Counter(
                {
                    ScenarioFamily.STATIONARY_PALLET: 20,
                    ScenarioFamily.CROSSING_WORKER: 20,
                    ScenarioFamily.COMBINED_PALLET_WORKER: 8,
                }
            ),
            "dev": Counter(
                {
                    ScenarioFamily.STATIONARY_PALLET: 6,
                    ScenarioFamily.CROSSING_WORKER: 6,
                    ScenarioFamily.COMBINED_PALLET_WORKER: 8,
                }
            ),
            "final": Counter(
                {
                    ScenarioFamily.STATIONARY_PALLET: 8,
                    ScenarioFamily.CROSSING_WORKER: 8,
                    ScenarioFamily.COMBINED_PALLET_WORKER: 16,
                }
            ),
        }
        for role, manifest in manifests.items():
            self.assertEqual(manifest.scenario_digest, EXPECTED_DIGESTS[role])
            self.assertEqual(
                manifest.geometry_signature_digest,
                EXPECTED_GEOMETRY_DIGESTS[role],
            )
            self.assertEqual(
                manifest.worker_trajectory_signature_digest,
                EXPECTED_TRAJECTORY_DIGESTS[role],
            )
            self.assertEqual(
                Counter(spec.family for spec in manifest.scenarios),
                expected_counts[role],
            )
            cases = generate_cases_for_specs(manifest.scenarios)
            self.assertEqual(
                dataset_summary(cases, specs=manifest.scenarios)["manifest_digest"],
                EXPECTED_DIGESTS[role],
            )

    def test_final_geometry_and_worker_trajectories_are_held_out(self) -> None:
        manifests = load_manifests()
        train_dev = manifests["train"].scenarios + manifests["dev"].scenarios
        final = manifests["final"].scenarios

        train_dev_geometry = {geometry_signature(spec) for spec in train_dev}
        final_geometry = {geometry_signature(spec) for spec in final}
        self.assertTrue(final_geometry.isdisjoint(train_dev_geometry))

        train_dev_trajectories = {
            signature
            for spec in train_dev
            for signature in worker_trajectory_signatures(spec)
        }
        final_trajectories = {
            signature
            for spec in final
            for signature in worker_trajectory_signatures(spec)
        }
        self.assertTrue(final_trajectories.isdisjoint(train_dev_trajectories))

    def test_scenario_ids_are_unique_across_roles(self) -> None:
        manifests = load_manifests()
        specs = tuple(
            spec
            for manifest in manifests.values()
            for spec in manifest.scenarios
        )
        scenario_ids = [spec.scenario_id for spec in specs]
        self.assertEqual(len(scenario_ids), len(set(scenario_ids)))

    def test_final_physical_states_are_new_relative_to_prior_final(self) -> None:
        def physical_signature(spec):
            return (
                spec.family,
                geometry_signature(spec),
                worker_trajectory_signatures(spec),
            )

        current = load_manifests()["final"].scenarios
        previous = final_benchmark_manifest()
        self.assertTrue(
            {physical_signature(spec) for spec in current}.isdisjoint(
                {physical_signature(spec) for spec in previous}
            )
        )

    def test_all_policy_observations_exclude_oracle_and_safety_labels(self) -> None:
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

        for manifest in load_manifests().values():
            cases = generate_cases_for_specs(manifest.scenarios)
            for case in cases:
                assert_clean(case["state"])
                self.assertEqual(set(case["labels"]), {"action", "path_blocked"})


if __name__ == "__main__":
    unittest.main()
