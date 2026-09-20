"""Verification for the preserved compositional-coverage diagnostic evidence."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from laya_warehouse.recording import verify_record


EVIDENCE_DIR = Path("benchmarks/compositional-coverage")


class CoverageEvidenceTests(unittest.TestCase):
    def test_final_report_digest_and_replays_are_verifiable(self) -> None:
        final_dir = EVIDENCE_DIR / "final"
        report = json.loads((final_dir / "report.json").read_text(encoding="utf-8"))
        payload = {
            key: report[key]
            for key in ("configuration", "manifest", "results")
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "76c46855490e0e6a5ea5314462575a4b503162cbae6ff5b310b13aa5ba87615d",
        )
        self.assertEqual(
            report["manifest"]["digest"],
            "bb40999de39c6f921c5a960de4e9dee4be9a13aab351f2719441d9d0f1ae4bfe",
        )

        paths = sorted((final_dir / "episodes").glob("*.json"))
        self.assertEqual(len(paths), 24)
        self.assertEqual(
            {path.name for path in paths},
            set(report["selected_episode_files"].values()),
        )
        self.assertTrue(
            all(
                not Path(filename).is_absolute()
                for filename in report["selected_episode_files"].values()
            )
        )
        for path in paths:
            verify_record(json.loads(path.read_text(encoding="utf-8")))

    def test_model_and_training_provenance_is_pinned(self) -> None:
        report_path = EVIDENCE_DIR / "final/report.json"
        training_path = EVIDENCE_DIR / "development/training_metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))

        self.assertEqual(
            hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "474f85e2b03682cfae12141967df178e74758359a90970e77e6937fd2b1b7061",
        )
        self.assertEqual(
            hashlib.sha256(training_path.read_bytes()).hexdigest(),
            "90583751e797af6190c0918263882689d0be3e83c8bde515c4078dd3ad99e147",
        )
        evidence = report["controller_evidence"]
        self.assertEqual(
            evidence["laya_baseline"]["weights_sha256"],
            "a4a3bd6746d5a01b6147cf673f1d7a8909406cf72dcc0f285981f738c6be44a8",
        )
        self.assertEqual(
            evidence["laya_combined_coverage"]["weights_sha256"],
            "f48d28b16208ecabedd6d165ccc65c7fa7a387fc7c0751a5c2d43995099b96a5",
        )
        self.assertEqual(training["best_epoch"], 7)
        self.assertEqual(
            training["implementation_commit"],
            "61413e8d4a42a4ff72e7b180f831ecc3fd9b455e",
        )

    def test_mixed_combined_family_result_is_preserved(self) -> None:
        report = json.loads(
            (EVIDENCE_DIR / "final/report.json").read_text(encoding="utf-8")
        )
        results = report["results"]
        baseline = results["laya_baseline"]["tracks"]
        coverage = results["laya_combined_coverage"]["tracks"]

        baseline_policy = baseline["policy_only"]["by_family"][
            "combined_pallet_worker"
        ]
        coverage_policy = coverage["policy_only"]["by_family"][
            "combined_pallet_worker"
        ]
        self.assertEqual(baseline_policy["completion_rate"], 0.0)
        self.assertEqual(coverage_policy["completion_rate"], 0.0)
        self.assertEqual(baseline_policy["collision_rate"], 1.0)
        self.assertEqual(coverage_policy["collision_rate"], 1.0)

        baseline_layered = baseline["layered"]["by_family"][
            "combined_pallet_worker"
        ]
        coverage_layered = coverage["layered"]["by_family"][
            "combined_pallet_worker"
        ]
        self.assertEqual(baseline_layered["completion_rate"], 0.0625)
        self.assertEqual(coverage_layered["completion_rate"], 0.75)
        self.assertEqual(baseline_layered["shield_interventions"], 567)
        self.assertEqual(coverage_layered["shield_interventions"], 31)


if __name__ == "__main__":
    unittest.main()
