"""Tests for deterministic benchmark metrics and report evidence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from laya_warehouse.benchmark import run_benchmark, save_benchmark_report
from laya_warehouse.controllers import Decision
from laya_warehouse.model import Action, World
from laya_warehouse.oracle import direct_path_blocked, shortest_collision_free_plan
from laya_warehouse.recording import verify_record
from laya_warehouse.scenarios import manifest_for_split


class OracleStubController:
    name = "oracle_stub"

    def decide(self, observation):
        world = World.from_observation(observation)
        action = shortest_collision_free_plan(world).action
        blocked = direct_path_blocked(world)
        return Decision(
            action=action,
            latency_ms=2.0,
            details={
                "runtime": "laya",
                "answers": {"path_blocked": {"noul": float(blocked)}},
            },
        )


def representative_specs():
    iid = manifest_for_split("iid_test")
    ood = manifest_for_split("ood_test")
    return (iid[0], iid[8], ood[0])


class BenchmarkTests(unittest.TestCase):
    def test_oracle_stub_has_known_metrics_in_both_tracks(self) -> None:
        report, selected = run_benchmark(
            {"laya": lambda _scenario: OracleStubController()},
            specs=representative_specs(),
            max_ticks=40,
            model_load_ms={"laya": 12.0},
        )

        for track in ("policy_only", "layered"):
            metrics = report["results"]["laya"]["tracks"][track]["overall"]
            self.assertEqual(metrics["completion_rate"], 1.0)
            self.assertEqual(metrics["collision_rate"], 0.0)
            self.assertEqual(metrics["unsafe_request_rate"], 0.0)
            self.assertEqual(metrics["boundary_guard_interventions"], 0)
            self.assertEqual(metrics["shield_interventions"], 0)
            self.assertEqual(metrics["oracle_action_accuracy"], 1.0)
            self.assertEqual(metrics["path_blocked"]["accuracy"], 1.0)
            self.assertEqual(metrics["path_blocked"]["brier"], 0.0)

        self.assertEqual(len(selected), 6)
        for record in selected.values():
            verify_record(record)

        runtime = report["runtime_evidence"]["laya"]
        self.assertEqual(runtime["model_load_ms"], 12.0)
        self.assertEqual(runtime["cold_inference_ms"], 2.0)
        self.assertEqual(runtime["warm_inference_ms"]["p50"], 2.0)
        self.assertEqual(runtime["warm_inference_ms"]["p95"], 2.0)

    def test_deterministic_result_digest_repeats(self) -> None:
        kwargs = {
            "controller_factories": {
                "laya": lambda _scenario: OracleStubController()
            },
            "specs": representative_specs(),
            "max_ticks": 40,
            "model_load_ms": {"laya": 12.0},
        }

        first, _ = run_benchmark(**kwargs)
        second, _ = run_benchmark(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first["deterministic_digest"], second["deterministic_digest"])

    def test_report_schema_and_refuse_overwrite(self) -> None:
        report, _ = run_benchmark(
            {"laya": lambda _scenario: OracleStubController()},
            specs=representative_specs(),
            max_ticks=40,
        )

        self.assertEqual(report["report_schema_version"], 1)
        self.assertEqual(report["manifest"]["version"], 2)
        self.assertEqual(report["evaluation_role"], "final")
        self.assertEqual(report["held_out_family"], "combined_pallet_worker")
        self.assertEqual(
            report["training_families"], ["stationary_pallet", "crossing_worker"]
        )
        self.assertEqual(
            report["manifest"]["scenario_counts"],
            {
                "combined_pallet_worker": 1,
                "crossing_worker": 1,
                "stationary_pallet": 1,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            save_benchmark_report(report, output)
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                save_benchmark_report(report, output)


if __name__ == "__main__":
    unittest.main()
