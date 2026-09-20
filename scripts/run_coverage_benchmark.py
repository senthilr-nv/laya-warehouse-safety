#!/usr/bin/env python3
"""Run the frozen compositional-coverage final comparison once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from laya_warehouse.benchmark import (
    random_seed_for_scenario,
    run_benchmark,
    save_benchmark_report,
    selected_episode_filenames,
)
from laya_warehouse.controllers import HeuristicController, LayaController, RandomController
from laya_warehouse.evidence import model_evidence
from laya_warehouse.recording import save_record, verify_record
from laya_warehouse.scenarios import VERSIONED_MANIFEST_SCHEMA, load_versioned_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--coverage-model", required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-ticks", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {args.output}")
    if args.episodes_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark episodes directory: {args.episodes_dir}"
        )

    manifest = load_versioned_manifest(args.final_manifest, expected_role="final")
    baseline = LayaController(
        model=args.baseline_model,
        device=args.device,
        questions=LayaController.BENCHMARK_QUESTIONS,
    )
    coverage = LayaController(
        model=args.coverage_model,
        device=args.device,
        questions=LayaController.BENCHMARK_QUESTIONS,
    )
    factories = {
        "heuristic": lambda _scenario: HeuristicController(),
        "laya_baseline": lambda _scenario: baseline,
        "laya_combined_coverage": lambda _scenario: coverage,
        "random": lambda scenario: RandomController(random_seed_for_scenario(scenario)),
    }
    report, selected_records = run_benchmark(
        factories,
        specs=manifest.scenarios,
        max_ticks=args.max_ticks,
        model_load_ms={
            "laya_baseline": baseline.model_load_ms,
            "laya_combined_coverage": coverage.model_load_ms,
        },
        evaluation_role="coverage_final",
        benchmark_name=manifest.experiment,
        manifest_version=VERSIONED_MANIFEST_SCHEMA,
        training_families=(
            "stationary_pallet",
            "crossing_worker",
            "combined_pallet_worker",
        ),
        held_out_family="combined_pallet_worker signatures",
    )
    report["diagnostic"] = {
        "hypothesis": (
            "The prior combined-family failure is primarily a training-coverage problem rather "
            "than an inherent Laya limitation."
        ),
        "implementation_commit": args.implementation_commit,
        "comparison": (
            "The preserved baseline checkpoint and the combined-coverage checkpoint are evaluated "
            "on the same frozen final scenarios in this run."
        ),
        "unchanged_contracts": [
            "observation schema",
            "typed questions and criteria",
            "oracle and path_blocked horizon",
            "safety shield and controller semantics",
            "policy-only and layered tracks",
            "metrics and max tick budget",
        ],
        "planned_difference": (
            "The coverage checkpoint adds eight combined-family training scenarios and uses eight "
            "combined-family dev scenarios for checkpoint selection."
        ),
        "final_manifest": {
            "path": str(args.final_manifest),
            "scenario_digest": manifest.scenario_digest,
            "geometry_signature_digest": manifest.geometry_signature_digest,
            "worker_trajectory_signature_digest": (
                manifest.worker_trajectory_signature_digest
            ),
            "split_rule": manifest.split_rule,
        },
    }
    report["controller_evidence"] = {
        "heuristic": {"implementation": "deterministic first-safe preference"},
        "laya_baseline": model_evidence(args.baseline_model),
        "laya_combined_coverage": model_evidence(args.coverage_model),
        "random": {"seed": "sha256(scenario_id) first 32 bits"},
    }

    for key, record in sorted(selected_records.items()):
        verify_record(record)
        save_record(record, args.episodes_dir / f"{key}.json")
    report["selected_episode_files"] = selected_episode_filenames(selected_records)
    save_benchmark_report(report, args.output)

    compact_results = {
        controller: {
            track: result["overall"]
            for track, result in controller_result["tracks"].items()
        }
        for controller, controller_result in report["results"].items()
    }
    print(
        json.dumps(
            {
                "report": str(args.output),
                "episodes": len(selected_records),
                "deterministic_digest": report["deterministic_digest"],
                "results": compact_results,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
