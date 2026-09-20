#!/usr/bin/env python3
"""Run the frozen warehouse benchmark and save replay-verifiable evidence."""

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
from laya_warehouse.scenarios import development_manifest, final_benchmark_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="fine-tuned Laya model path or ID")
    parser.add_argument("--device", default="cuda", help="torch device for Laya")
    parser.add_argument("--max-ticks", type=int, default=40)
    parser.add_argument(
        "--manifest-role",
        choices=("final", "development"),
        default="final",
        help="final is the frozen one-shot set; development selects earlier tuning evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark-v1.json"),
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path("results/benchmark-v1-episodes"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {args.output}")
    if args.episodes_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark episodes directory: {args.episodes_dir}"
        )

    laya_controller = LayaController(
        model=args.model,
        device=args.device,
        questions=LayaController.BENCHMARK_QUESTIONS,
    )
    factories = {
        "heuristic": lambda _scenario: HeuristicController(),
        "laya": lambda _scenario: laya_controller,
        "random": lambda scenario: RandomController(random_seed_for_scenario(scenario)),
    }
    specs = (
        final_benchmark_manifest()
        if args.manifest_role == "final"
        else development_manifest()
    )
    report, selected_records = run_benchmark(
        factories,
        specs=specs,
        max_ticks=args.max_ticks,
        model_load_ms={"laya": laya_controller.model_load_ms},
        evaluation_role=args.manifest_role,
    )
    report["controller_evidence"] = {
        "heuristic": {"implementation": "deterministic first-safe preference"},
        "laya": model_evidence(args.model),
        "random": {"seed": "sha256(scenario_id) first 32 bits"},
    }

    for key, record in sorted(selected_records.items()):
        verify_record(record)
        episode_path = args.episodes_dir / f"{key}.json"
        save_record(record, episode_path)
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
