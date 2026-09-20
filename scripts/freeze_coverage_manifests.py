#!/usr/bin/env python3
"""Freeze the pre-registered compositional-coverage experiment manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from laya_warehouse.scenarios import (
    VERSIONED_MANIFEST_SCHEMA,
    ActorSpec,
    ScenarioFamily,
    ScenarioSpec,
    _crossing_worker,
    geometry_signature_digest,
    manifest_digest,
    manifest_for_split,
    worker_trajectory_signature_digest,
)


EXPERIMENT = "warehouse-compositional-coverage-v1"
DEFAULT_OUTPUT = Path("benchmarks/compositional-coverage/manifests")


TRAIN_COMBINED_PLAN = (
    (3, 2, 5, 0),
    (4, 2, 5, 1),
    (5, 3, 5, 0),
    (6, 3, 5, 1),
    (7, 2, 4, 0),
    (3, 3, 4, 1),
    (5, 2, 4, 1),
    (7, 3, 4, 0),
)

DEV_COMBINED_PLAN = (
    (3, 4, 2, 0),
    (4, 4, 2, 1),
    (5, 5, 2, 0),
    (6, 5, 2, 1),
    (7, 4, 3, 0),
    (3, 5, 3, 1),
    (5, 4, 3, 1),
    (7, 5, 3, 0),
)

FINAL_STATIONARY_PLAN = (
    (3, 6),
    (4, 7),
    (5, 1),
    (6, 6),
    (6, 7),
    (7, 1),
    (7, 7),
    (2, 1),
)

FINAL_CROSSING_PLAN = (
    (3, 1, 0),
    (3, 1, 1),
    (7, 1, 0),
    (7, 1, 1),
    (3, 7, 0),
    (4, 7, 0),
    (6, 7, 1),
    (7, 7, 1),
)

FINAL_COMBINED_PLAN = (
    (3, 1, 7, 0),
    (4, 1, 7, 0),
    (5, 1, 7, 1),
    (6, 1, 7, 0),
    (7, 1, 7, 0),
    (3, 6, 1, 0),
    (4, 6, 1, 0),
    (5, 6, 1, 0),
    (6, 6, 1, 0),
    (7, 6, 1, 1),
    (3, 7, 1, 1),
    (4, 7, 1, 0),
    (5, 7, 1, 1),
    (6, 7, 1, 1),
    (7, 7, 1, 1),
    (3, 6, 7, 1),
)


def _copy_base_spec(spec: ScenarioSpec, role: str, index: int) -> ScenarioSpec:
    return replace(
        spec,
        scenario_id=f"coverage-{role}-{spec.family.value}-{index:03d}",
        split=f"coverage_{role}",
        actors=tuple(
            replace(actor, actor_id=f"{actor.kind}-{role}-{index:03d}-{actor_index}")
            for actor_index, actor in enumerate(spec.actors)
        ),
    )


def _combined_spec(
    role: str,
    index: int,
    robot_x: int,
    pallet_y: int,
    worker_y: int,
    worker_variant: int,
) -> ScenarioSpec:
    worker = replace(
        _crossing_worker(robot_x, worker_y, worker_variant, 11),
        actor_id=f"worker-{role}-{index:03d}",
    )
    return ScenarioSpec(
        scenario_id=f"coverage-{role}-combined-{index:03d}",
        family=ScenarioFamily.COMBINED_PALLET_WORKER,
        split=f"coverage_{role}",
        width=11,
        height=9,
        robot_x=robot_x,
        robot_y=8,
        goal_columns=(4, 5, 6),
        actors=(
            ActorSpec(
                actor_id=f"pallet-{role}-{index:03d}",
                kind="pallet",
                x=robot_x,
                y=pallet_y,
                dx=0,
            ),
            worker,
        ),
    )


def build_manifests() -> dict[str, tuple[ScenarioSpec, ...]]:
    train_base = tuple(
        _copy_base_spec(spec, "train", index)
        for index, spec in enumerate(manifest_for_split("train"))
    )
    dev_base = tuple(
        _copy_base_spec(spec, "dev", index)
        for index, spec in enumerate(manifest_for_split("validation"))
    )
    train_combined = tuple(
        _combined_spec("train", index, *parameters)
        for index, parameters in enumerate(TRAIN_COMBINED_PLAN)
    )
    dev_combined = tuple(
        _combined_spec("dev", index, *parameters)
        for index, parameters in enumerate(DEV_COMBINED_PLAN)
    )

    final_stationary = tuple(
        ScenarioSpec(
            scenario_id=f"coverage-final-stationary-{index:03d}",
            family=ScenarioFamily.STATIONARY_PALLET,
            split="coverage_final",
            width=11,
            height=9,
            robot_x=robot_x,
            robot_y=8,
            goal_columns=(4, 5, 6),
            actors=(
                ActorSpec(
                    actor_id=f"pallet-final-{index:03d}",
                    kind="pallet",
                    x=robot_x,
                    y=pallet_y,
                    dx=0,
                ),
            ),
        )
        for index, (robot_x, pallet_y) in enumerate(FINAL_STATIONARY_PLAN)
    )
    final_crossing = tuple(
        ScenarioSpec(
            scenario_id=f"coverage-final-crossing-{index:03d}",
            family=ScenarioFamily.CROSSING_WORKER,
            split="coverage_final",
            width=11,
            height=9,
            robot_x=robot_x,
            robot_y=8,
            goal_columns=(robot_x,),
            actors=(
                replace(
                    _crossing_worker(robot_x, worker_y, worker_variant, 11),
                    actor_id=f"worker-final-crossing-{index:03d}",
                ),
            ),
        )
        for index, (robot_x, worker_y, worker_variant) in enumerate(
            FINAL_CROSSING_PLAN
        )
    )
    final_combined = tuple(
        _combined_spec("final", index, *parameters)
        for index, parameters in enumerate(FINAL_COMBINED_PLAN)
    )
    return {
        "train": train_base + train_combined,
        "dev": dev_base + dev_combined,
        "final": final_stationary + final_crossing + final_combined,
    }


def split_rules() -> dict[str, dict[str, Any]]:
    common = {
        "grid": {"width": 11, "height": 9, "robot_start_row": 8},
        "geometry_signature": (
            "grid, robot start, goal columns, stationary actor positions, and moving actor aisles; "
            "scenario and actor IDs are excluded"
        ),
        "worker_trajectory_signature": (
            "grid width, actor kind, initial column, aisle row, and horizontal velocity; IDs and "
            "episode seeds are excluded"
        ),
    }
    return {
        "train": {
            **common,
            "base_scenarios": "physical layouts from legacy train: 20 stationary + 20 crossing",
            "combined_scenarios": 8,
            "combined_robot_columns": [3, 4, 5, 6, 7],
            "combined_pallet_rows": [2, 3],
            "combined_worker_rows": [4, 5],
        },
        "dev": {
            **common,
            "base_scenarios": "physical layouts from legacy validation: 6 stationary + 6 crossing",
            "combined_scenarios": 8,
            "combined_robot_columns": [3, 4, 5, 6, 7],
            "combined_pallet_rows": [4, 5],
            "combined_worker_rows": [2, 3],
        },
        "final": {
            **common,
            "scenarios": "8 stationary + 8 crossing + 16 combined",
            "stationary_pallet_rows": [1, 6, 7],
            "stationary_additional_robot_column": 2,
            "moving_worker_rows": [1, 7],
            "combined_pallet_rows": [1, 6, 7],
            "selection_constraint": (
                "every geometry and worker trajectory signature is absent from train and dev"
            ),
        },
    }


def manifest_payload(
    role: str,
    specs: tuple[ScenarioSpec, ...],
    rule: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "manifest_schema_version": VERSIONED_MANIFEST_SCHEMA,
        "role": role,
        "split_rule": rule,
        "scenario_digest": manifest_digest(specs),
        "geometry_signature_digest": geometry_signature_digest(specs),
        "worker_trajectory_signature_digest": worker_trajectory_signature_digest(specs),
        "scenarios": [spec.to_dict() for spec in specs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {role: args.output_dir / f"{role}.json" for role in ("train", "dev", "final")}
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite frozen manifests: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = build_manifests()
    rules = split_rules()
    for role, path in paths.items():
        payload = manifest_payload(role, manifests[role], rules[role])
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{role}: {len(manifests[role])} scenarios {payload['scenario_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
