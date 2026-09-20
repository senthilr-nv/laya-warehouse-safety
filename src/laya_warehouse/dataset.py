"""Deterministic synthetic training cases for the warehouse policy."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .model import Action, Actor, World


def _advance_world(rng: random.Random, index: int) -> World:
    robot_x = rng.choice((4, 5, 6))
    robot_y = rng.randint(3, 8)
    offset = rng.choice((-2, 2))
    pallet_x = max(0, min(10, robot_x + offset))
    pallet_y = max(0, robot_y - rng.choice((2, 3)))
    return World(
        robot_x=robot_x,
        robot_y=robot_y,
        tick=index,
        actors=[Actor(f"pallet-{index}", "pallet", pallet_x, pallet_y, 0)],
    )


def _lateral_world(action: Action, rng: random.Random, index: int) -> World:
    if action is Action.SHIFT_LEFT:
        robot_x = rng.randint(7, 8)
    else:
        robot_x = rng.randint(2, 3)
    robot_y = rng.randint(4, 8)
    pallet_y = robot_y - rng.choice((2, 3))
    return World(
        robot_x=robot_x,
        robot_y=robot_y,
        tick=index,
        actors=[Actor(f"pallet-{index}", "pallet", robot_x, pallet_y, 0)],
    )


def _wait_world(rng: random.Random, index: int) -> World:
    robot_x = rng.randint(3, 7)
    robot_y = rng.randint(2, 8)
    approach_from_left = rng.choice((True, False))
    actor_x = robot_x - 1 if approach_from_left else robot_x + 1
    actor_dx = 1 if approach_from_left else -1
    kind = rng.choice(("worker", "forklift"))
    return World(
        robot_x=robot_x,
        robot_y=robot_y,
        tick=index,
        actors=[Actor(f"{kind}-{index}", kind, actor_x, robot_y - 1, actor_dx)],
    )


def generate_cases(*, per_action: int = 128, seed: int = 0) -> list[dict[str, Any]]:
    """Generate balanced labeled observations without external data."""

    if per_action < 1:
        raise ValueError("per_action must be at least 1")

    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for action in Action:
        for index in range(per_action):
            case_index = len(cases)
            if action is Action.ADVANCE:
                world = _advance_world(rng, case_index)
            elif action is Action.WAIT:
                world = _wait_world(rng, case_index)
            else:
                world = _lateral_world(action, rng, case_index)
            cases.append(
                {
                    "id": f"synthetic-{seed}-{action.value}-{index:04d}",
                    "state": world.observe(),
                    "label": action.value,
                    "source": "deterministic simulator",
                }
            )

    rng.shuffle(cases)
    return cases


def dataset_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(case["label"] for case in cases)
    return {"cases": len(cases), "labels": dict(sorted(counts.items()))}


def save_cases(cases: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, separators=(",", ":")) + "\n")
