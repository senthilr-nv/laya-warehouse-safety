"""Immutable warehouse scenario specifications and frozen benchmark manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from .model import Actor, World


MANIFEST_VERSION = 1


class ScenarioFamily(str, Enum):
    STATIONARY_PALLET = "stationary_pallet"
    CROSSING_WORKER = "crossing_worker"
    COMBINED_PALLET_WORKER = "combined_pallet_worker"


@dataclass(frozen=True)
class ActorSpec:
    actor_id: str
    kind: str
    x: int
    y: int
    dx: int


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: ScenarioFamily
    split: str
    width: int
    height: int
    robot_x: int
    robot_y: int
    goal_columns: tuple[int, ...]
    actors: tuple[ActorSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family"] = self.family.value
        payload["goal_columns"] = list(self.goal_columns)
        payload["actors"] = [asdict(actor) for actor in self.actors]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScenarioSpec:
        return cls(
            scenario_id=payload["scenario_id"],
            family=ScenarioFamily(payload["family"]),
            split=payload["split"],
            width=int(payload["width"]),
            height=int(payload["height"]),
            robot_x=int(payload["robot_x"]),
            robot_y=int(payload["robot_y"]),
            goal_columns=tuple(int(value) for value in payload["goal_columns"]),
            actors=tuple(ActorSpec(**actor) for actor in payload["actors"]),
        )

    def build_world(self) -> World:
        return World(
            width=self.width,
            height=self.height,
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            goal_columns=self.goal_columns,
            actors=[Actor(**asdict(actor)) for actor in self.actors],
        )


def _actor_x_after(*, start_x: int, dx: int, ticks: int, width: int) -> int:
    actor = Actor("probe", "worker", start_x, 0, dx)
    for _ in range(ticks):
        actor.advance(width)
    return actor.x


def _crossing_worker(robot_x: int, row: int, variant: int, width: int) -> ActorSpec:
    """Place a worker so its forecast intersects a straight robot route."""

    arrival_tick = 8 - row
    direction_order = (1, -1) if variant % 2 == 0 else (-1, 1)
    candidates = range(width) if variant % 3 else range(width - 1, -1, -1)
    for dx in direction_order:
        for start_x in candidates:
            if _actor_x_after(
                start_x=start_x,
                dx=dx,
                ticks=arrival_tick,
                width=width,
            ) == robot_x:
                return ActorSpec(
                    actor_id=f"worker-{variant:03d}",
                    kind="worker",
                    x=start_x,
                    y=row,
                    dx=dx,
                )
    raise RuntimeError("could not construct a crossing worker")


def _stationary_spec(split: str, index: int, variant: int) -> ScenarioSpec:
    robot_x = 3 + variant % 5
    pallet_y = 2 + (variant // 2) % 4
    return ScenarioSpec(
        scenario_id=f"stationary-pallet-{split}-{index:03d}",
        family=ScenarioFamily.STATIONARY_PALLET,
        split=split,
        width=11,
        height=9,
        robot_x=robot_x,
        robot_y=8,
        goal_columns=(4, 5, 6),
        actors=(
            ActorSpec(
                actor_id=f"pallet-{variant:03d}",
                kind="pallet",
                x=robot_x,
                y=pallet_y,
                dx=0,
            ),
        ),
    )


def _crossing_spec(split: str, index: int, variant: int) -> ScenarioSpec:
    robot_x = 4 + variant % 3
    worker_y = 2 + (variant // 2) % 5
    return ScenarioSpec(
        scenario_id=f"crossing-worker-{split}-{index:03d}",
        family=ScenarioFamily.CROSSING_WORKER,
        split=split,
        width=11,
        height=9,
        robot_x=robot_x,
        robot_y=8,
        goal_columns=(robot_x,),
        actors=(_crossing_worker(robot_x, worker_y, variant, 11),),
    )


def _combined_spec(split: str, index: int, variant: int) -> ScenarioSpec:
    robot_x = 3 + variant % 5
    pallet_y = 2 + variant % 3
    worker_y = 5 + (variant // 3) % 2
    worker = _crossing_worker(robot_x, worker_y, variant, 11)
    return ScenarioSpec(
        scenario_id=f"combined-pallet-worker-{split}-{index:03d}",
        family=ScenarioFamily.COMBINED_PALLET_WORKER,
        split=split,
        width=11,
        height=9,
        robot_x=robot_x,
        robot_y=8,
        goal_columns=(4, 5, 6),
        actors=(
            ActorSpec(
                actor_id=f"pallet-{variant:03d}",
                kind="pallet",
                x=robot_x,
                y=pallet_y,
                dx=0,
            ),
            worker,
        ),
    )


def _build_manifest() -> tuple[ScenarioSpec, ...]:
    split_plan = (
        ("train", 20, (ScenarioFamily.STATIONARY_PALLET, ScenarioFamily.CROSSING_WORKER)),
        ("validation", 6, (ScenarioFamily.STATIONARY_PALLET, ScenarioFamily.CROSSING_WORKER)),
        ("iid_test", 8, (ScenarioFamily.STATIONARY_PALLET, ScenarioFamily.CROSSING_WORKER)),
        ("ood_test", 16, (ScenarioFamily.COMBINED_PALLET_WORKER,)),
    )
    offsets = {"train": 0, "validation": 100, "iid_test": 200, "ood_test": 300}
    manifest: list[ScenarioSpec] = []
    for split, count, families in split_plan:
        for family in families:
            for index in range(count):
                variant = offsets[split] + index
                if family is ScenarioFamily.STATIONARY_PALLET:
                    spec = _stationary_spec(split, index, variant)
                elif family is ScenarioFamily.CROSSING_WORKER:
                    spec = _crossing_spec(split, index, variant)
                else:
                    spec = _combined_spec(split, index, variant)
                manifest.append(spec)
    return tuple(manifest)


FROZEN_MANIFEST = _build_manifest()


def manifest_for_split(split: str) -> tuple[ScenarioSpec, ...]:
    specs = tuple(spec for spec in FROZEN_MANIFEST if spec.split == split)
    if not specs:
        raise ValueError(f"unknown or empty scenario split: {split}")
    return specs


def benchmark_manifest() -> tuple[ScenarioSpec, ...]:
    return manifest_for_split("iid_test") + manifest_for_split("ood_test")


def manifest_digest(specs: Iterable[ScenarioSpec] = FROZEN_MANIFEST) -> str:
    payload = [spec.to_dict() for spec in specs]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
