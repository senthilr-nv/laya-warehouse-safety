"""Run recording and deterministic replay verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .controllers import Controller
from .model import Action, World


def run_episode(
    controller: Controller,
    *,
    max_ticks: int = 100,
    safety_shield: bool = True,
) -> dict[str, Any]:
    world = World.default()
    frames = []
    while world.status == "running" and world.tick < max_ticks:
        before = world.snapshot()
        observation = world.observe()
        decision = controller.decide(observation)
        result = world.step(decision.action, safety_shield=safety_shield)
        frames.append(
            {
                "before": before,
                "observation": observation,
                "decision": {
                    "action": decision.action.value,
                    "latency_ms": round(decision.latency_ms, 4),
                    "details": decision.details,
                },
                "step": {
                    "requested_action": result.requested_action,
                    "applied_action": result.applied_action,
                    "safety_override": result.safety_override,
                    "status": result.status,
                    "reason": result.reason,
                },
                "after": world.snapshot(),
            }
        )

    return {
        "schema_version": 1,
        "scenario": "crossing-v1",
        "controller": controller.name,
        "safety_shield": safety_shield,
        "max_ticks": max_ticks,
        "outcome": world.status if world.status != "running" else "timeout",
        "ticks": world.tick,
        "frames": frames,
    }


def save_record(record: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_record(record: dict[str, Any]) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("unsupported record schema")
    if record.get("scenario") != "crossing-v1":
        raise ValueError("unsupported scenario")

    world = World.default()
    for index, frame in enumerate(record["frames"]):
        if world.snapshot() != frame["before"]:
            raise ValueError(f"frame {index}: recorded before-state does not match replay")
        requested = Action(frame["step"]["requested_action"])
        world.step(requested, safety_shield=record["safety_shield"])
        if world.snapshot() != frame["after"]:
            raise ValueError(f"frame {index}: recorded after-state does not match replay")

    expected = world.status if world.status != "running" else "timeout"
    if expected != record["outcome"]:
        raise ValueError(f"recorded outcome {record['outcome']!r} does not match {expected!r}")

