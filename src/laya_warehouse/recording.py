"""Run recording and deterministic replay verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .controllers import Controller
from .model import Action, World
from .oracle import oracle_labels
from .scenarios import ScenarioSpec


def run_episode(
    controller: Controller,
    *,
    max_ticks: int = 100,
    safety_shield: bool = True,
    scenario: ScenarioSpec | None = None,
    include_evaluation: bool = False,
) -> dict[str, Any]:
    world = scenario.build_world() if scenario else World.default()
    frames = []
    while world.status == "running" and world.tick < max_ticks:
        before = world.snapshot()
        observation = world.observe()
        decision = controller.decide(observation)
        requested_safety = world.move_safety(decision.action)
        labels = oracle_labels(world) if include_evaluation else None
        result = world.step(decision.action, safety_shield=safety_shield)
        frame = {
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
        if labels is not None:
            frame["evaluation"] = {
                "oracle_action": labels["action"],
                "path_blocked": labels["path_blocked"],
                "unsafe_request": not requested_safety["safe"],
                "unsafe_reason": (
                    None if requested_safety["safe"] else requested_safety["reason"]
                ),
            }
        frames.append(frame)

    record = {
        "schema_version": 2 if scenario else 1,
        "scenario": scenario.to_dict() if scenario else "crossing-v1",
        "controller": controller.name,
        "safety_shield": safety_shield,
        "max_ticks": max_ticks,
        "outcome": world.status if world.status != "running" else "timeout",
        "ticks": world.tick,
        "frames": frames,
    }
    if scenario:
        record["evaluation_included"] = include_evaluation
    return record


def save_record(record: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_record(record: dict[str, Any]) -> None:
    schema_version = record.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("unsupported record schema")
    if schema_version == 1:
        if record.get("scenario") != "crossing-v1":
            raise ValueError("unsupported scenario")
        world = World.default()
    else:
        world = ScenarioSpec.from_dict(record["scenario"]).build_world()
    for index, frame in enumerate(record["frames"]):
        if world.snapshot() != frame["before"]:
            raise ValueError(f"frame {index}: recorded before-state does not match replay")
        if schema_version == 2 and world.observe() != frame["observation"]:
            raise ValueError(f"frame {index}: recorded observation does not match replay")
        requested = Action(frame["step"]["requested_action"])
        if frame["decision"]["action"] != requested.value:
            raise ValueError(f"frame {index}: decision and requested action do not match")
        if schema_version == 2 and record.get("evaluation_included"):
            labels = oracle_labels(world)
            safety = world.move_safety(requested)
            expected_evaluation = {
                "oracle_action": labels["action"],
                "path_blocked": labels["path_blocked"],
                "unsafe_request": not safety["safe"],
                "unsafe_reason": None if safety["safe"] else safety["reason"],
            }
            if frame.get("evaluation") != expected_evaluation:
                raise ValueError(f"frame {index}: evaluation does not match replay")
        result = world.step(requested, safety_shield=record["safety_shield"])
        expected_step = {
            "requested_action": result.requested_action,
            "applied_action": result.applied_action,
            "safety_override": result.safety_override,
            "status": result.status,
            "reason": result.reason,
        }
        if frame["step"] != expected_step:
            raise ValueError(f"frame {index}: recorded step does not match replay")
        if world.snapshot() != frame["after"]:
            raise ValueError(f"frame {index}: recorded after-state does not match replay")

    expected = world.status if world.status != "running" else "timeout"
    if expected != record["outcome"]:
        raise ValueError(f"recorded outcome {record['outcome']!r} does not match {expected!r}")
    if world.tick != record["ticks"]:
        raise ValueError(f"recorded tick count {record['ticks']!r} does not match {world.tick!r}")
