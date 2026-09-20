"""Leak-free typed-decision datasets generated from frozen scenario manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .model import Action
from .oracle import oracle_labels
from .scenarios import manifest_digest, manifest_for_split


def generate_cases(*, split: str = "train", max_ticks: int = 30) -> list[dict[str, Any]]:
    """Roll out the oracle and label raw observations for one frozen split."""

    cases: list[dict[str, Any]] = []
    for scenario in manifest_for_split(split):
        world = scenario.build_world()
        while world.status == "running" and world.tick < max_ticks:
            labels = oracle_labels(world)
            cases.append(
                {
                    "id": f"{scenario.scenario_id}-tick-{world.tick:03d}",
                    "scenario_id": scenario.scenario_id,
                    "family": scenario.family.value,
                    "split": scenario.split,
                    "state": world.observe(),
                    "labels": {
                        "action": labels["action"],
                        "path_blocked": labels["path_blocked"],
                    },
                    "source": "frozen deterministic simulator manifest",
                }
            )
            world.step(Action(labels["action"]), safety_shield=False)
        if world.status != "completed":
            raise RuntimeError(f"oracle did not complete scenario {scenario.scenario_id}")
    return cases


def dataset_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(case["labels"]["action"] for case in cases)
    path_counts = Counter(str(case["labels"]["path_blocked"]).lower() for case in cases)
    family_counts = Counter(case["family"] for case in cases)
    splits = sorted({case["split"] for case in cases})
    specs = tuple(spec for split in splits for spec in manifest_for_split(split))
    return {
        "cases": len(cases),
        "splits": splits,
        "families": dict(sorted(family_counts.items())),
        "labels": {
            "action": dict(sorted(action_counts.items())),
            "path_blocked": dict(sorted(path_counts.items())),
        },
        "manifest_digest": manifest_digest(specs),
    }


def save_cases(cases: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, separators=(",", ":"), sort_keys=True) + "\n")
