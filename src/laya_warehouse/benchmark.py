"""Deterministic multi-scenario benchmark aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .controllers import Controller
from .model import Action
from .oracle import ORACLE_ACTION_ORDER, PATH_BLOCKED_HORIZON
from .recording import run_episode, verify_record
from .scenarios import (
    MANIFEST_VERSION,
    ScenarioSpec,
    benchmark_manifest,
    manifest_digest,
)


REPORT_SCHEMA_VERSION = 1
ControllerFactory = Callable[[ScenarioSpec], Controller]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def _episode_summary(record: dict[str, Any]) -> tuple[dict[str, Any], list[float], bool]:
    frames = record["frames"]
    unsafe_requests = sum(frame["evaluation"]["unsafe_request"] for frame in frames)
    boundary_interventions = sum(
        frame["step"]["safety_override"]
        and frame["evaluation"]["unsafe_reason"] == "outside warehouse grid"
        for frame in frames
    )
    interventions = sum(
        frame["step"]["safety_override"]
        and frame["evaluation"]["unsafe_reason"] != "outside warehouse grid"
        for frame in frames
    )
    unnecessary_waits = sum(
        frame["decision"]["action"] == Action.WAIT.value
        and frame["evaluation"]["oracle_action"] != Action.WAIT.value
        for frame in frames
    )
    oracle_correct = sum(
        frame["decision"]["action"] == frame["evaluation"]["oracle_action"]
        for frame in frames
    )
    route_length = sum(
        frame["step"]["applied_action"] != Action.WAIT.value for frame in frames
    )
    blocked_predictions = []
    is_laya = False
    for frame in frames:
        details = frame["decision"]["details"]
        is_laya = is_laya or details.get("runtime") == "laya"
        answer = details.get("answers", {}).get("path_blocked")
        if answer and "noul" in answer:
            blocked_predictions.append(
                (float(answer["noul"]), bool(frame["evaluation"]["path_blocked"]))
            )
    blocked_correct = sum(
        (probability >= 0.5) == target for probability, target in blocked_predictions
    )
    blocked_brier = sum(
        (probability - int(target)) ** 2 for probability, target in blocked_predictions
    )
    scenario = record["scenario"]
    summary = {
        "scenario_id": scenario["scenario_id"],
        "family": scenario["family"],
        "track": "layered" if record["safety_shield"] else "policy_only",
        "outcome": record["outcome"],
        "ticks": record["ticks"],
        "decisions": len(frames),
        "unsafe_requests": unsafe_requests,
        "boundary_guard_interventions": boundary_interventions,
        "shield_interventions": interventions,
        "unnecessary_waits": unnecessary_waits,
        "route_length": route_length,
        "oracle_correct": oracle_correct,
        "path_blocked_predictions": len(blocked_predictions),
        "path_blocked_correct": blocked_correct,
        "path_blocked_brier_sum": round(blocked_brier, 6),
    }
    latencies = [float(frame["decision"]["latency_ms"]) for frame in frames]
    return summary, latencies, is_laya


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = len(episodes)
    decisions = sum(episode["decisions"] for episode in episodes)
    blocked_predictions = sum(episode["path_blocked_predictions"] for episode in episodes)
    outcomes = Counter(episode["outcome"] for episode in episodes)
    return {
        "episodes": episode_count,
        "completion_rate": round(outcomes["completed"] / episode_count, 4),
        "collision_rate": round(outcomes["collision"] / episode_count, 4),
        "timeout_rate": round(outcomes["timeout"] / episode_count, 4),
        "decisions": decisions,
        "unsafe_requests": sum(episode["unsafe_requests"] for episode in episodes),
        "unsafe_request_rate": round(
            sum(episode["unsafe_requests"] for episode in episodes) / decisions, 4
        ),
        "shield_interventions": sum(
            episode["shield_interventions"] for episode in episodes
        ),
        "boundary_guard_interventions": sum(
            episode["boundary_guard_interventions"] for episode in episodes
        ),
        "unnecessary_waits": sum(episode["unnecessary_waits"] for episode in episodes),
        "unnecessary_wait_rate": round(
            sum(episode["unnecessary_waits"] for episode in episodes) / decisions, 4
        ),
        "mean_route_length": round(
            sum(episode["route_length"] for episode in episodes) / episode_count, 4
        ),
        "mean_ticks": round(
            sum(episode["ticks"] for episode in episodes) / episode_count, 4
        ),
        "oracle_action_accuracy": round(
            sum(episode["oracle_correct"] for episode in episodes) / decisions, 4
        ),
        "path_blocked": (
            {
                "predictions": blocked_predictions,
                "accuracy": round(
                    sum(episode["path_blocked_correct"] for episode in episodes)
                    / blocked_predictions,
                    4,
                ),
                "brier": round(
                    sum(episode["path_blocked_brier_sum"] for episode in episodes)
                    / blocked_predictions,
                    4,
                ),
            }
            if blocked_predictions
            else None
        ),
    }


def _stable_seed(scenario_id: str) -> int:
    digest = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_benchmark(
    controller_factories: dict[str, ControllerFactory],
    *,
    specs: tuple[ScenarioSpec, ...] | None = None,
    max_ticks: int = 40,
    model_load_ms: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Run every controller over the same manifests with and without the shield."""

    specs = benchmark_manifest() if specs is None else specs
    if not specs:
        raise ValueError("benchmark requires at least one scenario")
    if not controller_factories:
        raise ValueError("benchmark requires at least one controller")
    if max_ticks < 1:
        raise ValueError("max_ticks must be at least 1")
    selected_records: dict[str, dict[str, Any]] = {}
    results: dict[str, Any] = {}
    runtime_evidence: dict[str, Any] = {}
    for controller_name in sorted(controller_factories):
        track_results: dict[str, Any] = {}
        controller_latencies: list[float] = []
        controller_is_laya = False
        for track, safety_shield in (("policy_only", False), ("layered", True)):
            episode_summaries = []
            selected_families: set[str] = set()
            for spec in specs:
                controller = controller_factories[controller_name](spec)
                record = run_episode(
                    controller,
                    max_ticks=max_ticks,
                    safety_shield=safety_shield,
                    scenario=spec,
                    include_evaluation=True,
                )
                verify_record(record)
                summary, latencies, is_laya = _episode_summary(record)
                episode_summaries.append(summary)
                controller_latencies.extend(latencies)
                controller_is_laya = controller_is_laya or is_laya
                if spec.family.value not in selected_families:
                    key = f"{controller_name}-{track}-{spec.family.value}"
                    selected_records[key] = record
                    selected_families.add(spec.family.value)

            families = sorted({episode["family"] for episode in episode_summaries})
            track_results[track] = {
                "overall": _aggregate(episode_summaries),
                "by_family": {
                    family: _aggregate(
                        [episode for episode in episode_summaries if episode["family"] == family]
                    )
                    for family in families
                },
                "episodes": episode_summaries,
            }
        results[controller_name] = {"tracks": track_results}

        if controller_is_laya:
            cold = controller_latencies[0] if controller_latencies else None
            warm = controller_latencies[1:]
            runtime_evidence[controller_name] = {
                "model_load_ms": (
                    round(model_load_ms[controller_name], 4)
                    if model_load_ms and controller_name in model_load_ms
                    else None
                ),
                "cold_inference_ms": round(cold, 4) if cold is not None else None,
                "warm_inference_ms": {
                    "count": len(warm),
                    "p50": _percentile(warm, 0.5),
                    "p95": _percentile(warm, 0.95),
                },
            }
        else:
            runtime_evidence[controller_name] = {
                "decision_ms": {
                    "count": len(controller_latencies),
                    "p50": _percentile(controller_latencies, 0.5),
                    "p95": _percentile(controller_latencies, 0.95),
                }
            }

    scenario_counts = Counter(spec.family.value for spec in specs)
    deterministic_payload = {
        "configuration": {
            "max_ticks": max_ticks,
            "tracks": ["policy_only", "layered"],
            "path_blocked_horizon": PATH_BLOCKED_HORIZON,
            "oracle_tie_break": [action.value for action in ORACLE_ACTION_ORDER],
        },
        "manifest": {
            "version": MANIFEST_VERSION,
            "digest": manifest_digest(specs),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "scenario_ids": [spec.scenario_id for spec in specs],
        },
        "results": results,
    }
    canonical = json.dumps(deterministic_payload, sort_keys=True, separators=(",", ":"))
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": "warehouse-compositional-ood-v1",
        "training_families": ["stationary_pallet", "crossing_worker"],
        "held_out_family": "combined_pallet_worker",
        **deterministic_payload,
        "deterministic_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "runtime_evidence": runtime_evidence,
        "selected_episode_keys": sorted(selected_records),
    }
    return report, selected_records


def save_benchmark_report(report: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def random_seed_for_scenario(scenario: ScenarioSpec) -> int:
    return _stable_seed(scenario.scenario_id)
