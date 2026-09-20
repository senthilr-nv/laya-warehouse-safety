"""Robot controllers with one shared decision interface."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .model import Action


@dataclass
class Decision:
    action: Action
    latency_ms: float
    details: dict[str, Any]


class Controller(Protocol):
    name: str

    def decide(self, observation: dict[str, Any]) -> Decision: ...


class HeuristicController:
    """A deterministic baseline that uses the simulator's safety forecast."""

    name = "heuristic"

    def decide(self, observation: dict[str, Any]) -> Decision:
        start = time.perf_counter()
        moves = observation["candidate_moves"]
        robot_x = observation["robot"]["column"]
        goal_columns = observation["goal"]["columns"]

        preferences = [Action.ADVANCE]
        if robot_x < min(goal_columns):
            preferences.insert(0, Action.SHIFT_RIGHT)
        elif robot_x > max(goal_columns):
            preferences.insert(0, Action.SHIFT_LEFT)
        preferences.extend([Action.SHIFT_LEFT, Action.SHIFT_RIGHT, Action.WAIT])

        action = next(action for action in preferences if moves[action.value]["safe"])
        latency_ms = (time.perf_counter() - start) * 1000
        return Decision(action, latency_ms, {"policy": "first safe preferred move"})


class RandomController:
    """A seeded baseline that does not inspect the observation."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._random = random.Random(seed)

    def decide(self, observation: dict[str, Any]) -> Decision:
        del observation
        start = time.perf_counter()
        action = self._random.choice(list(Action))
        latency_ms = (time.perf_counter() - start) * 1000
        return Decision(action, latency_ms, {"policy": "seeded random"})


class LayaController:
    """Laya adapter that evaluates all warehouse questions in one forward pass."""

    name = "laya"

    QUESTIONS = {
        "action": {
            "type": "choice",
            "instructions": "Which action is the best safe next move toward the loading bay?",
            "criteria": {
                "advance": "move one row toward the loading bay when the route is clear",
                "shift_left": "move left around a stationary obstacle or toward a goal column",
                "shift_right": "move right around a stationary obstacle or toward a goal column",
                "wait": "hold position while temporary cross-aisle traffic passes",
            },
        },
        "collision_risk": {
            "type": "score",
            "instructions": "How risky is movement during the next tick?",
            "criteria": ["low", "moderate", "high", "critical"],
        },
        "path_blocked": {
            "type": "noul",
            "instructions": "Is the direct path toward the loading bay blocked?",
        },
        "needs_operator": {
            "type": "noul",
            "instructions": "Should a human operator review this situation?",
        },
    }

    def __init__(self, model: str = "convaiinnovations/laya", device: str | None = None):
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError(
                "The Laya controller requires the optional dependency. "
                "Install with: pip install -e '.[laya]'"
            ) from exc
        self._agent = laya.load(model, device=device)

    def decide(self, observation: dict[str, Any]) -> Decision:
        start = time.perf_counter()
        result = self._agent.predict(observation, self.QUESTIONS)
        latency_ms = (time.perf_counter() - start) * 1000
        answers = result["answers"]
        action_answer = answers["action"]
        action = Action(action_answer["choice"])
        return Decision(
            action=action,
            latency_ms=latency_ms,
            details={
                "action_scores": action_answer["probabilities"],
                "answers": answers,
                "usage": result["usage"],
            },
        )
