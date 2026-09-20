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

    ACTION_QUESTIONS = {
        Action.ADVANCE: "advance_is_best",
        Action.SHIFT_LEFT: "shift_left_is_best",
        Action.SHIFT_RIGHT: "shift_right_is_best",
        Action.WAIT: "wait_is_best",
    }

    QUESTIONS = {
        "advance_is_best": {
            "type": "noul",
            "instructions": (
                "Is advance the best safe next action? Answer true only when "
                "candidate_moves.advance.safe is true and advancing improves progress to the goal."
            ),
        },
        "shift_left_is_best": {
            "type": "noul",
            "instructions": (
                "Is shift_left the best safe next action? Answer true only when "
                "candidate_moves.shift_left.safe is true and shifting left avoids a blocked path "
                "or moves the robot toward a goal column."
            ),
        },
        "shift_right_is_best": {
            "type": "noul",
            "instructions": (
                "Is shift_right the best safe next action? Answer true only when "
                "candidate_moves.shift_right.safe is true and shifting right avoids a blocked path "
                "or moves the robot toward a goal column."
            ),
        },
        "wait_is_best": {
            "type": "noul",
            "instructions": (
                "Is wait the best safe next action? Answer true when traffic should pass before "
                "the robot moves and candidate_moves.wait.safe is true."
            ),
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
        action_scores = {
            action: answers[question_id]["noul"]
            for action, question_id in self.ACTION_QUESTIONS.items()
        }
        action = max(action_scores, key=action_scores.get)
        return Decision(
            action=action,
            latency_ms=latency_ms,
            details={
                "action_scores": {item.value: score for item, score in action_scores.items()},
                "answers": answers,
                "usage": result["usage"],
            },
        )
