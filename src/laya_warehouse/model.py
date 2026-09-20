"""Deterministic grid simulation and structured Laya observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Action(str, Enum):
    """Actions available to the warehouse robot."""

    ADVANCE = "advance"
    SHIFT_LEFT = "shift_left"
    SHIFT_RIGHT = "shift_right"
    WAIT = "wait"


@dataclass
class Actor:
    """A moving warehouse actor that follows one horizontal aisle."""

    actor_id: str
    kind: str
    x: int
    y: int
    dx: int

    def forecast(self, width: int) -> tuple[int, int, int]:
        next_x = self.x + self.dx
        next_dx = self.dx
        if next_x < 0 or next_x >= width:
            next_dx = -self.dx
            next_x = self.x + next_dx
        return next_x, self.y, next_dx

    def advance(self, width: int) -> None:
        self.x, self.y, self.dx = self.forecast(width)


@dataclass
class StepResult:
    """Observable result of one simulation step."""

    requested_action: str
    applied_action: str
    safety_override: bool
    status: str
    reason: str


@dataclass
class World:
    """One robot crossing warehouse traffic toward a loading bay."""

    width: int = 11
    height: int = 9
    robot_x: int = 5
    robot_y: int = 8
    goal_columns: tuple[int, ...] = (4, 5, 6)
    actors: list[Actor] = field(default_factory=list)
    tick: int = 0
    status: str = "running"

    @classmethod
    def default(cls) -> World:
        return cls(
            actors=[
                Actor("worker-a", "worker", 1, 6, 1),
                Actor("forklift-a", "forklift", 9, 4, -1),
                Actor("worker-b", "worker", 2, 2, 1),
            ]
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "status": self.status,
            "width": self.width,
            "height": self.height,
            "robot": {"x": self.robot_x, "y": self.robot_y},
            "goal": {"row": 0, "columns": list(self.goal_columns)},
            "actors": [asdict(actor) for actor in self.actors],
        }

    def _target(self, action: Action) -> tuple[int, int]:
        if action is Action.ADVANCE:
            return self.robot_x, self.robot_y - 1
        if action is Action.SHIFT_LEFT:
            return self.robot_x - 1, self.robot_y
        if action is Action.SHIFT_RIGHT:
            return self.robot_x + 1, self.robot_y
        return self.robot_x, self.robot_y

    def _forecast_actor_positions(self) -> dict[tuple[int, int], Actor]:
        return {(actor.forecast(self.width)[0], actor.y): actor for actor in self.actors}

    def move_safety(self, action: Action) -> dict[str, Any]:
        target = self._target(action)
        in_bounds = 0 <= target[0] < self.width and 0 <= target[1] < self.height
        actor_now = next((a for a in self.actors if (a.x, a.y) == target), None)
        actor_next = self._forecast_actor_positions().get(target)

        if not in_bounds:
            return {"safe": False, "reason": "outside warehouse grid"}
        if actor_now:
            return {"safe": False, "reason": f"{actor_now.kind} occupies target cell"}
        if actor_next:
            return {"safe": False, "reason": f"{actor_next.kind} will enter target cell"}
        return {"safe": True, "reason": "clear now and next tick"}

    def observe(self) -> dict[str, Any]:
        candidate_moves = {
            action.value: self.move_safety(action)
            for action in (Action.ADVANCE, Action.SHIFT_LEFT, Action.SHIFT_RIGHT, Action.WAIT)
        }
        hazards = []
        for actor in self.actors:
            distance = abs(actor.x - self.robot_x) + abs(actor.y - self.robot_y)
            if distance <= 5:
                next_x, next_y, _ = actor.forecast(self.width)
                next_distance = abs(next_x - self.robot_x) + abs(next_y - self.robot_y)
                hazards.append(
                    {
                        "id": actor.actor_id,
                        "kind": actor.kind,
                        "relative_column": actor.x - self.robot_x,
                        "rows_ahead": self.robot_y - actor.y,
                        "distance": "near" if distance <= 2 else "visible",
                        "motion": "approaching" if next_distance < distance else "moving away",
                    }
                )

        return {
            "tick": self.tick,
            "task": "Reach the loading bay on row 0 without colliding with workers or forklifts.",
            "robot": {"column": self.robot_x, "row": self.robot_y},
            "goal": {"row": 0, "columns": list(self.goal_columns)},
            "candidate_moves": candidate_moves,
            "visible_hazards": hazards,
        }

    def step(self, requested: Action, *, safety_shield: bool = True) -> StepResult:
        if self.status != "running":
            raise RuntimeError(f"cannot step a world with status {self.status!r}")

        applied = requested
        safety = self.move_safety(requested)
        safety_override = False
        reason = safety["reason"]
        if not safety["safe"] and "outside warehouse grid" in reason:
            applied = Action.WAIT
            safety_override = applied is not requested
            reason = f"boundary guard replaced {requested.value}: {reason}"
        elif safety_shield and not safety["safe"]:
            original_reason = reason
            safe_fallback = next(
                (
                    candidate
                    for candidate in (
                        Action.WAIT,
                        Action.SHIFT_LEFT,
                        Action.SHIFT_RIGHT,
                        Action.ADVANCE,
                    )
                    if self.move_safety(candidate)["safe"]
                ),
                Action.WAIT,
            )
            applied = safe_fallback
            safety_override = applied is not requested
            reason = (
                f"safety shield replaced {requested.value} with {applied.value}: "
                f"{original_reason}"
            )

        target = self._target(applied)
        for actor in self.actors:
            actor.advance(self.width)
        self.robot_x, self.robot_y = target
        self.tick += 1

        collision = next(
            (actor for actor in self.actors if (actor.x, actor.y) == (self.robot_x, self.robot_y)),
            None,
        )
        if collision:
            self.status = "collision"
            reason = f"collision with {collision.actor_id}"
        elif self.robot_y == 0 and self.robot_x in self.goal_columns:
            self.status = "completed"
            reason = "robot reached the loading bay"

        return StepResult(
            requested_action=requested.value,
            applied_action=applied.value,
            safety_override=safety_override,
            status=self.status,
            reason=reason,
        )
