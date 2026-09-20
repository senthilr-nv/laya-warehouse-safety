"""Deterministic planning and labels kept outside model observations."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from .model import Action, World


ORACLE_ACTION_ORDER = (
    Action.ADVANCE,
    Action.SHIFT_LEFT,
    Action.SHIFT_RIGHT,
    Action.WAIT,
)
PATH_BLOCKED_HORIZON = 3


@dataclass(frozen=True)
class OraclePlan:
    actions: tuple[Action, ...]
    steps: int
    lateral_moves: int
    waits: int

    @property
    def action(self) -> Action:
        return self.actions[0] if self.actions else Action.WAIT


def _state_key(world: World) -> tuple:
    return (
        world.robot_x,
        world.robot_y,
        tuple((actor.x, actor.y, actor.dx) for actor in world.actors),
        world.status,
    )


def shortest_collision_free_plan(world: World, *, max_expansions: int = 50_000) -> OraclePlan:
    """Find a route using a documented lexicographic cost.

    Plans minimize total steps, then lateral moves, then waits. Remaining ties use
    ``advance, shift_left, shift_right, wait`` lexicographic action order. Only actions
    accepted by the deterministic one-tick safety model are expanded.
    """

    if world.status == "completed" or (
        world.robot_y == 0 and world.robot_x in world.goal_columns
    ):
        return OraclePlan((), 0, 0, 0)

    action_rank = {action: index for index, action in enumerate(ORACLE_ACTION_ORDER)}
    counter = itertools.count()
    start = world.clone()
    start_cost = (0, 0, 0, ())
    queue: list[tuple[tuple, int, World, tuple[Action, ...]]] = [
        (start_cost, next(counter), start, ())
    ]
    best = {_state_key(start): start_cost}
    expansions = 0

    while queue:
        cost, _, current, actions = heapq.heappop(queue)
        if cost != best.get(_state_key(current)):
            continue
        if current.status == "completed":
            return OraclePlan(actions, cost[0], cost[1], cost[2])
        expansions += 1
        if expansions > max_expansions:
            raise RuntimeError("oracle search exceeded the expansion limit")

        for action in ORACLE_ACTION_ORDER:
            if not current.move_safety(action)["safe"]:
                continue
            candidate = current.clone()
            candidate.step(action, safety_shield=False)
            if candidate.status == "collision":
                continue
            candidate_actions = actions + (action,)
            candidate_cost = (
                len(candidate_actions),
                cost[1] + int(action in (Action.SHIFT_LEFT, Action.SHIFT_RIGHT)),
                cost[2] + int(action is Action.WAIT),
                cost[3] + (action_rank[action],),
            )
            key = _state_key(candidate)
            if key in best and best[key] <= candidate_cost:
                continue
            best[key] = candidate_cost
            heapq.heappush(
                queue,
                (candidate_cost, next(counter), candidate, candidate_actions),
            )

    raise RuntimeError("scenario has no collision-free route to the loading bay")


def direct_path_blocked(world: World, *, horizon: int = PATH_BLOCKED_HORIZON) -> bool:
    """Whether forward-only travel is blocked within the next ``horizon`` ticks."""

    probe = world.clone()
    for _ in range(horizon):
        if probe.robot_y == 0:
            return probe.robot_x not in probe.goal_columns
        if not probe.move_safety(Action.ADVANCE)["safe"]:
            return True
        probe.step(Action.ADVANCE, safety_shield=False)
        if probe.status == "collision":
            return True
        if probe.status == "completed":
            return False
    return False


def oracle_labels(world: World) -> dict[str, object]:
    plan = shortest_collision_free_plan(world)
    return {
        "action": plan.action.value,
        "path_blocked": direct_path_blocked(world),
        "oracle_steps": plan.steps,
    }
