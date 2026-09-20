"""Tests for deterministic planning and typed oracle labels."""

from __future__ import annotations

import unittest

from laya_warehouse.model import Action, Actor, World
from laya_warehouse.oracle import direct_path_blocked, shortest_collision_free_plan


class OracleTests(unittest.TestCase):
    def test_symmetric_route_uses_documented_left_before_right_tie_break(self) -> None:
        world = World(
            robot_x=5,
            robot_y=2,
            goal_columns=(4, 5, 6),
            actors=[Actor("pallet", "pallet", 5, 1, 0)],
        )

        first = shortest_collision_free_plan(world)
        second = shortest_collision_free_plan(world)

        self.assertEqual(first, second)
        self.assertEqual(first.actions[0], Action.SHIFT_LEFT)
        self.assertEqual(first.steps, 3)
        self.assertEqual(first.lateral_moves, 1)
        self.assertEqual(first.waits, 0)

    def test_path_blocked_is_true_for_a_worker_entering_the_next_cell(self) -> None:
        world = World(
            robot_x=5,
            robot_y=5,
            actors=[Actor("worker", "worker", 4, 4, 1)],
        )

        self.assertTrue(direct_path_blocked(world))

    def test_path_blocked_is_false_when_three_forward_ticks_are_clear(self) -> None:
        world = World(
            robot_x=5,
            robot_y=5,
            actors=[Actor("worker", "worker", 0, 4, 1)],
        )

        self.assertFalse(direct_path_blocked(world))

    def test_path_blocked_is_true_at_row_zero_outside_a_goal_column(self) -> None:
        world = World(robot_x=3, robot_y=0, goal_columns=(4, 5, 6))

        self.assertTrue(direct_path_blocked(world))


if __name__ == "__main__":
    unittest.main()
