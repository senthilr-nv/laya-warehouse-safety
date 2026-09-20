from __future__ import annotations

import unittest

from laya_warehouse.controllers import HeuristicController, RandomController
from laya_warehouse.model import Action, Actor, World
from laya_warehouse.recording import run_episode, verify_record


class SimulationTests(unittest.TestCase):
    def test_safety_shield_blocks_an_actor_entering_the_target_cell(self) -> None:
        world = World(robot_x=5, robot_y=5, actors=[Actor("worker", "worker", 4, 4, 1)])

        result = world.step(Action.ADVANCE, safety_shield=True)

        self.assertTrue(result.safety_override)
        self.assertEqual(result.applied_action, Action.WAIT.value)
        self.assertEqual(world.status, "running")
        self.assertEqual((world.robot_x, world.robot_y), (5, 5))

    def test_disabling_safety_shield_allows_a_collision(self) -> None:
        world = World(robot_x=5, robot_y=5, actors=[Actor("worker", "worker", 4, 4, 1)])

        result = world.step(Action.ADVANCE, safety_shield=False)

        self.assertFalse(result.safety_override)
        self.assertEqual(world.status, "collision")

    def test_safety_shield_moves_when_waiting_would_cause_a_collision(self) -> None:
        world = World(robot_x=5, robot_y=5, actors=[Actor("worker", "worker", 6, 5, -1)])

        result = world.step(Action.WAIT, safety_shield=True)

        self.assertTrue(result.safety_override)
        self.assertEqual(result.applied_action, Action.SHIFT_LEFT.value)
        self.assertEqual(world.status, "running")
        self.assertEqual((world.robot_x, world.robot_y), (4, 5))

    def test_heuristic_controller_completes_the_default_scenario(self) -> None:
        record = run_episode(HeuristicController(), max_ticks=100)

        self.assertEqual(record["outcome"], "completed")
        self.assertLessEqual(record["ticks"], 100)
        verify_record(record)

    def test_seeded_random_episode_replays_deterministically(self) -> None:
        record = run_episode(RandomController(seed=7), max_ticks=25)

        verify_record(record)


if __name__ == "__main__":
    unittest.main()
