from __future__ import annotations

import unittest

from laya_warehouse.controllers import HeuristicController, RandomController
from laya_warehouse.controllers import LayaController
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

    def test_laya_controller_uses_the_typed_action_choice(self) -> None:
        class FakeAgent:
            def predict(self, observation, questions):
                self.observation = observation
                self.questions = questions
                return {
                    "answers": {
                        "action": {
                            "choice": "shift_right",
                            "probabilities": {
                                "advance": 0.2,
                                "shift_left": 0.1,
                                "shift_right": 0.6,
                                "wait": 0.1,
                            },
                        },
                        "collision_risk": {"score": 1.0},
                        "path_blocked": {"noul": 0.7},
                        "needs_operator": {"noul": 0.2},
                    },
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                }

        controller = LayaController.__new__(LayaController)
        controller._agent = FakeAgent()

        decision = controller.decide(World.default().observe())

        self.assertEqual(decision.action, Action.SHIFT_RIGHT)
        self.assertEqual(decision.details["action_scores"]["shift_right"], 0.6)


if __name__ == "__main__":
    unittest.main()
