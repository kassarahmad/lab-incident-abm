import random
import unittest

import numpy as np

from agents import Occupant, Responder
from environment import FLOOR_PLAN, GUIDED_CORRIDOR_CLOSURE, Environment
from messaging import INFORM, BeliefMap, MessageBus
from model import COORDINATION_MODES, LabIncidentModel
from options import EXPLORE, FOLLOW_TASK, NEAREST, choose_option
from run_experiments import holm, rank_biserial


def run(**kw):
    m = LabIncidentModel(**kw)
    while m.running:
        m.step()
    return m


class EnvironmentTests(unittest.TestCase):
    def test_floor_plan_is_rectangular(self):
        self.assertTrue(all(len(r) == len(FLOOR_PLAN[0]) for r in FLOOR_PLAN))

    def test_walls_never_burn_or_hold_smoke(self):
        env = Environment()
        env.ignite(env.cabinets[0])
        rng = np.random.default_rng(0)
        for _ in range(150):
            env.update(rng)
        walls = env.tiles == "#"
        self.assertFalse(env.fire[walls].any())
        self.assertFalse(env.smoke[walls].any())

    def test_fire_needs_fuel(self):
        env = Environment()
        env.fuel[:] = 0
        env.ignite((5, 5), 0.5)
        rng = np.random.default_rng(0)
        for _ in range(30):
            env.update(rng)
        self.assertEqual(int((env.fire > 0).sum()), 0)


class MessagingTests(unittest.TestCase):
    def test_latency_and_total_loss(self):
        bus = MessageBus(random.Random(0), latency=2, loss=0.0)
        bus.send(INFORM, 1, 2, {}, now=0)
        bus.deliver(1)
        self.assertEqual(bus.receive(2), [])
        bus.deliver(2)
        self.assertEqual(len(bus.receive(2)), 1)
        lossy = MessageBus(random.Random(0), latency=0, loss=1.0)
        lossy.send(INFORM, 1, 2, {}, now=0)
        lossy.deliver(0)
        self.assertEqual(lossy.receive(2), [])
        self.assertEqual(lossy.dropped[INFORM], 1)

    def test_newer_observation_wins(self):
        b = BeliefMap()
        b.merge([((1, 1), 5, True)])
        b.merge([((1, 1), 3, False)])     # older: ignored
        self.assertEqual(b.known_fires(6), [(1, 1)])
        b.merge([((1, 1), 7, False)])
        self.assertEqual(b.known_fires(8), [])


class ModelTests(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        for mode in COORDINATION_MODES:
            a = run(seed=7, coordination=mode).datacollector.get_model_vars_dataframe()
            b = run(seed=7, coordination=mode).datacollector.get_model_vars_dataframe()
            self.assertTrue(a.equals(b), mode)

    def test_occupants_are_conserved_and_cells_exclusive(self):
        m = LabIncidentModel(seed=3, alarm_delay=10, n_ignitions=2)
        for _ in range(120):
            m.step()
            people = [a for a in m.agents if isinstance(a, (Occupant, Responder))]
            inside = sum(isinstance(a, Occupant) for a in people)
            self.assertEqual(inside + m.evacuated, m.n_occupants)
            pos = [a.pos for a in people]
            self.assertEqual(len(pos), len(set(pos)))
            self.assertTrue(all(m.env.walkable[p] for p in pos))

    def test_responders_reduce_damage(self):
        seeds = range(6)
        team = sum(run(seed=s).summary()["burned_area"] for s in seeds)
        alone = sum(run(seed=s, n_responders=0).summary()["burned_area"] for s in seeds)
        self.assertLess(team, alone)

    def test_dispatcher_only_knows_reported_positions(self):
        m = LabIncidentModel(seed=11, coordination="dispatcher", n_ignitions=2)
        for _ in range(60):
            m.step()
            self.assertTrue(set(m.dispatcher.status) <= set(m.responder_ids))
        self.assertIsNone(LabIncidentModel(seed=11, coordination="none").dispatcher)

    def test_invalid_coordination(self):
        with self.assertRaises(ValueError):
            LabIncidentModel(coordination="chaos")

    def test_robot_guides_can_use_both_exits_after_closure(self):
        m = LabIncidentModel(seed=1006, robot_guidance="local")
        self.assertEqual(m.closed_cells, ())
        while m.running:
            m.step()
        self.assertEqual(m.closed_cells, GUIDED_CORRIDOR_CLOSURE)
        self.assertEqual(m.alarm_step, m.ignition_step)
        self.assertTrue(all(not m.env.walkable[c] for c in m.closed_cells))
        self.assertEqual(len(m.env.exits), 2)
        for field in m.guided_exit_fields.values():
            self.assertTrue(all(np.isfinite(field[r.location]) for r in m.robots))
        self.assertEqual(m.evacuated, m.n_occupants)
        self.assertIsNotNone(m.summary()["all_out_step"])
        self.assertEqual(sum(sum(r.assigned.values()) for r in m.robots), 0)

    def test_robot_uses_other_junctions_count_only_when_shared(self):
        for mode, expected in (("local", "east"), ("shared", "west")):
            with self.subTest(mode=mode):
                m = LabIncidentModel(seed=1000, n_responders=0, n_occupants=1, robot_guidance=mode)
                for _ in range(m.ignition_step):
                    m.step()
                west, east = m.robots
                west.assigned["east"] = 20
                occupant = m.occupants()[0]
                m.grid.move_agent(occupant, east.location)
                occupant.state = "evacuating"
                east.direct([occupant])
                self.assertEqual(occupant.guided_exit, expected)
                self.assertEqual(east.assigned[expected], 1)

    def test_waiting_is_recorded_only_when_other_people_block_progress(self):
        m = LabIncidentModel(seed=1000, n_responders=0, n_occupants=2)
        waiting, ahead = m.occupants()
        m.grid.move_agent(waiting, (2, 9))
        m.grid.move_agent(ahead, (1, 9))
        field = np.full_like(m.exit_field, np.inf)
        field[waiting.pos] = 2
        self.assertFalse(m.move_down_field(waiting, field))
        self.assertEqual(m.queueing, 0)
        field[ahead.pos] = 1
        self.assertFalse(m.move_down_field(waiting, field))
        self.assertEqual(m.queueing, 1)
        self.assertEqual(m.queue_wait[waiting.pos], 1)


class DecisionAndStatsTests(unittest.TestCase):
    def test_decision_rule(self):
        self.assertEqual(choose_option([FOLLOW_TASK, NEAREST, EXPLORE]), FOLLOW_TASK)
        self.assertEqual(choose_option([NEAREST, EXPLORE]), NEAREST)
        self.assertEqual(choose_option([EXPLORE]), EXPLORE)

    def test_holm_and_effect_size(self):
        np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
        self.assertEqual(rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
