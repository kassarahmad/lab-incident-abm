import random
import unittest

import numpy as np

from agents import Occupant, Responder
from environment import FLOOR_PLAN, Environment, fire_clusters
from messaging import INFORM, BeliefMap, MessageBus
from model import COORDINATION_MODES, LabIncidentModel
from policies import QLearningPolicy, RulePolicy
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

    def test_clusters(self):
        self.assertEqual(len(fire_clusters([(1, 1), (2, 2), (10, 10)])), 2)


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

    def test_contract_net_one_task_per_responder(self):
        m = LabIncidentModel(seed=11, coordination="contract_net", n_ignitions=2)
        for _ in range(80):
            m.step()
            self.assertLessEqual(len(m.coordinator.contracts), len(m.responder_list))
            self.assertTrue(set(m.coordinator.contracts) <= set(m.responder_ids))

    def test_invalid_coordination(self):
        with self.assertRaises(ValueError):
            LabIncidentModel(coordination="chaos")


class LearningAndStatsTests(unittest.TestCase):
    def test_q_update(self):
        p = QLearningPolicy(alpha=0.5, gamma=0.9)
        s, s2 = (1, 1, 1, 0, 0), (1, 1, 0, 0, 0)
        p.q[s2][2] = 10
        p.update(s, 1, reward=-2.0, tau=2, s_next=s2, valid_next=[1, 2], terminal=False)
        # first visit: step size max(alpha, 1/1^0.6) = 1
        self.assertAlmostEqual(p.q[s][1], -2.0 + 0.81 * 10)
        p.update(s, 1, reward=-2.0, tau=2, s_next=s2, valid_next=[1, 2], terminal=False)
        self.assertAlmostEqual(p.q[s][1], 6.1)  # same target, stays put

    def test_greedy_policy_does_not_learn(self):
        g = QLearningPolicy().greedy()
        g.update((0, 0, 0, 0, 0), 0, 5.0, 1, (0, 0, 0, 0, 0), [0], True)
        self.assertEqual(g.q.sum(), 0)

    def test_rule_policy_prefers_task(self):
        self.assertEqual(RulePolicy().choose((3, 1, 1, 1, 0), [0, 1, 2, 3], random.Random(0)), 0)

    def test_holm_and_effect_size(self):
        np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
        self.assertEqual(rank_biserial(np.array([1.0, 2.0, 3.0])), 1.0)


if __name__ == "__main__":
    unittest.main()
