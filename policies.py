"""Decision policies of the responders.

Responders have two reflexes that are never learned (shoot at fire within
reach, go refill when the extinguisher is empty). Everything else is a choice
between *options* (temporally extended actions, Sutton et al. 1999):

    0 FOLLOW_TASK   go to the task given by the coordinator (Contract Net / dispatch)
    1 NEAREST       go to the nearest fire in my belief map
    2 FRONT         go to the believed fire cell with the most unburned fuel around it
    3 EXPLORE       climb the smoke gradient to find fire I do not know about
    4 REFILL        top up the extinguisher before it is empty

RulePolicy is the hand-written baseline. QLearningPolicy learns the choice
with SMDP Q-learning; all responders share one Q-table (parameter sharing,
independent learners) and the same team reward.
"""

import json
from pathlib import Path

import numpy as np

FOLLOW_TASK, NEAREST, FRONT, EXPLORE, REFILL = range(5)
OPTION_NAMES = ["follow_task", "nearest", "front", "explore", "refill"]
N_OPTIONS = len(OPTION_NAMES)
STATE_SHAPE = (4, 4, 3, 2, 3)   # charges, known fire size, distance, has task, teammates near


def encode_state(charges, max_charges, n_known, dist, has_task, mates_near):
    c = 0 if charges == 0 else 1 if charges <= 0.3 * max_charges else 2 if charges <= 0.7 * max_charges else 3
    k = 0 if n_known == 0 else 1 if n_known <= 5 else 2 if n_known <= 20 else 3
    d = 0 if dist <= 3 else 1 if dist <= 10 else 2
    m = min(mates_near, 2)
    return (c, k, d, int(has_task), m)


class RulePolicy:
    name = "rule"
    learning = False

    def choose(self, state, valid, rng):
        charges, known, _, has_task, _ = state
        if has_task and FOLLOW_TASK in valid:
            return FOLLOW_TASK
        if known and NEAREST in valid:
            return NEAREST
        return EXPLORE

    def update(self, *args):
        pass


class QLearningPolicy:
    """Tabular SMDP Q-learning over options.

    Q(s,o) <- Q(s,o) + alpha * [R + gamma^tau * max_o' Q(s',o') - Q(s,o)]
    where R is the discounted team reward collected during the tau steps the
    option ran.
    """
    name = "learned"

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1, q=None, learning=True):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.learning = learning
        self.q = np.zeros(STATE_SHAPE + (N_OPTIONS,)) if q is None else q
        self.visits = np.zeros(STATE_SHAPE + (N_OPTIONS,), dtype=int)

    def choose(self, state, valid, rng):
        valid = list(valid)
        if self.learning and rng.random() < self.epsilon:
            return rng.choice(valid)
        values = self.q[state][valid]
        best = np.flatnonzero(values == values.max())
        return valid[int(best[rng.randrange(len(best))])]

    def update(self, s, o, reward, tau, s_next, valid_next, terminal):
        if not self.learning:
            return
        target = reward
        if not terminal:
            target += self.gamma ** tau * max(self.q[s_next][v] for v in valid_next)
        self.visits[s][o] += 1
        # visit-count step size (Robbins-Monro), never below alpha
        lr = max(self.alpha, 1.0 / self.visits[s][o] ** 0.6)
        self.q[s][o] += lr * (target - self.q[s][o])

    def greedy(self):
        g = QLearningPolicy(q=self.q.copy(), learning=False, epsilon=0.0)
        g.visits = self.visits.copy()
        return g

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "state_shape": STATE_SHAPE, "options": OPTION_NAMES,
            "q": self.q.round(5).tolist(), "visits": self.visits.tolist(),
        }))

    @classmethod
    def load(cls, path, learning=False):
        data = json.loads(Path(path).read_text())
        pol = cls(q=np.array(data["q"]), learning=learning, epsilon=0.0)
        pol.visits = np.array(data["visits"])
        return pol
