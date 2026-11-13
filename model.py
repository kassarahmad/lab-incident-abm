"""
Lab Incident Response - Adaptive Multi-Agent Model (Mesa 3.x)

A fire starts at a chemical cabinet on a university lab floor. Detectors feed
an alarm panel and a coordinator; responders with limited extinguishers fight
the fire using beliefs built from their own vision and from messages sent over
an unreliable channel; heterogeneous occupants evacuate.

Coordination modes
    none            responders use only their own perception
    peer            responders share observations with each other (no coordinator)
    central_greedy  a coordinator dispatches the nearest free responder
    contract_net    a coordinator auctions fire clusters with the Contract Net Protocol

Policies
    rule            hand-written option choice
    learned         SMDP Q-learning over options (train_qlearning.py)
"""

from collections import Counter
from pathlib import Path
from typing import Optional

import mesa
import numpy as np

from agents import Coordinator, Occupant, Responder, SmokeDetector
from environment import CORRIDOR_ROWS, MOORE, Environment
from messaging import MessageBus
from policies import QLearningPolicy, RulePolicy

COORDINATION_MODES = ("none", "peer", "central_greedy", "contract_net")
DEFAULT_Q_TABLE = Path(__file__).resolve().parent / "policies" / "q_table.json"


class LabIncidentModel(mesa.Model):

    def __init__(
        self,
        n_responders: int = 4,
        n_occupants: int = 40,
        coordination: str = "contract_net",
        policy="rule",                      # "rule", "learned" or a policy object
        alarm_delay: int = 0,
        n_ignitions: int = 1,
        comm_latency: int = 1,
        comm_loss: float = 0.0,
        detector_reliability: float = 0.9,
        detector_spacing: int = 6,
        extinguisher_charges: int = 10,
        refill_time: int = 3,
        p_spread: float = 0.25,
        familiarity: float = 0.6,
        full_deployment: bool = True,
        ignition_step: int = 3,
        max_steps: int = 300,
        seed: Optional[int] = None,
    ):
        super().__init__(rng=seed)
        if coordination not in COORDINATION_MODES:
            raise ValueError(f"coordination must be one of {COORDINATION_MODES}")

        self.coordination = coordination
        self.alarm_delay = alarm_delay
        self.detector_reliability = detector_reliability
        self.extinguisher_charges = extinguisher_charges
        self.refill_time = refill_time
        self.familiarity = familiarity
        self.full_deployment = full_deployment
        self.ignition_step = ignition_step
        self.max_steps = max_steps
        # fixed behavioural constants (see ODD.md)
        self.smoke_threshold = 0.04
        self.extinguish_power = 0.5
        self.shot_reward = 2.0               # learning: individual reward per unit of fire removed
        self.smoke_damage = 0.05
        self.fire_damage = 0.25
        self.vision = 5
        self.status_period = 3              # radio check-in of responders (central modes)
        self.contract_timeout = 12          # Contract Net: drop silent contractors
        self.commit = 4                      # max steps an option is kept
        self.premovement_mu, self.premovement_sigma = 1.2, 0.6  # median ~3.3 steps

        if policy == "rule":
            policy = RulePolicy()
        elif policy == "learned":
            policy = QLearningPolicy.load(DEFAULT_Q_TABLE)
        self.policy = policy

        self.env = Environment(p_spread=p_spread)
        self.bus = MessageBus(self.random, latency=comm_latency, loss=comm_loss)
        self.grid = mesa.space.SingleGrid(self.env.width, self.env.height, torus=False)
        env = self.env

        # incident state
        self.alarm_on = False
        self.alarm_step = None
        self.first_detection = None
        self.contained_step = None
        self.evacuation_done_step = None
        self.ignition_cells = self.random.sample(env.cabinets, n_ignitions)
        self.ignition_cell = self.ignition_cells[0]
        self.evacuated = 0
        self.incapacitated = 0
        self.injured_evacuated = 0
        self.total_shots = 0
        self.first_shot_step = None
        self.total_dose = 0.0
        self.option_counts = Counter()
        self.reward_total = 0.0

        # fixed agents
        self.coordinator = Coordinator(self) if coordination in ("central_greedy", "contract_net") else None
        self.detectors = []
        for x in range(2, env.width - 1, detector_spacing):
            for y in range(2, env.height - 1, 3):
                if env.air[x, y]:
                    self.detectors.append(SmokeDetector(self, (x, y)))

        # responders start at the base around the station
        sx, sy = env.stations[0]
        base = sorted((tuple(map(int, c)) for c in np.argwhere(env.walkable) if env.tiles[tuple(c)] != "E"),
                      key=lambda c: (abs(c[0] - sx) + abs(c[1] - sy), c))
        self.responder_list = []
        for c in base[:n_responders]:
            r = Responder(self, self.policy)
            self.grid.place_agent(r, c)
            self.responder_list.append(r)
        self.responder_ids = [r.unique_id for r in self.responder_list]
        if self.coordinator is not None:   # the roster: starting positions are known
            self.coordinator.status = {r.unique_id: (r.pos, r.charges, False, 0) for r in self.responder_list}

        # occupants in the labs
        lab_cells = [tuple(map(int, c)) for c in np.argwhere(env.tiles == ".")
                     if c[1] not in CORRIDOR_ROWS and self.grid.is_cell_empty(tuple(map(int, c)))]
        self.random.shuffle(lab_cells)
        for c in lab_cells[:n_occupants]:
            self.grid.place_agent(Occupant(self), c)
        self.n_occupants = min(n_occupants, len(lab_cells))

        self.station_field = env.distance_field(env.stations)
        self.west_exits = [e for e in env.exits if e[0] == 0]
        self.update_exit_fields()

        self.datacollector = mesa.DataCollector(model_reporters={
            "Burning": lambda m: int((m.env.fire > 0).sum()),
            "BurnedArea": lambda m: int(m.env.ever_burned.sum()),
            "MeanSmoke": lambda m: round(float(m.env.smoke[m.env.air].mean()), 4),
            "Inside": lambda m: m.n_occupants - m.evacuated - m.incapacitated,
            "Evacuated": lambda m: m.evacuated,
            "Incapacitated": lambda m: m.incapacitated,
            "Exposure": lambda m: round(m.total_dose, 4),
            "Alarm": lambda m: int(m.alarm_on),
            "Shots": lambda m: m.total_shots,
            "Messages": lambda m: m.bus.total_sent,
        })
        self.datacollector.collect(self)

    # ------------------------------------------------------------- helpers
    def update_exit_fields(self):
        self.exit_field = self.env.distance_field(self.env.exits, avoid_fire=True)
        self.west_exit_field = self.env.distance_field(self.west_exits, avoid_fire=True)
        if np.isinf(self.exit_field).all():
            self.exit_field = self.env.distance_field(self.env.exits)

    def near_station(self, pos):
        return any(max(abs(pos[0] - x), abs(pos[1] - y)) <= 1 for x, y in self.env.stations)

    def free_neighbours(self, pos):
        x, y = pos
        env = self.env
        return [(x + dx, y + dy) for dx, dy in MOORE
                if env.in_bounds(x + dx, y + dy) and env.walkable[x + dx, y + dy]
                and self.grid.is_cell_empty((x + dx, y + dy))]

    def move_random(self, agent):
        options = self.free_neighbours(agent.pos)
        if options:
            self.grid.move_agent(agent, self.random.choice(options))

    def move_down_field(self, agent, field):
        """Step to the free neighbour that most reduces the distance. False if
        the agent cannot get closer (arrived, blocked, no path)."""
        here = field[agent.pos]
        options = [c for c in self.free_neighbours(agent.pos) if field[c] < here]
        if not options:
            return self.try_swap(agent, field)
        best = min(field[c] for c in options)
        self.grid.move_agent(agent, self.random.choice([c for c in options if field[c] == best]))
        return True

    def try_swap(self, agent, field):
        """Counter-flow: anyone can squeeze past a responder (and a responder past
        anyone); occupants queue behind other occupants."""
        x, y = agent.pos
        here = field[x, y]
        for dx, dy in MOORE:
            c = (x + dx, y + dy)
            if not self.env.in_bounds(*c) or field[c] >= here or self.grid.is_cell_empty(c):
                continue
            other = self.grid.get_cell_list_contents([c])[0]
            if (isinstance(other, Responder) or isinstance(agent, Responder)) \
                    and getattr(other, "state", "") != "incapacitated" and self.random.random() < 0.5:
                self.grid.swap_pos(agent, other)
                return True
        return False

    def evacuate(self, agent):
        self.evacuated += 1
        self.injured_evacuated += agent.injured
        self.grid.remove_agent(agent)
        agent.remove()

    def panel_detection(self, location):
        if self.first_detection is None:
            self.first_detection = self.steps

    def occupants(self):
        return [a for a in self.agents if isinstance(a, Occupant)]

    def injured_count(self):
        return self.injured_evacuated + sum(1 for a in self.occupants() if a.injured)

    # ------------------------------------------------------------- step
    def step(self):
        env = self.env
        burning_before = int((env.fire > 0).sum())
        burned_before = int(env.ever_burned.sum())

        # 1. environment (synchronous)
        if self.steps == self.ignition_step:
            for c in self.ignition_cells:
                env.ignite(c)
        env.update(self.rng)
        self.update_exit_fields()
        self.bus.deliver(self.steps)

        # 2. sensing, alarm panel, coordination
        for d in self.detectors:
            d.step()
        if self.first_detection is not None and not self.alarm_on \
                and self.steps - self.first_detection >= self.alarm_delay:
            self.alarm_on, self.alarm_step = True, self.steps
        if self.coordinator is not None:
            self.coordinator.step()

        # 3. people act in random order
        self.agents.select(lambda a: isinstance(a, (Responder, Occupant))).shuffle_do("step")

        # 4. team reward for learning responders: new fire is bad, burning fire is bad
        new_burned = int(env.ever_burned.sum()) - burned_before
        burning = int((env.fire > 0).sum())
        reward = -1.0 * new_burned - 0.05 * burning
        self.reward_total += reward
        for r in self.responder_list:
            r.add_reward(reward)

        # 5. record and stop
        self.datacollector.collect(self)
        fire_out = self.steps > self.ignition_step and burning == 0
        if fire_out and self.contained_step is None:
            self.contained_step = self.steps
        if self.evacuation_done_step is None and self.n_occupants - self.evacuated - self.incapacitated == 0:
            self.evacuation_done_step = self.steps
        if (fire_out and self.evacuation_done_step is not None) or self.steps >= self.max_steps:
            self.running = False
            for r in self.responder_list:
                r.end_option(terminal=fire_out)

    def summary(self):
        return {
            "steps": self.steps,
            "alarm_step": self.alarm_step,
            "contained": self.contained_step is not None,
            "response_time": (self.first_shot_step - self.alarm_step)
            if self.first_shot_step is not None and self.alarm_step is not None else None,
            "contained_step": self.contained_step,
            "burned_area": int(self.env.ever_burned.sum()),
            "evacuated": self.evacuated,
            "evacuation_time": self.evacuation_done_step,
            "exposure": round(self.total_dose, 4),
            "injured": self.injured_count(),
            "incapacitated": self.incapacitated,
            "shots": self.total_shots,
            "messages": self.bus.total_sent,
            "messages_dropped": sum(self.bus.dropped.values()),
            "team_reward": round(self.reward_total, 2),
        }
