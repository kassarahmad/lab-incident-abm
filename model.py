"""
Lab Incident Response - Multi-Agent Model (Mesa 3.x)

A fire starts at a chemical cabinet on a university lab floor. Detectors feed
an alarm panel and (optionally) a dispatcher; responders with limited
extinguishers fight the fire using beliefs built from their own vision and
from messages sent over an unreliable radio; occupants evacuate.

Coordination modes
    none        no dispatcher: responders use only their own eyes
    dispatcher  a control room sends each free responder to the nearest known fire
"""

from typing import Optional

import mesa
import numpy as np

from agents import Dispatcher, EvacuationRobot, Occupant, Responder, SmokeDetector
from environment import (
    BYPASS_CELLS, CORRIDOR_ROWS, EAST_CORRIDOR_CLOSURE, FLOOR_PLAN,
    GUIDED_CORRIDOR_CLOSURE, GUIDE_JUNCTIONS, MOORE, Environment, guided_floor_plan,
)
from messaging import MessageBus

COORDINATION_MODES = ("none", "dispatcher")


class LabIncidentModel(mesa.Model):

    def __init__(
        self,
        n_responders: int = 4,
        n_occupants: int = 40,
        coordination: str = "dispatcher",
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
        ignition_step: int = 3,
        max_steps: int = 300,
        seed: Optional[int] = None,
        block_corridor: bool = False,
        robot_guidance: Optional[str] = None,
    ):
        super().__init__(rng=seed)
        if coordination not in COORDINATION_MODES:
            raise ValueError(f"coordination must be one of {COORDINATION_MODES}")
        if robot_guidance not in (None, "local", "shared"):
            raise ValueError("robot_guidance must be local, shared, or None")

        self.coordination = coordination
        self.alarm_delay = alarm_delay
        self.detector_reliability = detector_reliability
        self.extinguisher_charges = extinguisher_charges
        self.refill_time = refill_time
        self.familiarity = familiarity
        self.ignition_step = ignition_step
        self.max_steps = max_steps
        self.robot_guidance = robot_guidance
        self.corridor_to_close = EAST_CORRIDOR_CLOSURE if block_corridor else ()
        if robot_guidance:
            self.corridor_to_close = GUIDED_CORRIDOR_CLOSURE
        # fixed behavioural constants (see ODD.md)
        self.smoke_threshold = 0.04
        self.extinguish_power = 0.5
        self.smoke_damage = 0.05
        self.fire_damage = 0.25
        self.vision = 5
        self.status_period = 3              # radio check-in of responders
        self.commit = 4                     # max steps an option is kept
        self.premovement_mu, self.premovement_sigma = 1.2, 0.6  # median ~3.3 steps

        self.env = Environment(
            p_spread=p_spread, floor_plan=guided_floor_plan() if robot_guidance else FLOOR_PLAN,
            corridor_extension=BYPASS_CELLS if robot_guidance else (),
        )
        self.bus = MessageBus(self.random, latency=comm_latency, loss=comm_loss)
        self.grid = mesa.space.SingleGrid(self.env.width, self.env.height, torus=False)
        env = self.env

        # incident state
        self.alarm_on = False
        self.alarm_step = None
        self.first_detection = None
        self.contained_step = None
        self.evacuation_done_step = None
        self.closed_cells = ()
        self.queue_wait = np.zeros((env.width, env.height), dtype=int)
        self.queueing = 0
        self.peak_queueing = 0
        self.ignition_cells = self.random.sample(env.cabinets, n_ignitions)
        self.ignition_cell = self.ignition_cells[0]
        self.evacuated = 0
        self.incapacitated = 0
        self.injured_evacuated = 0
        self.total_shots = 0
        self.first_shot_step = None
        self.total_dose = 0.0

        # fixed agents
        self.dispatcher = Dispatcher(self) if coordination == "dispatcher" else None
        self.detectors = []
        for x in range(2, env.width - 1, detector_spacing):
            for y in range(2, env.height - 1, 3):
                if env.air[x, y]:
                    self.detectors.append(SmokeDetector(self, (x, y)))
        self.robots = [EvacuationRobot(self, cells) for cells in GUIDE_JUNCTIONS] if robot_guidance else []

        # responders start at the base around the station
        sx, sy = env.stations[0]
        base = sorted((tuple(map(int, c)) for c in np.argwhere(env.walkable) if env.tiles[tuple(c)] != "E"),
                      key=lambda c: (abs(c[0] - sx) + abs(c[1] - sy), c))
        self.responder_list = []
        for c in base[:n_responders]:
            r = Responder(self)
            self.grid.place_agent(r, c)
            self.responder_list.append(r)
        self.responder_ids = [r.unique_id for r in self.responder_list]
        if self.dispatcher is not None:   # the roster: starting positions are known
            self.dispatcher.status = {r.unique_id: (r.pos, r.charges, False, 0) for r in self.responder_list}

        # occupants in the labs
        floor = (tuple(map(int, c)) for c in np.argwhere(env.tiles == "."))
        lab_cells = [c for c in floor if c[1] not in CORRIDOR_ROWS
                     and c not in env.corridor_extension and self.grid.is_cell_empty(c)]
        self.random.shuffle(lab_cells)
        for c in lab_cells[:n_occupants]:
            self.grid.place_agent(Occupant(self), c)
        self.n_occupants = min(n_occupants, len(lab_cells))

        self.station_field = env.distance_field(env.stations)
        self.west_exits = [e for e in env.exits if e[0] == 0]
        self.exits_by_side = {"west": self.west_exits, "east": [e for e in env.exits if e[0] == env.width - 1]}
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
        if self.robots:
            self.guided_exit_fields = {
                "west": self.west_exit_field,
                "east": self.env.distance_field(self.exits_by_side["east"], avoid_fire=True),
            }
            for robot in self.robots:
                available = [c for c in robot.junction if self.env.fire[c] <= 0.3]
                robot.field = self.env.distance_field(available, avoid_fire=True)

    def near_station(self, pos):
        return any(max(abs(pos[0] - x), abs(pos[1] - y)) <= 1 for x, y in self.env.stations)

    def walkable_neighbours(self, pos):
        x, y = pos
        env = self.env
        return [(x + dx, y + dy) for dx, dy in MOORE
                if env.in_bounds(x + dx, y + dy) and env.walkable[x + dx, y + dy]]

    def free_neighbours(self, pos):
        return [c for c in self.walkable_neighbours(pos) if self.grid.is_cell_empty(c)]

    def move_random(self, agent):
        options = self.free_neighbours(agent.pos)
        if options:
            self.grid.move_agent(agent, self.random.choice(options))

    def move_down_field(self, agent, field):
        """Step to the free neighbour that most reduces the distance. False if
        the agent cannot get closer (arrived, blocked, no path). An occupant
        queues when all closer walkable cells are occupied."""
        here = field[agent.pos]
        closer = [c for c in self.walkable_neighbours(agent.pos) if field[c] < here]
        options = [c for c in closer if self.grid.is_cell_empty(c)]
        if options:
            best = min(field[c] for c in options)
            self.grid.move_agent(agent, self.random.choice([c for c in options if field[c] == best]))
            return True
        if self._try_swap(agent, closer):
            return True
        if closer and isinstance(agent, Occupant):
            self.queue_wait[agent.pos] += 1
            self.queueing += 1
        return False

    def _try_swap(self, agent, occupied_neighbours):
        """Pass a responder, or let guided evacuees pass in opposite directions."""
        for c in occupied_neighbours:
            other = self.grid.get_cell_list_contents([c])[0]
            if (isinstance(other, Responder) or isinstance(agent, Responder)) \
                    and getattr(other, "state", "") != "incapacitated" and self.random.random() < 0.5:
                self.grid.swap_pos(agent, other)
                return True
            if self.robots and isinstance(agent, Occupant) and isinstance(other, Occupant) \
                    and other.state == "evacuating":
                return_field = other.evacuation_field()
                if return_field[agent.pos] < return_field[c]:
                    self.grid.swap_pos(agent, other)
                    return True
        return False

    def evacuate(self, agent):
        self.evacuated += 1
        self.injured_evacuated += agent.injured
        agent.finish_guidance()
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

        # 1. environment (synchronous)
        if self.steps == self.ignition_step:
            for c in self.ignition_cells:
                env.ignite(c)
            if self.corridor_to_close:
                self.closed_cells = self.corridor_to_close
                env.close_passage(self.closed_cells)
                self.station_field = env.distance_field(env.stations)
                if self.robots:
                    self.panel_detection(self.closed_cells[0])
        env.update(self.rng)
        self.update_exit_fields()
        self.bus.deliver(self.steps)

        # 2. sensing, alarm panel, dispatcher
        for d in self.detectors:
            d.step()
        if self.first_detection is not None and not self.alarm_on \
                and self.steps - self.first_detection >= self.alarm_delay:
            self.alarm_on, self.alarm_step = True, self.steps
        if self.dispatcher is not None:
            self.dispatcher.step()
        if self.robots:
            occupants = self.occupants()
            for robot in self.robots:
                robot.direct(occupants)

        # 3. people act in random order
        self.queueing = 0
        self.agents.select(lambda a: isinstance(a, (Responder, Occupant))).shuffle_do("step")

        # 4. record and stop
        self.peak_queueing = max(self.peak_queueing, self.queueing)
        burning = int((env.fire > 0).sum())
        self.datacollector.collect(self)
        fire_out = self.steps > self.ignition_step and burning == 0
        if fire_out and self.contained_step is None:
            self.contained_step = self.steps
        if self.evacuation_done_step is None and self.n_occupants - self.evacuated - self.incapacitated == 0:
            self.evacuation_done_step = self.steps
        if (fire_out and self.evacuation_done_step is not None) or self.steps >= self.max_steps:
            self.running = False

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
            "all_out_step": self.evacuation_done_step if self.evacuated == self.n_occupants else None,
            "queueing": self.queueing,
            "peak_queueing": self.peak_queueing,
            "exposure": round(self.total_dose, 4),
            "injured": self.injured_count(),
            "incapacitated": self.incapacitated,
            "shots": self.total_shots,
            "messages": self.bus.total_sent,
            "messages_dropped": sum(self.bus.dropped.values()),
        }
