"""Agents of the lab incident model.

SmokeDetector  fixed sensor, feeds the alarm panel and reports to the dispatcher
Dispatcher     control room: sends each free responder to the nearest known fire
Responder      safety officer with an extinguisher; acts on its own beliefs
EvacuationRobot  fixed guide at a junction; directs waiting groups to an exit
Occupant       evacuee with a reaction delay, herding and exit familiarity
"""

from collections import Counter

import mesa
import numpy as np

from environment import MOORE
from messaging import DONE, FAILURE, INFORM, REQUEST, BeliefMap
from options import EXPLORE, FOLLOW_TASK, NEAREST, choose_option


# ==========================================================================
class SmokeDetector(mesa.Agent):
    """Ceiling detector. Senses smoke above a threshold or flames within 3 cells
    in the same room. A detection can be missed (reliability < 1)."""

    def __init__(self, model, location):
        super().__init__(model)
        self.location = location
        self.room = model.env.room_of(location)
        self.triggered = False

    def step(self):
        m, env = self.model, self.model.env
        x, y = self.location
        flames = [c for c in env.fire_cells_within(self.location, 3) if env.room_of(c) == self.room]
        sensing = env.smoke[x, y] > m.smoke_threshold or bool(flames)
        if sensing and self.random.random() < m.detector_reliability:
            self.triggered = True
            m.panel_detection(self.location)
            if m.dispatcher is not None:
                m.bus.send(INFORM, self.unique_id, m.dispatcher.unique_id,
                           {"burning": [(c, m.steps, True) for c in flames],
                            "detector": self.location}, m.steps)
        elif not sensing:
            self.triggered = False


# ==========================================================================
class Dispatcher(mesa.Agent):
    """Control room (not on the grid). It knows only what detectors and
    responders tell it by message. Every step it sends each free responder to
    the nearest known fire, avoiding sending everybody to the same spot. If a
    detector alarms in a room where no fire is known, someone is sent to check."""

    def __init__(self, model):
        super().__init__(model)
        self.beliefs = BeliefMap()
        self.status = {}             # responder id -> (pos, charges, refilling, step) last reported
        self.last_assignment = {}
        self.zone_alerts = {}        # detector location -> last step it reported
        self.investigated = {}       # detector location -> step a responder checked it

    def step(self):
        m = self.model
        for msg in m.bus.receive(self.unique_id):
            if msg.performative == INFORM:
                self.beliefs.merge(msg.content["burning"])
                self.beliefs.merge(msg.content.get("clear", []))
                if "status" in msg.content:
                    self.status[msg.sender] = msg.content["status"]
                if "detector" in msg.content:
                    self.zone_alerts[msg.content["detector"]] = msg.sent
            elif msg.performative in (DONE, FAILURE):
                if msg.content and "investigated" in msg.content:
                    self.investigated[tuple(msg.content["investigated"])] = m.steps
                self.beliefs.merge(msg.content.get("clear", []) if msg.content else [])
        if not m.alarm_on:
            return
        known = self.beliefs.known_fires(m.steps)
        self.dispatch(known, self.suspect_zones(known))

    def suspect_zones(self, known):
        """Detectors that alarmed recently in a room where no fire is known and
        that nobody has checked: someone must go and look (one per room)."""
        m, env = self.model, self.model.env
        per_room = {}
        for z, t in self.zone_alerts.items():
            if m.steps - t > 10 or m.steps - self.investigated.get(z, -999) <= 30:
                continue
            room = env.room_of(z)
            if any(env.room_of(c) == room for c in known):
                continue
            if room not in per_room or t > per_room[room][1]:
                per_room[room] = (z, t)
        return sorted(z for z, _ in per_room.values())

    def dispatch(self, known, suspects):
        m = self.model
        targets = known + suspects
        if not targets:
            return
        assigned = []
        # uses the last *reported* position and charges, not the true ones
        for rid in sorted(self.status):
            (rx, ry), charges, refilling, _ = self.status[rid]
            if charges == 0 or refilling:
                continue

            def cost(c):
                crowd = sum(1 for a in assigned if abs(a[0] - c[0]) + abs(a[1] - c[1]) <= 3)
                return abs(c[0] - rx) + abs(c[1] - ry) + 8 * crowd

            target = min(targets, key=cost)
            assigned.append(target)
            if self.last_assignment.get(rid) != target:
                self.last_assignment[rid] = target
                m.bus.send(REQUEST, self.unique_id, rid,
                           {"target": target, "investigate": target in suspects}, m.steps)


# ==========================================================================
class Responder(mesa.Agent):
    """Lab safety officer with an extinguisher.

    Beliefs   : map of where it thinks the fire is (own vision + messages)
    Reflexes  : shoot fire within reach; refill when empty
    Otherwise : follow the dispatcher's task, else go to the nearest known
                fire, else search by following the smoke (options.py)
    """

    def __init__(self, model):
        super().__init__(model)
        self.charges = model.extinguisher_charges
        self.beliefs = BeliefMap()
        self.task = None            # target cell from the dispatcher
        self.investigate = False    # task is "go and check a detector zone"
        self.refilling = 0
        self.shots = 0
        self.option = None          # current intention
        self.option_steps = 0

    # ---------------------------------------------------------- perception
    def perceive(self):
        m, env = self.model, self.model.env
        room = env.room_of(self.pos)
        seen = [c for c in env.fire_cells_within(self.pos, m.vision)
                if env.room_of(c) == room or room == 4 or env.room_of(c) == 4]
        x, y = self.pos
        r = m.vision
        # burned-out cells in view are reported as clear, so stale beliefs get corrected
        x0, y0 = max(0, x - r), max(0, y - r)
        out = np.argwhere(env.ever_burned[x0:x + r + 1, y0:y + r + 1] & (env.fire[x0:x + r + 1, y0:y + r + 1] == 0))
        clear = [c for c in (tuple(map(int, v)) for v in out + [x0, y0])
                 if env.room_of(c) == room or room == 4]
        # avoid re-sending clear cells nobody believes burning
        clear = [c for c in clear if self.beliefs.cells.get(c, (0, True))[1]]
        self.beliefs.observe(seen, clear, m.steps)
        if m.dispatcher is not None and m.alarm_on:
            # observations when there are any, otherwise a periodic radio check-in;
            # the dispatcher only knows where responders are from these messages
            if seen or clear or m.steps % m.status_period == 0:
                report = {"burning": [(c, m.steps, True) for c in seen],
                          "clear": [(c, m.steps, False) for c in clear],
                          "status": (self.pos, self.charges, self.refilling > 0, m.steps)}
                m.bus.send(INFORM, self.unique_id, m.dispatcher.unique_id, report, m.steps)
        return seen

    def handle_messages(self):
        for msg in self.model.bus.receive(self.unique_id):
            if msg.performative == REQUEST and self.charges > 0:
                self.task = tuple(msg.content["target"])
                self.investigate = bool(msg.content.get("investigate"))
                if not self.investigate:
                    self.beliefs.merge([(self.task, msg.sent, True)])

    def valid_options(self):
        opts = [EXPLORE]
        if self.task is not None:
            opts.append(FOLLOW_TASK)
        if self.beliefs.known_fires(self.model.steps):
            opts.append(NEAREST)
        return sorted(opts)

    # ---------------------------------------------------------- main loop
    def step(self):
        if self.pos is None:
            return
        m, env = self.model, self.model.env
        self.handle_messages()
        self.perceive()

        # Reflex 1: refilling / empty extinguisher
        if self.refilling > 0:
            self.refilling -= 1
            if self.refilling == 0:
                self.charges = m.extinguisher_charges
            return
        if self.charges == 0:
            self.drop_task(failure=True)
            self.option = None
            self.go_refill()
            return

        # Reflex 2: shoot the strongest fire within reach
        best, best_val = None, 0.0
        x, y = self.pos
        for dx, dy in ((0, 0),) + MOORE:
            c = (x + dx, y + dy)
            if env.in_bounds(*c) and env.fire[c] > best_val:
                best, best_val = c, env.fire[c]
        if best is not None:
            env.fire[best] = max(0.0, env.fire[best] - m.extinguish_power)
            if env.fire[best] < 0.05:
                env.fire[best] = 0.0
                self.beliefs.observe([], [best], m.steps)
            self.charges -= 1
            self.shots += 1
            m.total_shots += 1
            if m.first_shot_step is None:
                m.first_shot_step = m.steps
            return

        # Task finished? (target believed out and no fire around it)
        if self.task is not None:
            if self.investigate:
                # zone checked once I am there (anything burning is now in my beliefs)
                if max(abs(self.task[0] - x), abs(self.task[1] - y)) <= 1:
                    self.drop_task(failure=False)
            elif not any(abs(c[0] - self.task[0]) + abs(c[1] - self.task[1]) <= 4
                         for c in self.beliefs.known_fires(m.steps)):
                self.drop_task(failure=False)

        if not m.alarm_on and not self.beliefs.known_fires(m.steps):
            return  # idle at base

        # Keep the current option for a few steps, or choose a new one
        if self.option is None or self.option_steps >= m.commit or not self.option_valid():
            self.option = choose_option(self.valid_options())
            self.option_steps = 0
        self.execute_option()
        self.option_steps += 1

    def option_valid(self):
        if self.option == FOLLOW_TASK:
            return self.task is not None
        if self.option == NEAREST:
            return bool(self.beliefs.known_fires(self.model.steps))
        return True

    def execute_option(self):
        m = self.model
        known = self.beliefs.known_fires(m.steps)
        if self.option == FOLLOW_TASK:
            area = [c for c in known if abs(c[0] - self.task[0]) + abs(c[1] - self.task[1]) <= 4]
            goal = min(area, key=lambda c: (max(abs(c[0] - self.pos[0]), abs(c[1] - self.pos[1])), c)) \
                if area else self.task
        elif self.option == NEAREST:
            goal = min(known, key=lambda c: (max(abs(c[0] - self.pos[0]), abs(c[1] - self.pos[1])), c))
        else:
            self.explore()
            return
        if not m.move_down_field(self, m.env.distance_field([goal])):
            # arrived next to it but no fire there any more, or blocked
            if m.env.fire[goal] == 0:
                self.beliefs.observe([], [goal], m.steps)
            self.option_steps = m.commit  # force a new decision

    def go_refill(self):
        m = self.model
        if m.near_station(self.pos):
            if self.charges < m.extinguisher_charges:
                self.refilling = m.refill_time
            self.option_steps = m.commit
        else:
            m.move_down_field(self, m.station_field)

    def explore(self):
        """Search: climb the smoke gradient (like T-cell chemotaxis); random step otherwise."""
        m = self.model
        options = m.free_neighbours(self.pos)
        if not options:
            return
        here = m.env.smoke[self.pos]
        best = max(options, key=lambda c: (m.env.smoke[c], c))
        if m.env.smoke[best] > here + 1e-4 and self.random.random() < 0.8:
            m.grid.move_agent(self, best)
        else:
            m.grid.move_agent(self, self.random.choice(options))

    def drop_task(self, failure):
        if self.task is None:
            return
        m = self.model
        content = {"clear": []}
        if self.investigate and not failure:
            content["investigated"] = self.task
        self.task, self.investigate = None, False
        if m.dispatcher is not None:
            m.bus.send(FAILURE if failure else DONE, self.unique_id, m.dispatcher.unique_id, content, m.steps)


# ==========================================================================
class EvacuationRobot(mesa.Agent):
    """Direct a group to the exit with the lowest estimated travel and wait."""

    def __init__(self, model, junction):
        super().__init__(model)
        self.location = junction[0]
        self.junction = junction
        self.field = None
        # The assigned headcount falls as people leave the building or stop moving.
        self.assigned = Counter()

    def direct(self, occupants):
        m = self.model
        group = [a for a in occupants if a.state == "evacuating" and a.guided_exit is None
                 and self.field[a.pos] == 0]
        if not group:
            return

        if m.robot_guidance == "shared":
            demand = sum((robot.assigned for robot in m.robots), Counter())
        else:
            demand = Counter(a.guided_exit for a in occupants
                             if a.state == "evacuating" and a.guided_exit is not None
                             and max(abs(a.pos[0] - self.location[0]),
                                     abs(a.pos[1] - self.location[1])) <= m.vision)

        costs = {}
        for side, field in m.guided_exit_fields.items():
            walking_time = max(field[a.pos] for a in group)
            if np.isfinite(walking_time):
                queue_time = (demand[side] + len(group) - 1) / len(m.exits_by_side[side])
                costs[side] = walking_time + queue_time
        if not costs:
            return

        side = min(costs, key=costs.get)
        for a in group:
            a.guide = self
            a.guided_exit = side
        self.assigned[side] += len(group)


class Occupant(mesa.Agent):
    """Researcher. States: working -> alerted (pre-movement delay) -> evacuating.

    Heterogeneity: pre-movement delay ~ lognormal; a share of occupants only
    knows the west exits (where they came in); herding: an unaware occupant
    starts reacting when it sees two neighbours evacuating."""

    def __init__(self, model):
        super().__init__(model)
        self.state = "working"
        self.health = 1.0
        self.injured = False
        self.delay = 0
        self.familiar = self.random.random() < model.familiarity
        self.guide = None
        self.guided_exit = None

    def finish_guidance(self):
        if self.guide is not None:
            self.guide.assigned[self.guided_exit] -= 1

    def evacuation_field(self):
        m = self.model
        if self.guided_exit is not None:
            return m.guided_exit_fields[self.guided_exit]
        if m.robots:
            return min(m.robots, key=lambda robot: robot.field[self.pos]).field
        if not self.familiar and np.isfinite(m.west_exit_field[self.pos]):
            return m.west_exit_field
        return m.exit_field

    def step(self):
        if self.pos is None or self.state == "incapacitated":
            return
        m, env = self.model, self.model.env
        x, y = self.pos
        damage = m.smoke_damage * env.smoke[x, y] + m.fire_damage * env.fire[x, y]
        self.health = max(0.0, self.health - damage)
        m.total_dose += damage
        if self.health < 0.7:
            self.injured = True
        if self.health <= 0.0:
            self.state = "incapacitated"
            m.incapacitated += 1
            self.finish_guidance()
            return

        if self.state == "working":
            room = env.room_of(self.pos)
            sees_fire = any(env.room_of(c) == room for c in env.fire_cells_within(self.pos, 3))
            herd = sum(1 for a in m.grid.get_neighbors(self.pos, moore=True, radius=2)
                       if isinstance(a, Occupant) and a.state == "evacuating")
            if m.alarm_on or env.smoke[x, y] > 0.15 or sees_fire or herd >= 2:
                self.state = "alerted"
                self.delay = int(round(self.random.lognormvariate(m.premovement_mu, m.premovement_sigma)))
            else:
                if self.random.random() < 0.2:
                    m.move_random(self)
                return

        if self.state == "alerted":
            if self.delay > 0:
                self.delay -= 1
                return
            self.state = "evacuating"

        m.move_down_field(self, self.evacuation_field())
        if self.guided_exit is not None:
            exits = m.exits_by_side[self.guided_exit]
        elif not m.robots:
            exits = env.exits
        else:
            return
        if self.pos in exits:
            m.evacuate(self)
