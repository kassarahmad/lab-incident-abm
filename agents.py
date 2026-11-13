"""Agents of the lab incident model.

SmokeDetector  fixed sensor, feeds the alarm panel and reports to the coordinator
Coordinator    control room: Contract Net manager or greedy dispatcher
Responder      BDI-like safety officer: beliefs (fire map), desires (put the fire out,
               stay supplied), intentions (the option it is committed to)
Occupant       heterogeneous evacuee: pre-movement delay, herding, exit familiarity
"""

import math

import mesa
import numpy as np

from environment import MOORE, fire_clusters
from messaging import (ACCEPT, CFP, DONE, FAILURE, INFORM, PROPOSE, REFUSE, REJECT,
                       REQUEST, BeliefMap)
from policies import EXPLORE, FOLLOW_TASK, FRONT, NEAREST, REFILL, encode_state


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
            if m.coordinator is not None:
                m.bus.send(INFORM, self.unique_id, m.coordinator.unique_id,
                           {"burning": [(c, m.steps, True) for c in flames],
                            "detector": self.location}, m.steps)
        elif not sensing:
            self.triggered = False


# ==========================================================================
class Coordinator(mesa.Agent):
    """Control room (not on the grid). Knows only what detectors and responders
    tell it. Two strategies:

    central_greedy  every step, send each free responder to the nearest believed
                    fire (penalising crowding). Direct REQUEST messages.
    contract_net    Contract Net Protocol (Smith 1980): tasks = believed fire
                    clusters; CFP -> PROPOSE/REFUSE -> ACCEPT/REJECT -> DONE/FAILURE.
    """

    def __init__(self, model):
        super().__init__(model)
        self.beliefs = BeliefMap()
        self.contracts = {}          # responder id -> task target cell
        self.last_contact = {}       # responder id -> step of last message
        self.status = {}             # responder id -> (pos, charges, refilling, step) last reported
        self.open_round = None       # (deadline, tasks, proposals)
        self.last_assignment = {}
        self.zone_alerts = {}        # detector location -> last step it reported
        self.investigated = {}       # detector location -> step a responder checked it

    def step(self):
        m = self.model
        for msg in m.bus.receive(self.unique_id):
            self.last_contact[msg.sender] = m.steps
            if msg.performative == INFORM:
                self.beliefs.merge(msg.content["burning"])
                self.beliefs.merge(msg.content.get("clear", []))
                if "status" in msg.content:
                    self.status[msg.sender] = msg.content["status"]
                if "detector" in msg.content:
                    self.zone_alerts[msg.content["detector"]] = msg.sent
            elif msg.performative in (DONE, FAILURE):
                self.contracts.pop(msg.sender, None)
                if msg.content and "investigated" in msg.content:
                    self.investigated[tuple(msg.content["investigated"])] = m.steps
                self.beliefs.merge(msg.content.get("clear", []) if msg.content else [])
            elif msg.performative in (PROPOSE, REFUSE) and self.open_round is not None:
                self.open_round["proposals"][msg.sender] = msg.content if msg.performative == PROPOSE else None
        if not m.alarm_on:
            return
        known = self.beliefs.known_fires(m.steps)
        suspects = self.suspect_zones(known)
        if m.coordination == "central_greedy":
            self.greedy_dispatch(known, suspects)
        else:
            self.contract_net(known, suspects)

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

    # ---------------------------------------------------------- greedy
    def greedy_dispatch(self, known, suspects):
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

    # ---------------------------------------------------------- contract net
    def make_tasks(self, known, suspects):
        """One task per believed fire cluster (target = cell with the highest
        spread risk, need grows with size) and one 'investigate' task per
        suspect detector zone."""
        tasks = []
        for i, z in enumerate(suspects):
            if not any(abs(t[0] - z[0]) + abs(t[1] - z[1]) <= 2 for t, _, _ in self.contracts.values()):
                tasks.append({"id": f"{self.model.steps}-z{i}", "target": z, "cells": [],
                              "need": 1, "investigate": True})
        for i, cluster in enumerate(fire_clusters(known)):
            target = max(cluster, key=lambda c: (self.model.env.spread_risk(c), c))
            need = min(2 + len(cluster) // 4, 4)
            covered = sum(1 for t, _, _ in self.contracts.values() if t in cluster or
                          min(abs(t[0] - c[0]) + abs(t[1] - c[1]) for c in cluster) <= 2)
            if covered < need:
                tasks.append({"id": f"{self.model.steps}-{i}", "target": target,
                              "cells": cluster, "need": need - covered, "investigate": False})
        return tasks

    def contract_net(self, known, suspects):
        m = self.model
        known_set = set(known)
        # Drop contracts whose fire is believed out, or whose contractor went
        # silent (lost ACCEPT, lost DONE): timeouts make the protocol robust
        for rid, (t, since, investigate) in list(self.contracts.items()):
            silent = m.steps - max(since, self.last_contact.get(rid, -1)) > m.contract_timeout
            fire_gone = not investigate and not any(abs(t[0] - c[0]) + abs(t[1] - c[1]) <= 3 for c in known_set)
            if silent or fire_gone:
                del self.contracts[rid]

        if self.open_round is not None and m.steps >= self.open_round["deadline"]:
            self.award(self.open_round)
            self.open_round = None

        if self.open_round is None and (known or suspects):
            tasks = self.make_tasks(known, suspects)
            if tasks:
                bidders = [r.unique_id for r in m.responder_list if r.unique_id not in self.contracts]
                if bidders:
                    content = [{"id": t["id"], "target": t["target"]} for t in tasks]  # task announcement
                    m.bus.broadcast(CFP, self.unique_id, bidders, content, m.steps)
                    # wait for replies: message there and back
                    self.open_round = {"deadline": m.steps + 2 * m.bus.latency + 1,
                                       "tasks": tasks, "proposals": {}}

    def award(self, rnd):
        """Greedy assignment over all (bid cost, responder, task) triples:
        cheapest bid first, each responder wins at most one task."""
        m = self.model
        bids = []
        for rid, offer in rnd["proposals"].items():
            if offer:
                for tid, cost in offer.items():
                    bids.append((cost, rid, tid))
        bids.sort()
        need = {t["id"]: t["need"] for t in rnd["tasks"]}
        target = {t["id"]: t["target"] for t in rnd["tasks"]}
        kind = {t["id"]: t["investigate"] for t in rnd["tasks"]}
        winners = {}
        # pass 1: cover each task's minimum need with the cheapest bidders
        for cost, rid, tid in bids:
            if rid in winners or need[tid] <= 0 or rid in self.contracts:
                continue
            winners[rid] = tid
            need[tid] -= 1
        # pass 2 (full deployment): remaining bidders join their cheapest task
        if self.model.full_deployment:
            for cost, rid, tid in bids:
                if rid not in winners and rid not in self.contracts:
                    winners[rid] = tid
        for rid, offer in rnd["proposals"].items():
            if rid in winners:
                tid = winners[rid]
                self.contracts[rid] = (target[tid], m.steps, kind[tid])
                cells = next(t["cells"] for t in rnd["tasks"] if t["id"] == tid)
                m.bus.send(ACCEPT, self.unique_id, rid, {"id": tid, "target": target[tid],
                                                         "cells": cells, "investigate": kind[tid]}, m.steps)
            elif offer:
                m.bus.send(REJECT, self.unique_id, rid, None, m.steps)


# ==========================================================================
class Responder(mesa.Agent):
    """Lab safety officer with an extinguisher.

    Beliefs   : BeliefMap of the fire (own vision + messages)
    Reflexes  : shoot fire within reach; refill when empty
    Intention : one option (policies.py) kept for up to `commit` steps
    """

    def __init__(self, model, policy):
        super().__init__(model)
        self.policy = policy
        self.charges = model.extinguisher_charges
        self.beliefs = BeliefMap()
        self.task = None            # target cell from the coordinator
        self.investigate = False    # task is "go and check a detector zone"
        self.refilling = 0
        self.shots = 0
        # option bookkeeping (for SMDP learning)
        self.option = None
        self.option_goal = None
        self.option_steps = 0
        self.option_state = None
        self.option_reward = 0.0
        self.option_tau = 0
        self.local_reward = 0.0

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
        report = {"burning": [(c, m.steps, True) for c in seen],
                  "clear": [(c, m.steps, False) for c in clear]}
        if m.coordination == "peer":
            if seen or clear:
                m.bus.broadcast(INFORM, self.unique_id, m.responder_ids, report, m.steps)
        elif m.coordinator is not None and m.alarm_on:
            # observations when there are any, otherwise a periodic radio check-in;
            # the coordinator only knows where responders are from these messages
            if seen or clear or m.steps % m.status_period == 0:
                report["status"] = (self.pos, self.charges, self.refilling > 0, m.steps)
                m.bus.send(INFORM, self.unique_id, m.coordinator.unique_id, report, m.steps)
        return seen

    def handle_messages(self):
        m = self.model
        for msg in m.bus.receive(self.unique_id):
            p = msg.performative
            if p == INFORM:
                self.beliefs.merge(msg.content["burning"])
                self.beliefs.merge(msg.content.get("clear", []))
            elif p == CFP:
                if self.charges == 0 or self.refilling or self.task is not None:
                    m.bus.send(REFUSE, self.unique_id, msg.sender, None, m.steps)
                else:
                    offer = {t["id"]: self.bid(t["target"]) for t in msg.content}
                    m.bus.send(PROPOSE, self.unique_id, msg.sender, offer, m.steps)
            elif p in (ACCEPT, REQUEST):
                if self.charges > 0:
                    self.task = tuple(msg.content["target"])
                    self.investigate = bool(msg.content.get("investigate"))
                    cells = msg.content.get("cells", [] if self.investigate else [self.task])
                    self.beliefs.merge([(tuple(c), msg.sent, True) for c in cells])
                elif p == ACCEPT:
                    m.bus.send(FAILURE, self.unique_id, msg.sender, None, m.steps)

    def bid(self, target):
        """Cost = path length to the task + penalty for a nearly empty extinguisher."""
        d = self.model.env.distance_field([tuple(target)])[self.pos]
        if math.isinf(d):
            d = 200
        low = 15 if self.charges < 0.3 * self.model.extinguisher_charges else 0
        return float(d + low)

    # ---------------------------------------------------------- state for learning
    def state(self):
        m = self.model
        known = self.beliefs.known_fires(m.steps)
        if known:
            dist = min(max(abs(c[0] - self.pos[0]), abs(c[1] - self.pos[1])) for c in known)
            near = min(known, key=lambda c: abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]))
            mates = sum(1 for r in m.responder_list if r is not self and
                        max(abs(r.pos[0] - near[0]), abs(r.pos[1] - near[1])) <= 4)
        else:
            dist, mates = 99, 0
        return encode_state(self.charges, m.extinguisher_charges, len(known), dist,
                            self.task is not None, mates)

    def valid_options(self):
        opts = [EXPLORE]
        if self.task is not None:
            opts.append(FOLLOW_TASK)
        if self.beliefs.known_fires(self.model.steps):
            opts += [NEAREST, FRONT]
        if self.charges < self.model.extinguisher_charges:
            opts.append(REFILL)
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
            self.end_option()
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
            removed = min(env.fire[best], m.extinguish_power)
            env.fire[best] = max(0.0, env.fire[best] - m.extinguish_power)
            self.local_reward += m.shot_reward * removed   # individual shaping term
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

        # Intention: keep the current option or choose a new one
        if self.option is None or self.option_steps >= m.commit or not self.option_valid():
            self.end_option()
            s = self.state()
            self.option = self.policy.choose(s, self.valid_options(), self.random)
            self.option_state = s
            self.option_steps = 0
            self.option_reward, self.option_tau = 0.0, 0
        self.execute_option()
        self.option_steps += 1

    def option_valid(self):
        if self.option == FOLLOW_TASK:
            return self.task is not None
        if self.option in (NEAREST, FRONT):
            return bool(self.beliefs.known_fires(self.model.steps))
        if self.option == REFILL:
            return self.refilling == 0 and self.charges < self.model.extinguisher_charges
        return True

    def execute_option(self):
        m = self.model
        known = self.beliefs.known_fires(m.steps)
        o = self.option
        if o == FOLLOW_TASK:
            area = [c for c in known if abs(c[0] - self.task[0]) + abs(c[1] - self.task[1]) <= 4]
            goal = min(area, key=lambda c: (max(abs(c[0] - self.pos[0]), abs(c[1] - self.pos[1])), c)) \
                if area else self.task
        elif o == NEAREST:
            goal = min(known, key=lambda c: (max(abs(c[0] - self.pos[0]), abs(c[1] - self.pos[1])), c))
        elif o == FRONT:
            goal = max(known, key=lambda c: (m.env.spread_risk(c), -abs(c[0] - self.pos[0]) - abs(c[1] - self.pos[1]), c))
        elif o == REFILL:
            self.go_refill()
            return
        else:
            self.explore()
            return
        if not m.move_down_field(self, m.env.distance_field([goal])):
            # arrived next to it but no fire there any more, or blocked
            self.beliefs.observe([], [goal], m.steps) if m.env.fire[goal] == 0 else None
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
        if m.coordinator is not None:
            m.bus.send(FAILURE if failure else DONE, self.unique_id, m.coordinator.unique_id, content, m.steps)

    # ---------------------------------------------------------- learning hooks
    def add_reward(self, team_reward):
        """Team reward shared by all + individual reward for fire put out this step."""
        r = team_reward + self.local_reward
        self.local_reward = 0.0
        if self.option is not None:
            self.option_reward += (self.policy_gamma() ** self.option_tau) * r
            self.option_tau += 1

    def policy_gamma(self):
        return getattr(self.policy, "gamma", 1.0)

    def end_option(self, terminal=False):
        if self.option is None or self.option_tau == 0:
            self.option = None
            return
        s_next = self.state()
        self.policy.update(self.option_state, self.option, self.option_reward, self.option_tau,
                           s_next, self.valid_options(), terminal)
        self.model.option_counts[self.option] += 1
        self.option = None


# ==========================================================================
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

        field = m.exit_field if self.familiar else m.west_exit_field
        if np.isinf(field[self.pos]):
            field = m.exit_field
        m.move_down_field(self, field)
        if self.pos is not None and env.tiles[self.pos] == "E":
            m.evacuate(self)
