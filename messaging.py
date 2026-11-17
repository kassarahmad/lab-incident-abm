"""Agent communication: FIPA-style messages over an unreliable bus, and belief maps.

The bus delivers each message after `latency` steps and drops it with
probability `loss`. Every agent therefore acts on *beliefs*, which can be
stale or incomplete, instead of on the true state of the fire.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

# Message types (FIPA-style names)
INFORM = "inform"            # observations of burning / extinguished cells, radio check-in
REQUEST = "request"          # order from the dispatcher: go to this cell
DONE = "inform-done"         # responder finished its task
FAILURE = "failure"          # responder gives up the task (out of charges)


@dataclass
class Message:
    performative: str
    sender: int
    receiver: int                 # unique_id of the receiver
    content: Any = None
    sent: int = 0
    deliver_at: int = 0


class MessageBus:
    def __init__(self, rng, latency=1, loss=0.0):
        self.rng = rng                 # python random.Random of the model
        self.latency = latency
        self.loss = loss
        self.in_flight = []
        self.inboxes = defaultdict(list)
        self.sent = Counter()
        self.dropped = Counter()
        self.log = []                  # recent messages, for the GUI

    def send(self, performative, sender, receiver, content, now):
        self.sent[performative] += 1
        if self.loss > 0 and self.rng.random() < self.loss:
            self.dropped[performative] += 1
            return
        self.in_flight.append(Message(performative, sender, receiver, content, now, now + self.latency))

    def broadcast(self, performative, sender, receivers, content, now):
        for r in receivers:
            if r != sender:
                self.send(performative, sender, r, content, now)

    def deliver(self, now):
        keep = []
        for m in self.in_flight:
            if m.deliver_at <= now:
                self.inboxes[m.receiver].append(m)
                self.log.append((now, m.performative, m.sender, m.receiver))
            else:
                keep.append(m)
        self.in_flight = keep
        self.log = self.log[-60:]

    def receive(self, agent_id):
        msgs = self.inboxes.pop(agent_id, [])
        return msgs

    @property
    def total_sent(self):
        return sum(self.sent.values())


@dataclass
class BeliefMap:
    """What one agent believes about the fire: cell -> (step observed, burning?).
    A newer observation always overrides an older one."""
    cells: dict = field(default_factory=dict)
    max_age: int = 40

    def observe(self, burning, clear, now):
        for c in burning:
            self.cells[c] = (now, True)
        for c in clear:
            if c in self.cells:
                self.cells[c] = (now, False)

    def merge(self, report):
        """report = list of (cell, step, burning) received from another agent."""
        changed = False
        for c, t, b in report:
            old = self.cells.get(c)
            if old is None or t > old[0]:
                self.cells[c] = (t, b)
                changed = True
        return changed

    def known_fires(self, now):
        return [c for c, (t, b) in self.cells.items() if b and now - t <= self.max_age]

    def as_report(self, cells, now):
        return [(c, now, True) for c in cells]
