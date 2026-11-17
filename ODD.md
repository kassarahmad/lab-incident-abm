# ODD protocol – Lab Incident Response model

Model description following the ODD protocol (Overview, Design concepts,
Details; Grimm et al. 2006, updated 2020).

## 1. Purpose and patterns

**Purpose.** To explore how a central dispatcher, an unreliable radio, the
number of responders and the delay of the fire alarm affect the containment
of a fire in a university laboratory and the safety of the people evacuating it.

**Questions.**
- Q1 Does a dispatcher contain the fire better than responders acting on their own?
- Q2 What happens when part of the radio messages is lost?
- Q3 How do team size and alarm delay change containment and occupant exposure?
- Q4 How does sharing how many occupants are heading to each exit affect evacuation time?

**Patterns the model should show:** fire growth limited by fuel; smoke filling
rooms faster than fire spreads; queueing at doors during evacuation; longer
detection-to-intervention times worsen outcomes.

## 2. Entities, state variables and scales

| Entity | State variables |
|---|---|
| Cell (40 × 20 grid) | tile type (wall, floor, bench, cabinet, exit, station), fuel ∈ [0, 2], fire intensity ∈ [0, 1], smoke ∈ [0, 1], ever burned |
| SmokeDetector (≈ 40, fixed) | location, room, triggered |
| Dispatcher (1 or 0, not spatial) | belief map of the fire, last reported status of each responder (position, charges, refilling), last order sent, alarmed zones |
| Responder (default 4) | position, charges (10), refill timer, belief map, task, current option and how long it has been kept |
| EvacuationRobot (2 in guided scenarios) | fixed junction, route distances, number of occupants it has directed who are still heading to each exit |
| Occupant (default 40) | position, state (working / alerted / evacuating / incapacitated), health ∈ [0, 1], reaction delay, exit familiarity, assigned guide and exit when guided |
| MessageBus | messages in flight (type, sender, receiver, content, delivery step), latency, loss probability |

One cell represents roughly 0.5 m and one step roughly 1 s in the illustrative
model. A run stops when the fire is out and all occupants have either left
or become incapacitated, or at 300 steps.

## 3. Process overview and scheduling

Each step:
1. **Environment (synchronous):** ignition at step 3; fire growth, spread,
   fuel consumption; smoke emission, diffusion, venting. Escape paths are
   recomputed avoiding cells with fire > 0.3. In guided scenarios, ignition
   closes the main corridor and triggers the evacuation alarm at the default
   delay of zero. Due messages are delivered.
2. **Sensing and dispatch:** detectors sense and report; the alarm sounds
   `alarm_delay` steps after the first detection; the dispatcher reads its
   messages and sends orders. Each robot directs the group waiting in its
   junction to a reachable exit.
3. **People (random sequential order, reshuffled every step):** responders
   and occupants act; later agents see earlier agents' moves.
4. **Observation:** observables are recorded; stopping rule checked.

## 4. Design concepts

- **Basic principles.** Cellular fire spread (probability ∝ intensity × fuel),
  smoke diffusion, breadth-first-search navigation, message passing.
- **Emergence.** Containment time, burned area, door congestion, evacuation
  time and exposure emerge from local fire dynamics, movement and messages.
- **Adaptation.** Responders choose what to do from what they believe:
  dispatcher's task first, then the nearest known fire, otherwise follow the
  smoke. Occupants react to alarm, smoke, flames and neighbours.
- **Objectives.** Implicit: responders reduce the fire they know about;
  occupants reach an exit.
- **Learning.** None.
- **Prediction.** Robots estimate walking and queueing time for each reachable exit.
- **Sensing.** Detectors: smoke above 0.04 or flames within 3 cells in the same
  room, missed with probability 1 − reliability. Responders: fire within 5 cells
  in the same room (or seen from the corridor). Occupants: alarm, smoke > 0.15
  on their cell, flames within 3 cells in the same room, or ≥ 2 neighbours
  evacuating (herding).
- **Interaction.** Indirect through the fields and space (one person per cell;
  a responder can swap places with someone in a door). In guided runs, two
  occupants can pass when both moves bring them closer to their destinations.
  Responders and the dispatcher communicate through radio messages.
- **Stochasticity.** Ignition cabinet(s), fire spread, detector misses,
  message loss, occupant positions, exit knowledge, reaction delays
  (lognormal, median ≈ 3.3 steps), tie breaking and activation order.
- **Collectives.** The response team.
- **Observation.** Burning cells, burned area, inside / evacuated /
  incapacitated, exposure, shots, messages per step; per run: containment
  step, response time, evacuation time, injured, messages sent and lost.
  Evacuation experiments record the step when everyone leaves, the peak
  number of occupants waiting behind another person, and wait counts by cell.

## 5. Initialization

Floor plan as in `environment.py` (4 labs, central corridor, 2 exits on each
side, 1 station). Fuel: floor 0.3, bench 1.0, cabinet 2.0. Detectors every 6
cells in x and 3 in y on air cells. Responders at the station; occupants on
random lab floor cells. Ignition: `n_ignitions` random cabinets at step 3,
intensity 0.5. All randomness derives from the run seed, so a seed reproduces
a run exactly.

Guided scenarios use a bypass around the central closure, with one usable
exit at each end and one fixed robot at each end of the bypass.

## 6. Input data

None.

## 7. Submodels

**Fire.** Burning cells with fuel grow by 0.08 per step; without fuel they
decay ×0.7. An unburning cell with fuel ignites with probability
(1 − Π_n (1 − p_spread · I_n)) · min(1, fuel), over its 4 neighbours n.
Fuel decreases by 0.015 · intensity.

**Smoke.** +0.25 · fire per step; 4 diffusion sub-steps with coefficient 0.5
over air neighbours (walls block it); ×0.97 venting, ×0.5 at exits.

**Messages.** INFORM (fire seen, cells clear, own position), REQUEST (go to a
cell), DONE / FAILURE (task finished / extinguisher empty). Each message is
delivered after `comm_latency` steps or lost with probability `comm_loss`.
Belief maps keep the most recent observation per cell; a believed fire older
than 40 steps is forgotten.

**Dispatcher.** Each step, each responder that last reported itself free
(charges > 0, not refilling), at its last reported position, gets the known
fire cell (or unchecked alarmed zone) minimising Manhattan distance + 8 ×
(number of targets already given within 3 cells). A REQUEST is sent only if
the target changes. A detector that alarmed in a room where no fire is known
becomes a zone to check (one per room).

**Responder.** Reads messages; observes and reports to the dispatcher (at least
every 3 steps as a radio check-in); refills when empty (go to the station,
3 steps); sprays the strongest fire within reach (−0.5 intensity, one charge);
otherwise keeps or chooses an option (re-chosen after 4 steps or when it is no
longer valid) and moves one cell along a BFS field.

**Occupant.** Exposure damage 0.05 · smoke + 0.25 · fire per step; injured
below 0.7, incapacitated at 0. After alerting, waits the reaction delay and
walks to the nearest exit (or to the west exits if it only knows those),
avoiding fire; queues behind other occupants.

**Robot guidance.** Evacuating occupants go to the nearest reachable guide.
Each guide assigns its waiting group to the exit that minimizes estimated
walking time plus estimated wait. Walking time is the largest shortest-path
distance among group members. Estimated wait is the number already headed
to that exit plus the group size minus one, divided by exit width in cells.
`local` counts directed evacuees within five cells of the junction.
`shared` uses the sum of both robots' active counts by exit. Shared counts
update immediately as people are assigned, evacuate or become incapacitated.

## Parameters

| Parameter | Default | Values in experiments |
|---|---|---|
| coordination | dispatcher | none, dispatcher |
| n_responders | 4 | 2, 4, 8 |
| alarm_delay | 0 | 0, 10, 20, 30 |
| comm_loss | 0 | 0, 0.2, 0.5 |
| n_ignitions | 1 | 1, 2 |
| comm_latency | 1 | – |
| p_spread | 0.25 | – |
| detector_reliability | 0.9 | – |
| extinguisher_charges | 10 | – |
| n_occupants | 40 | – |
| familiarity | 0.6 | – |
| robot_guidance | None | local, shared |
| refill_time | 3 | – |

Values are illustrative, not calibrated to real fire or evacuation data.
