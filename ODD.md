# ODD protocol – Lab Incident Response model

Model description following the ODD protocol (Overview, Design concepts,
Details; Grimm et al. 2006, updated 2020).

## 1. Purpose and patterns

**Purpose.** To explore how detection, communication and coordination
mechanisms, and adaptive (learned) decision making of emergency responders,
affect the containment of a fire in a university laboratory and the safety of
the people evacuating it.

**Questions.**
- Q1 Which coordination mechanism contains the fire best: none, peer-to-peer
  information sharing, a central greedy dispatcher, or the Contract Net Protocol?
- Q2 How robust are the centralized mechanisms to an unreliable channel
  (message loss)?
- Q3 Can responders learn a better option policy (SMDP Q-learning) than a
  hand-written rule set?
- Q4 How do team size and alarm delay change containment and occupant exposure?

**Patterns the model must reproduce** (used as qualitative validation):
fire growth limited by fuel; smoke filling rooms faster than fire spreads;
queueing at doors during evacuation; congestion when responders move against
the evacuation flow; longer detection-to-intervention times worsen outcomes.

## 2. Entities, state variables and scales

| Entity | State variables |
|---|---|
| Cell (40 × 20 grid) | tile type (wall, floor, bench, cabinet, exit, station), fuel ∈ [0, 2], fire intensity ∈ [0, 1], smoke ∈ [0, 1], ever burned |
| SmokeDetector (≈ 40, fixed) | location, room, triggered |
| Coordinator (1 or 0, not spatial) | belief map, last reported status of each responder (position, charges, refilling), contracts (responder → target, start step), last contact per responder, open CFP round |
| Responder (default 4) | position, charges (10), refill timer, belief map, task, current option, option start state, accumulated option reward |
| Occupant (default 40) | position, state (working / alerted / evacuating / incapacitated), health ∈ [0, 1], pre-movement delay, exit familiarity, injured |
| MessageBus | messages in flight (performative, sender, receiver, content, delivery step), latency, loss probability |

One cell ≈ 0.5 m; one step ≈ 1 s (illustrative). A run lasts until the fire is
out and nobody is left inside, or 300 steps.

## 3. Process overview and scheduling

Each step:
1. **Environment (synchronous):** ignition at step 3; fire growth, spread,
   fuel consumption; smoke emission, diffusion (4 sub-steps), venting.
   Escape-route distance fields are recomputed avoiding cells with fire > 0.3.
   The bus delivers messages whose delivery step has come.
2. **Sensing and coordination:** detectors sense and report; the alarm panel
   sounds `alarm_delay` steps after the first detection; the coordinator
   processes its inbox and dispatches (greedy) or runs the Contract Net.
3. **People (random sequential order, reshuffled every step):** responders
   and occupants act; later agents see earlier agents' moves.
4. **Reward and observation:** team reward is credited to every responder's
   running option; observables are recorded; stopping rule checked.

## 4. Design concepts

- **Basic principles.** Cellular fire spread (probability ∝ intensity × fuel),
  diffusion of smoke, BFS shortest-path navigation, FIPA-ACL messaging,
  Contract Net Protocol (Smith 1980), options framework (Sutton, Precup &
  Singh 1999), SMDP Q-learning.
- **Emergence.** Containment time, burned area, door congestion, evacuation
  time and exposure are not imposed; they emerge from local fire dynamics,
  agent movement and communication.
- **Adaptation.** Responders choose among five options (follow task, nearest
  known fire, fire front with most fuel, explore smoke gradient, refill early).
  The rule policy is fixed; the learned policy uses a Q-table trained over
  1500 incidents (random coordination mode and number of fires) and is
  evaluated greedily (no exploration).
- **Objectives.** Reward of responder i per step = team reward
  [−(newly burned cells) − 0.05 × (burning cells)] + 2 × (fire intensity i
  removed this step). The individual term eases credit assignment.
- **Learning.** Tabular SMDP Q-learning, step size max(0.1, 1/n^0.6) with n the
  visits of (s, o), γ = 0.95, ε decaying 0.3 → 0.05, one Q-table shared by all
  responders (parameter sharing, independent learners). Every 150 episodes the
  greedy policy is evaluated on 24 validation incidents (seeds 5000–5011); the
  best snapshot is kept (model selection). Experiments use different seeds
  (1000–1029), so reported effects are out-of-sample.
  State: charges (4 levels), known fire size (4), distance to nearest known
  fire (3), has task (2), teammates near that fire (3) = 288 states.
- **Prediction.** None: agents react to their current beliefs.
- **Sensing.** Detectors: smoke above 0.04 or flames within 3 cells in the same
  room, missed with probability 1 − reliability. Responders: fire within 5 cells
  in the same room (or seen from the corridor). Occupants: alarm, smoke > 0.15
  on their cell, flames within 3 cells in the same room, or ≥ 2 neighbours
  evacuating (herding).
- **Interaction.** Indirect via the shared fields and space (one person per
  cell; counter-flow swaps with responders); direct via messages.
- **Stochasticity.** Ignition cabinet(s), fire spread, detector misses,
  message loss, occupant positions, familiarity, pre-movement delays
  (lognormal, median ≈ 3.3 steps), tie breaking and activation order.
- **Collectives.** The response team; in Contract Net, groups of contractors on
  the same fire cluster (task).
- **Observation.** Burning cells, burned area, inside / evacuated /
  incapacitated, exposure dose, shots, messages per step; per run: containment
  step, evacuation time, injured, messages sent and dropped, team reward.

## 5. Initialization

Floor plan as in `environment.py` (4 labs, central corridor, 2 exits on each
side, 1 station). Fuel: floor 0.3, bench 1.0, cabinet 2.0. Detectors every 6
cells in x and 3 in y on air cells. Responders at the station; occupants on
random lab floor cells. Ignition: `n_ignitions` random cabinets at step 3,
intensity 0.5. All randomness derives from the run seed (Mesa `rng`), so a
seed reproduces a run exactly.

## 6. Input data

None (no external time series).

## 7. Submodels

**Fire.** Burning cells with fuel grow by 0.08 per step; without fuel they
decay ×0.7. An unburning cell with fuel ignites with probability
(1 − Π_n (1 − p_spread · I_n)) · min(1, fuel), over its 4 neighbours n.
Fuel decreases by 0.015 · intensity.

**Smoke.** +0.25 · fire per step; 4 explicit diffusion sub-steps with
coefficient 0.5 over air neighbours (walls are impermeable); ×0.97 venting,
×0.5 at exits.

**Messaging.** Each message is delivered after `comm_latency` steps or lost
with probability `comm_loss`. Belief maps keep the most recent observation per
cell; a believed fire older than 40 steps is forgotten.

**Coordinator – greedy.** Each step, each responder that last reported itself
free (charges > 0, not refilling), at its last reported position, gets the believed fire cell or suspect zone minimising Manhattan distance + 8 ×
(number of already assigned targets within 3 cells). REQUEST is sent only if
the target changes.

**Suspect zones.** A detector that alarmed in the last 10 steps, in a room
where the coordinator knows no fire and that nobody checked in the last 30
steps, becomes an *investigate* target (one per room): someone must go and look.

**Coordinator – Contract Net.** Tasks are 8-connected clusters of believed
fire, plus one investigate task (need 1) per suspect zone; target = cell with most unburned fuel around it; need = min(2 + size/4, 4)
minus current contractors. CFP to all responders without a contract; they
answer PROPOSE (cost = path length + 15 if charges < 30 %) or REFUSE. After
2 × latency + 1 steps the coordinator awards greedily by cost (pass 1: cover each
task's need; pass 2, full deployment: every remaining bidder joins its
cheapest task) and sends ACCEPT / REJECT with the cluster's cells. Contractors
report DONE or FAILURE; contracts silent for 12 steps or whose fire is
believed out are cancelled (timeouts make the protocol tolerate lost messages).

**Responder.** Reads messages; observes and reports (peer: broadcast of
observations to teammates; centralized: observations plus own status to the
coordinator, at least every 3 steps as a radio check-in); reflexes: refill when empty
(go to the station, 3 steps), shoot the strongest fire within reach (−0.5
intensity, one charge); otherwise keep or choose an option (re-chosen after 4
steps or when it becomes invalid) and move one cell along the BFS field.

**Occupant.** Exposure damage 0.05 · smoke + 0.25 · fire per step; injured
below 0.7, incapacitated at 0. After alerting, waits the pre-movement delay
and walks to the nearest exit (familiar occupants) or to the west exits
(unfamiliar), avoiding fire; queues behind other occupants.

## Parameters

| Parameter | Default | Range in Sobol analysis |
|---|---|---|
| p_spread | 0.25 | 0.15–0.35 |
| n_responders | 4 | 2–8 |
| alarm_delay | 0 | 0–30 |
| comm_loss | 0 | 0–0.5 |
| comm_latency | 1 | – |
| detector_reliability | 0.9 | 0.6–1.0 |
| extinguisher_charges | 10 | 5–15 |
| n_occupants | 40 | – |
| familiarity | 0.6 | – |
| refill_time | 3 | – |

Values are illustrative, not calibrated to real fire or evacuation data.
