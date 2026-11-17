# Lab Incident Response – a Multi-Agent Model

Agent-Based Adaptive Modelling · Assignment 2 · Python / **Mesa 3.5**

A fire starts at a chemical cabinet on a university lab floor. Smoke detectors
start the alarm and inform a **dispatcher**; **responders** with limited
extinguishers fight the fire using what they have seen or been told by
**radio messages** (which can be late or lost); **occupants** evacuate.
Full model description in [ODD.md](ODD.md) (ODD protocol).

![run](plots/run_two_fires.gif)

## Run it

Windows (PowerShell, one command per line):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then:

```bash
python run_gui.py                                  # live simulator → http://127.0.0.1:8090
python run_experiments.py --runs 30 --jobs 4       # E1–E3, statistics, plots  (~3 min)
python render_run.py --preset two_fires            # GIF of one run
python -m unittest -v                              # 12 tests
```

## Files

| File | Contents |
|---|---|
| `environment.py` | Floor plan, fuel / fire / smoke fields, BFS path planning |
| `messaging.py` | Message types, `MessageBus` (delay, loss), `BeliefMap` |
| `agents.py` | `SmokeDetector`, `Dispatcher`, `Responder`, `Occupant` |
| `options.py` | What a responder does when no reflex applies |
| `model.py` | `LabIncidentModel`: step order, stopping rule, data collection |
| `simulation_config.py` | GUI scenarios and the experiments E1–E3 |
| `run_experiments.py` | 30 paired runs per condition, Wilcoxon test, plots |
| `run_gui.py`, `index.html`, `app.js`, `style.css` | Browser simulator; all rules run in Python |
| `render_run.py` | GIF of a single run |
| `test_model.py` | Unit tests |

## The model in short

- **Environment.** 40 × 20 cells: four labs with benches and chemical cabinets,
  a corridor, two exits on each side, one extinguisher station. Fire spreads to
  the 4 neighbours with probability ∝ intensity × fuel; smoke spreads through
  air (not walls) faster than fire.
- **Detectors** sense smoke or flames and miss 10 % of readings. The alarm
  sounds `alarm_delay` steps after the first detection.
- **Coordination.** `none`: responders use only their own eyes.
  `dispatcher`: a control room sends each free responder to the nearest known
  fire, using the positions responders report by radio.
- **Responders.** Each step: spray fire within reach; refill if empty; else
  follow the dispatcher's task; else go to the nearest fire they know; else
  follow the smoke.
- **Occupants.** Random reaction delay, herding, 40 % only know the west
  exits, health loss from smoke and fire, queueing at doors.

## Experiments

Each condition runs **30 times on the same seeds (1000–1029)**, so every
condition faces the same fires and people. Differences are tested with the
**Wilcoxon signed-rank test** (paired by seed), Holm-corrected. Full tables:
`results/runs.csv`, `results/stats.csv`.

### E1 – Does a dispatcher help?

| | burned cells | time to put out | alarm → first spray | contained |
|---|---:|---:|---:|---:|
| No dispatcher, 1 fire | 67.4 | 149 | 35.5 | 100 % |
| **Dispatcher, 1 fire** | **8.4** | **40** | **18.9** | 100 % |
| No dispatcher, 2 fires | 196.2 | 246 | 29.8 | 87 % |
| **Dispatcher, 2 fires** | **67.2** | **126** | **18.6** | 100 % |

The dispatcher hears the detectors, so responders start spraying about 17
steps earlier; burned area drops 8× with one fire and 3× with two
(p < 0.001).

### E2 – Lost messages (dispatcher)

| messages lost | 0 % | 20 % | 50 % |
|---|---:|---:|---:|
| burned cells | 8.4 | 20.5 | 26.9 |
| time to put out | 40 | 60 | 72 |

Worse (p ≤ 0.011) but every fire is still put out: responders keep their own
eyes and the check-ins every 3 steps repeat lost information. Evacuation time
does not change (≈ 54 steps).

### E3 – Team size and alarm delay (dispatcher)

| | burned cells | evacuation time | smoke breathed |
|---|---:|---:|---:|
| 2 responders | 47.1 | 54 | 0.052 |
| **4 responders** (reference) | 8.4 | 53 | 0.075 |
| 8 responders | 7.7 | 53 | 0.065 |
| alarm delay 10 | 43.2 | 62 | 0.110 |
| alarm delay 20 | 52.4 | 73 | 0.222 |
| alarm delay 30 | 68.4 | 82 | **0.374** |

Four responders are enough here. A late alarm is what puts people in danger:
30 steps of delay multiply the smoke they breathe by 5 (p < 0.001).

## Limitations

Illustrative parameters, not calibrated to real fire or evacuation data; 2D
floor; no doors that close or ventilation; responders are never hurt;
messages are lost only at random.
