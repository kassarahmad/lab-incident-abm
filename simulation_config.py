"""Experiment design shared by the batch runner and the browser GUI.

Every configuration is run with the same seeds (common random numbers), so
two configurations face exactly the same fire start, occupants and luck;
differences are then tested with paired tests.
"""

DEFAULT_SEED = 1000

# Scenarios selectable in the GUI
PRESETS = {
    "contract_net": {"name": "Contract Net (4 responders)", "params": {"coordination": "contract_net"}},
    "greedy": {"name": "Central dispatcher (greedy)", "params": {"coordination": "central_greedy"}},
    "peer": {"name": "Peer-to-peer sharing", "params": {"coordination": "peer"}},
    "none": {"name": "No communication", "params": {"coordination": "none"}},
    "learned": {"name": "Contract Net + learned policy", "params": {"coordination": "contract_net", "policy": "learned"}},
    "lossy": {"name": "Contract Net, 50% message loss", "params": {"coordination": "contract_net", "comm_loss": 0.5}},
    "two_fires": {"name": "Two simultaneous fires", "params": {"coordination": "contract_net", "n_ignitions": 2}},
    "late_alarm": {"name": "Alarm delayed 20 steps", "params": {"coordination": "contract_net", "alarm_delay": 20}},
    "no_response": {"name": "No responders (control)", "params": {"n_responders": 0}},
}

# Batch experiments. Each experiment: question, configurations, and the
# planned comparisons (reference, other). Holm correction is applied within
# each experiment and outcome.
COORD = {"No communication": "none", "Peer-to-peer": "peer",
         "Greedy dispatcher": "central_greedy", "Contract Net": "contract_net"}


def _fires(k):
    return f"{k} fire{'s' if k > 1 else ''}"


EXPERIMENTS = {
    "E1_coordination": {
        "question": "Which coordination mechanism contains the fire best?",
        "configs": {f"{label} | {_fires(k)}": {"coordination": mode, "n_ignitions": k}
                    for k in (1, 2) for label, mode in COORD.items()},
        "comparisons": [(f"No communication | {_fires(k)}", f"{label} | {_fires(k)}")
                        for k in (1, 2) for label in list(COORD)[1:]]
                       + [(f"Greedy dispatcher | {_fires(k)}", f"Contract Net | {_fires(k)}") for k in (1, 2)],
    },
    "E2_communication": {
        "question": "How robust are the centralized protocols to message loss?",
        "configs": {f"{label} | loss {int(loss * 100)}%": {"coordination": mode, "comm_loss": loss}
                    for label, mode in (("Greedy dispatcher", "central_greedy"), ("Contract Net", "contract_net"))
                    for loss in (0.0, 0.2, 0.5)},
        "comparisons": [(f"{label} | loss 0%", f"{label} | loss {l}%")
                        for label in ("Greedy dispatcher", "Contract Net") for l in (20, 50)],
    },
    "E3_learning": {
        "question": "Does the learned (Q-learning) option policy beat the hand-written rules? (2 fires)",
        "configs": {f"{pol.capitalize()} | {label}": {"coordination": mode, "policy": pol, "n_ignitions": 2}
                    for label, mode in (("No communication", "none"), ("Peer-to-peer", "peer"),
                                        ("Contract Net", "contract_net"))
                    for pol in ("rule", "learned")},
        "comparisons": [(f"Rule | {label}", f"Learned | {label}")
                        for label in ("No communication", "Peer-to-peer", "Contract Net")],
    },
    "E4_resources": {
        "question": "How do team size and alarm delay change the outcome? (Contract Net)",
        "configs": {**{f"{n} responders | delay 0": {"coordination": "contract_net", "n_responders": n}
                       for n in (2, 4, 8)},
                    **{f"4 responders | delay {d}": {"coordination": "contract_net", "alarm_delay": d}
                       for d in (10, 20, 30)}},
        "comparisons": [("4 responders | delay 0", x) for x in
                        ("2 responders | delay 0", "8 responders | delay 0", "4 responders | delay 10",
                         "4 responders | delay 20", "4 responders | delay 30")],
    },
}
