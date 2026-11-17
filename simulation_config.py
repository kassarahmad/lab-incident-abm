"""Experiment design shared by the batch runner and the browser GUI.

Every configuration is run with the same seeds, so two configurations face
exactly the same fire start and the same occupants; the difference between
them is then due to the setting we changed, not to luck.
"""

DEFAULT_SEED = 1000

# Scenarios selectable in the GUI
PRESETS = {
    "dispatcher": {"name": "Dispatcher (4 responders)", "params": {"coordination": "dispatcher"}},
    "none": {"name": "No dispatcher", "params": {"coordination": "none"}},
    "two_fires": {"name": "Two fires, dispatcher", "params": {"coordination": "dispatcher", "n_ignitions": 2}},
    "lossy": {"name": "Dispatcher, 50% messages lost", "params": {"coordination": "dispatcher", "comm_loss": 0.5}},
    "late_alarm": {"name": "Alarm delayed 20 steps", "params": {"coordination": "dispatcher", "alarm_delay": 20}},
    "no_response": {"name": "No responders (control)", "params": {"n_responders": 0}},
    "blocked_corridor": {"name": "East corridor blocked", "params": {"block_corridor": True}},
    "local_guides": {"name": "Robot guides (local observations)", "params": {"robot_guidance": "local"}},
    "shared_guides": {"name": "Robot guides (shared counts)", "params": {"robot_guidance": "shared"}},
}


def _fires(k):
    return f"{k} fire{'s' if k > 1 else ''}"


# Batch experiments: question, configurations, planned comparisons (reference, other)
EXPERIMENTS = {
    "E1_coordination": {
        "question": "Does a dispatcher help? (no dispatcher vs dispatcher)",
        "configs": {f"{label} | {_fires(k)}": {"coordination": mode, "n_ignitions": k}
                    for k in (1, 2) for label, mode in (("No dispatcher", "none"), ("Dispatcher", "dispatcher"))},
        "comparisons": [(f"No dispatcher | {_fires(k)}", f"Dispatcher | {_fires(k)}") for k in (1, 2)],
    },
    "E2_communication": {
        "question": "What happens when radio messages get lost?",
        "configs": {f"Dispatcher | loss {int(loss * 100)}%": {"coordination": "dispatcher", "comm_loss": loss}
                    for loss in (0.0, 0.2, 0.5)},
        "comparisons": [("Dispatcher | loss 0%", f"Dispatcher | loss {l}%") for l in (20, 50)],
    },
    "E3_resources": {
        "question": "How do team size and alarm delay change the outcome?",
        "configs": {**{f"{n} responders | delay 0": {"coordination": "dispatcher", "n_responders": n}
                       for n in (2, 4, 8)},
                    **{f"4 responders | delay {d}": {"coordination": "dispatcher", "alarm_delay": d}
                       for d in (10, 20, 30)}},
        "comparisons": [("4 responders | delay 0", x) for x in
                        ("2 responders | delay 0", "8 responders | delay 0", "4 responders | delay 10",
                         "4 responders | delay 20", "4 responders | delay 30")],
    },
    "E4_robot_guidance": {
        "question": "Does sharing how many people were sent to each exit change evacuation time?",
        "configs": {"Local observations": PRESETS["local_guides"]["params"],
                    "Shared counts": PRESETS["shared_guides"]["params"]},
        "comparisons": [("Local observations", "Shared counts")],
        "outcomes": {"peak_queueing": "Peak occupants waiting",
                     "all_out_time_capped": "Steps to everyone out (300 if incomplete)",
                     "evacuated": "People evacuated"},
    },
}
