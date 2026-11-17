"""Render one run as an animated GIF and a few PNG frames (for the slides).

    python render_run.py --preset baseline --seed 1000
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np

from agents import Occupant, Responder, SmokeDetector
from model import LabIncidentModel
from simulation_config import PRESETS

TILE_COLORS = {"#": 0, ".": 1, "B": 2, "C": 3, "E": 4, "S": 5}
CMAP = ListedColormap(["#2B2D42", "#F8F9FA", "#C9B79C", "#9D4EDD", "#52B788", "#4895EF"])


def draw(ax, m, title):
    ax.clear()
    base = np.vectorize(TILE_COLORS.get)(m.env.tiles).T
    ax.imshow(base, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
    ax.imshow(np.ma.masked_less(m.env.smoke.T, 0.02), cmap="Greys", alpha=0.55, vmin=0, vmax=0.6)
    ax.imshow(np.ma.masked_equal(m.env.fire.T, 0), cmap="autumn_r", vmin=0, vmax=1, alpha=0.95)
    ax.imshow(np.ma.masked_equal(m.queue_wait.T, 0), cmap="summer", alpha=0.6, vmin=0, vmax=16)
    if m.closed_cells:
        ax.scatter(*zip(*m.closed_cells), marker="x", s=40, c="#343943", zorder=3)
    if m.robots:
        ax.scatter(*zip(*(r.location for r in m.robots)), marker="s", s=65,
                   facecolors="none", edgecolors="#7158be", linewidths=1.5, zorder=4)
    det = [a.location for a in m.detectors]
    trig = [a.location for a in m.detectors if a.triggered]
    if det:
        ax.scatter(*zip(*det), marker="s", s=8, c="#ADB5BD", zorder=3)
    if trig:
        ax.scatter(*zip(*trig), marker="s", s=14, c="#FF006E", zorder=3)
    for cls, color, marker, label in [(Occupant, "#1D3557", "o", "Occupant"),
                                      (Responder, "#E63946", "^", "Responder")]:
        pts = [a.pos for a in m.agents if isinstance(a, cls) and a.pos is not None]
        down = [a.pos for a in m.agents if isinstance(a, Occupant) and a.state == "incapacitated"]
        if pts:
            ax.scatter(*zip(*pts), c=color, marker=marker, s=28, edgecolors="white", linewidths=0.5, zorder=4)
        if down:
            ax.scatter(*zip(*down), c="black", marker="x", s=28, zorder=5)
    alarm = "ALARM ON" if m.alarm_on else "no alarm"
    ax.set_title(f"{title} | step {m.steps} | {alarm} | evacuated {m.evacuated}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="dispatcher", choices=PRESETS)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    m = LabIncidentModel(seed=args.seed, **PRESETS[args.preset]["params"])
    title = PRESETS[args.preset]["name"]
    fig, ax = plt.subplots(figsize=(8, 4.6))

    def update(_):
        if m.running and m.steps < args.steps:
            m.step()
        draw(ax, m, title)
        if m.steps in (0, 10, 30, 60):
            fig.savefig(out / f"frame_{args.preset}_{m.steps:03d}.png", dpi=130)

    draw(ax, m, title)
    anim = FuncAnimation(fig, update, frames=args.steps, interval=80)
    anim.save(out / f"run_{args.preset}.gif", writer=PillowWriter(fps=12))
    print(f"Saved plots/run_{args.preset}.gif")


if __name__ == "__main__":
    main()
