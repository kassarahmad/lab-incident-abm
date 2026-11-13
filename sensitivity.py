"""Global sensitivity analysis (Sobol indices, Saltelli sampling) with SALib.

    python sensitivity.py --n 128 --jobs 2

Each parameter sample is averaged over 4 replicate seeds shared by all samples.
Which parameters drive the burned area and the occupants' exposure, alone (S1)
and through interactions (ST - S1)? Coordination is fixed to Contract Net.
Outputs: results/sobol.csv, plots/sobol.png
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample

from model import LabIncidentModel

ROOT = Path(__file__).resolve().parent
PROBLEM = {
    "num_vars": 6,
    "names": ["p_spread", "n_responders", "alarm_delay", "comm_loss",
              "detector_reliability", "extinguisher_charges"],
    "bounds": [[0.15, 0.35], [2, 8.999], [0, 30.999], [0.0, 0.5], [0.6, 1.0], [5, 15.999]],
}
INTEGER = {"n_responders", "alarm_delay", "extinguisher_charges"}
OUTPUTS = ["burned_area", "exposure"]


REPLICATES = 4


def run(x):
    """Mean output over the same replicate seeds for every parameter sample
    (common random numbers): parameter effects are not confounded with the
    luck of a single run, which would inflate every total-effect index."""
    params = {k: (int(v) if k in INTEGER else float(v)) for k, v in zip(PROBLEM["names"], x)}
    out = []
    for r in range(REPLICATES):
        m = LabIncidentModel(seed=30000 + r, coordination="contract_net", **params)
        while m.running:
            m.step()
        s = m.summary()
        out.append([s[o] for o in OUTPUTS])
    return np.mean(out, axis=0).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64, help="base sample size (power of 2)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--plot-only", action="store_true", help="redraw from results/sobol.csv")
    args = ap.parse_args()
    if args.plot_only:
        n_samples = len(pd.read_csv(ROOT / "results" / "sobol_samples.csv"))
        plot(pd.read_csv(ROOT / "results" / "sobol.csv"),
             f"Sobol indices (Saltelli, N={args.n}, {n_samples} samples x {REPLICATES} CRN replicates, 95% CI)")
        return

    X = sobol_sample.sample(PROBLEM, args.n, calc_second_order=False, seed=42)
    print(f"{len(X)} parameter samples x {REPLICATES} replicates")
    with Pool(args.jobs) as pool:
        Y = np.array(pool.map(run, list(X), chunksize=8), dtype=float)

    rows = []
    for j, out in enumerate(OUTPUTS):
        res = sobol.analyze(PROBLEM, Y[:, j], calc_second_order=False, seed=42)
        for k, name in enumerate(PROBLEM["names"]):
            rows.append({"output": out, "parameter": name,
                         "S1": res["S1"][k], "S1_conf": res["S1_conf"][k],
                         "ST": res["ST"][k], "ST_conf": res["ST_conf"][k]})
    df = pd.DataFrame(rows).round(4)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "sobol.csv", index=False)
    pd.DataFrame(X, columns=PROBLEM["names"]).assign(**{o: Y[:, j] for j, o in enumerate(OUTPUTS)}) \
        .to_csv(ROOT / "results" / "sobol_samples.csv", index=False)
    print(df.to_string(index=False))

    plot(df, f"Sobol indices (Saltelli, N={args.n}, {len(X)} samples x {REPLICATES} CRN replicates, 95% CI)")


def plot(df, title):
    # same parameter order in both panels (by total effect on burned area)
    order = df[df.output == "burned_area"].sort_values("ST")["parameter"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, out in zip(axes, OUTPUTS):
        d = df[df.output == out].set_index("parameter").loc[order]
        y = np.arange(len(d))
        ax.barh(y + 0.2, d["ST"], 0.4, xerr=d["ST_conf"], color="#1D3557", label="Total effect ST")
        ax.barh(y - 0.2, d["S1"], 0.4, xerr=d["S1_conf"], color="#A8DADC", label="First order S1")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_title({"burned_area": "Burned area", "exposure": "Occupant exposure"}[out])
        ax.axvline(0, color="grey", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    fig.savefig(ROOT / "plots" / "sobol.png", dpi=150)


if __name__ == "__main__":
    main()
