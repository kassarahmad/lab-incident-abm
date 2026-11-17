"""Replicated experiments with paired statistics.

    python run_experiments.py --runs 30 --jobs 2

For every experiment in simulation_config.EXPERIMENTS, each configuration is
run on the same seeds (common random numbers). Each configuration is compared
with the experiment's reference using

  * Wilcoxon signed-rank test (paired by seed), Holm-Bonferroni corrected
  * matched-pairs rank-biserial correlation r (effect size, -1..1)
  * bootstrap 95% CI of the mean paired difference

Outputs (results/): runs.csv, stats.csv; plots/E*_*.png
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from model import LabIncidentModel
from simulation_config import DEFAULT_SEED, EXPERIMENTS

ROOT = Path(__file__).resolve().parent
OUTCOMES = {"burned_area": "Burned area (cells)",
            "contain_time": "Time to containment (steps, 300 = not contained)",
            "exposure": "Occupant smoke exposure (dose)",
            "response_time": "Alarm to first extinguisher shot (steps)",
            "messages": "Messages sent"}
DEFAULT_PLOTS = {outcome: OUTCOMES[outcome] for outcome in ("burned_area", "contain_time", "exposure")}


def run(job):
    exp, label, params, seed = job
    m = LabIncidentModel(seed=seed, **params)
    while m.running:
        m.step()
    s = m.summary()
    s["contain_time"] = s["contained_step"] if s["contained"] else m.max_steps
    s["all_out_time_capped"] = m.max_steps if s["all_out_step"] is None else s["all_out_step"]
    return {"experiment": exp, "config": label, "seed": seed, **s}


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation (Kerby 2014). Positive = config larger."""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def bootstrap_ci(diff, rng, n=4000):
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def analyse(runs):
    rng = np.random.default_rng(0)
    rows = []
    for exp, spec in EXPERIMENTS.items():
        d = runs[runs.experiment == exp]
        if d.empty:
            continue
        for outcome in spec.get("outcomes", OUTCOMES):
            block = []
            for ref, label in spec["comparisons"]:
                base = d[d.config == ref].set_index("seed")[outcome].astype(float)
                other = d[d.config == label].set_index("seed")[outcome].astype(float)
                diff = (other - base).dropna().to_numpy()
                p = 1.0 if np.all(diff == 0) else stats.wilcoxon(diff, zero_method="wilcox").pvalue
                lo, hi = bootstrap_ci(diff, rng)
                block.append({"experiment": exp, "outcome": outcome, "reference": ref, "config": label,
                              "ref_mean": base.mean(), "config_mean": other.mean(),
                              "mean_diff": diff.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                              "rank_biserial": rank_biserial(diff), "p_wilcoxon": p})
            adj = holm(np.array([b["p_wilcoxon"] for b in block]))
            for b, a in zip(block, adj):
                b["p_holm"] = a
                b["significant"] = a < 0.05
            rows += block
    return pd.DataFrame(rows).round(4)


def plot(runs):
    out = ROOT / "plots"
    out.mkdir(exist_ok=True)
    for exp, spec in EXPERIMENTS.items():
        d = runs[runs.experiment == exp]
        if d.empty:
            continue
        labels = list(spec["configs"])
        refs = {ref for ref, _ in spec["comparisons"]}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, (outcome, title) in zip(axes, spec.get("outcomes", DEFAULT_PLOTS).items()):
            data = [d[d.config == l][outcome].astype(float).to_numpy() for l in labels]
            bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6,
                            medianprops={"color": "black"}, flierprops={"markersize": 3})
            for patch, l in zip(bp["boxes"], labels):
                patch.set_facecolor("#E9C46A" if l in refs else "#A8DADC")
            ax.scatter([x.mean() for x in data], range(1, len(data) + 1), marker="D", s=18,
                       color="#E63946", zorder=3, label="mean")
            ax.set_yticks(range(1, len(labels) + 1))
            ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle(f"{exp}: {spec['question']}  (yellow = reference of a comparison)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{exp}.png", dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*", help="experiment names to run")
    args = ap.parse_args()

    jobs = [(exp, label, params, args.seed + i)
            for exp, spec in EXPERIMENTS.items() if not args.only or exp in args.only
            for label, params in spec["configs"].items() for i in range(args.runs)]
    print(f"{len(jobs)} runs on {args.jobs} processes")
    with Pool(args.jobs) as pool:
        rows = pool.map(run, jobs, chunksize=4)
    runs = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    old = ROOT / "results" / "runs.csv"
    if args.only and old.exists():   # keep the other experiments' runs
        prev = pd.read_csv(old)
        runs = pd.concat([prev[~prev.experiment.isin(args.only)], runs], ignore_index=True)
    runs.to_csv(ROOT / "results" / "runs.csv", index=False)

    stats_df = analyse(runs)
    stats_df.to_csv(ROOT / "results" / "stats.csv", index=False)
    plot(runs[runs.experiment.isin(args.only)] if args.only else runs)

    summary = runs.groupby(["experiment", "config"], sort=False)[
        ["contained", "burned_area", "contain_time", "response_time", "evacuation_time", "exposure", "messages"]].mean().round(2)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary)
        print(stats_df[stats_df.outcome == "burned_area"][
            ["experiment", "config", "mean_diff", "diff_ci_low", "diff_ci_high", "rank_biserial", "p_holm"]])
        guidance = runs[runs.experiment == "E4_robot_guidance"]
        if not guidance.empty:
            print(guidance.groupby("config", sort=False)[
                ["peak_queueing", "all_out_time_capped", "evacuated"]].mean().round(2))


if __name__ == "__main__":
    main()
