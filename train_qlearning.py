"""Train the responders' option policy with shared SMDP Q-learning.

    python train_qlearning.py --episodes 800

Each episode is a full incident with a random training seed, coordination
mode and number of fires, so the policy must work in every setting.
Training seeds (0..9999) never overlap the evaluation seeds (1000x range is
avoided: training uses seeds >= 100000).

Outputs: policies/q_table.json, results/learning_curve.csv, plots/learning_curve.png
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import LabIncidentModel
from policies import OPTION_NAMES, QLearningPolicy, RulePolicy

ROOT = Path(__file__).resolve().parent
TRAIN_MODES = ["none", "peer", "central_greedy", "contract_net"]
EVAL_SEEDS = list(range(5000, 5012))       # validation seeds, not used for training


def episode(policy, seed, mode, n_fires):
    m = LabIncidentModel(seed=seed, coordination=mode, policy=policy, n_ignitions=n_fires)
    while m.running:
        m.step()
    return m


def evaluate(policy, modes=("none", "contract_net")):
    out = []
    for mode in modes:
        for s in EVAL_SEEDS:
            m = episode(policy, s, mode, 1 + s % 2)
            out.append((m.summary()["burned_area"], m.reward_total))
    a = np.array(out)
    return a[:, 0].mean(), a[:, 1].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    policy = QLearningPolicy(alpha=0.1, gamma=0.95, epsilon=0.3)
    rule_burned, rule_reward = evaluate(RulePolicy())
    print(f"rule baseline on validation: burned={rule_burned:.1f} reward={rule_reward:.1f}", flush=True)

    curve, best = [], None
    for ep in range(1, args.episodes + 1):
        policy.epsilon = max(0.05, 0.3 * (1 - ep / (0.8 * args.episodes)))
        m = episode(policy, 100000 + rng.randrange(10**6), rng.choice(TRAIN_MODES), rng.choice([1, 1, 2]))
        curve.append({"episode": ep, "train_reward": m.reward_total,
                      "train_burned": m.summary()["burned_area"], "epsilon": policy.epsilon})
        if ep % args.eval_every == 0:
            snapshot = policy.greedy()
            burned, reward = evaluate(snapshot)
            curve[-1].update(val_burned=burned, val_reward=reward)
            print(f"episode {ep}: validation burned={burned:.1f} reward={reward:.1f} (eps={policy.epsilon:.2f})", flush=True)
            if best is None or burned < best[0]:
                best = (burned, ep, snapshot)   # model selection on validation seeds

    best_burned, best_ep, policy = best
    print(f"selected Q-table from episode {best_ep} (validation burned={best_burned:.1f})")
    policy.save(ROOT / "policies" / "q_table.json")
    df = pd.DataFrame(curve)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "learning_curve.csv", index=False)

    # learning curve
    (ROOT / "plots").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    roll = df["train_burned"].rolling(50, min_periods=10).mean()
    ax.plot(df["episode"], roll, color="#8D99AE", label="training episodes (rolling mean, with exploration)")
    v = df.dropna(subset=["val_burned"])
    ax.plot(v["episode"], v["val_burned"], "o-", color="#E63946", label="learned policy, validation (greedy)")
    ax.axhline(rule_burned, color="#1D3557", ls="--", label="rule-based policy, validation")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Burned area (cells)")
    ax.set_title("SMDP Q-learning of the responders' option policy")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "plots" / "learning_curve.png", dpi=150)

    # what was learned: preferred option per state, summarised
    q = policy.q
    visited = policy.visits.sum(-1) > 0
    best = q.argmax(-1)
    counts = {OPTION_NAMES[i]: int(((best == i) & visited).sum()) for i in range(len(OPTION_NAMES))}
    print("preferred option in visited states:", counts)


if __name__ == "__main__":
    main()
