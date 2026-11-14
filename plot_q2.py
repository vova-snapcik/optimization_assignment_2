"""Question 2 figures from the CSVs written by run_q2.py:  python plot_q2.py   (writes results/Q2/fig_*.png)

fig_q2_prices.png    posted prices of the three steps against the duals of C and the discounted import price
fig_q2_members.png   gain of every member against its best alternative (min of R and I), by step
fig_q2_sweep.png     discount sweep: efficiency cost of each step and the IR violation that remains
Colours are fixed per step in every figure. Hollow markers / hatched bars = the MILP stopped at the time limit (not proven optimal).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results" / "Q2"
COL = {"C": "#0b6fb8", "DP^cap": "#2a9d8f", "DP^BB": "#e08a00", "DP^BB+IR": "#b5179e", "R": "#6b7280"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})


def fig_prices():
    d = pd.read_csv(OUT / "prices.csv")
    from src.data_loader import load_community
    com = load_community("base")
    imp = com.energy_price + (1 - com.contract.tariff_discount_ratio) * com.import_tariff
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.step(d.hour, d.lambda_C, where="mid", color=COL["C"], lw=3.2, alpha=0.35, label=r"duals $\lambda_t$ of C")
    for k in ("DP^cap", "DP^BB", "DP^BB+IR"):
        ax.step(d.hour, d[f"x_{k}"], where="mid", color=COL[k], lw=1.8, label=f"posted price x, {k}")
    ax.step(d.hour, imp, where="mid", color="k", ls=":", lw=1.2, label="day-ahead + discounted import tariff")
    ax.set_xlabel("Hour")
    ax.set_ylabel("DKK/kWh")
    ax.set_xticks(range(0, 24, 3))
    ax.set_title("Prices follow the duals of C where the cap binds; elsewhere they are only pinned within a band (indifference)",
                 loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.savefig(OUT / "fig_q2_prices.png")
    plt.close(fig)


def fig_members():
    d = pd.read_csv(OUT / "members_by_step.csv", index_col=0)
    steps = ["DP^cap", "DP^BB", "DP^BB+IR"]
    d = d.sort_values("gain_vs_best_DP^cap")
    y = np.arange(len(d))
    h = 0.27
    fig, ax = plt.subplots(figsize=(8, 7))
    for j, k in enumerate(steps):
        ax.barh(y + (1 - j) * h, d[f"gain_vs_best_{k}"], height=h, color=COL[k], label=k)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i} ({a})" for i, a in zip(d.index, d.archetype)], fontsize=8)
    ax.set_xlabel("Gain against the better of R and I (DKK/day; negative = individually irrational)")
    ax.set_title("Budget balance returns the rent and cuts the losers from 13 to 8; the stiff, asset-free members stay below zero\n"
                 "(DP^BB+IR is the best incumbent found, not a proven optimum)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0)
    fig.savefig(OUT / "fig_q2_members.png")
    plt.close(fig)


def fig_sweep():
    d = pd.read_csv(OUT / "sweep_beta.csv")
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(7.5, 8.4), sharex=True)
    for k in ("DP^cap", "DP^BB", "DP^BB+IR"):
        s = d[d.step == k]
        loss = s.cost - s.cost_C
        a.plot(s.beta, loss, color=COL[k], lw=1.8, label=k)
        proven = s.status == "OPTIMAL"
        a.scatter(s.beta[proven], loss[proven], color=COL[k], zorder=3, s=36)
        a.scatter(s.beta[~proven], loss[~proven], facecolors="white", edgecolors=COL[k], zorder=3, s=36, lw=1.5)
    a.set_ylabel("Welfare loss against C (DKK/day)")
    a.set_title("One hourly price recovers C fully; budget balance is free; individual rationality costs about 10 DKK/day\n"
                "(filled = proven optimal, hollow = time limit)", loc="left", fontsize=9.5)
    a.legend(frameon=False, fontsize=9)
    s = d[d.step == "DP^BB+IR"]
    b.plot(s.beta, s.ir_slack_total, color=COL["DP^BB+IR"], lw=1.8, marker="o", mfc="white")
    b.set_ylabel("IR violation left (DKK/day)")
    b.set_title("The IR violation shrinks as the discount grows (best incumbent = upper bound) but never vanishes", loc="left", fontsize=9.5)
    for k in ("DP^cap", "DP^BB", "DP^BB+IR"):
        t = d[d.step == k]
        c.plot(t.beta, t.n_worse_than_best, color=COL[k], lw=1.8, marker="o", label=k)
    c.set_ylabel("Members below benchmark")
    c.set_xlabel(r"tariff discount $\beta^{DSO}$")
    c.legend(frameon=False, fontsize=9)
    c.set_title("Members worse off than the better of R and I", loc="left", fontsize=9.5)
    fig.savefig(OUT / "fig_q2_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_prices()
    fig_members()
    fig_sweep()
    print("wrote", sorted(p.name for p in OUT.glob("fig_q2*.png")))
