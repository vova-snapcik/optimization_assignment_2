"""Question 1 figures:  python plot_q1.py   (writes results/Q1/fig_*.png)

fig_q1_profile.png   hourly import against the cap under R, I, C + internal price of C
fig_q1_map.png       when is a contract worth signing? gain over R of I and of C over (discount, cap energy factor)
fig_q1_members.png   who gains under the dual settlement of C? gain vs R and vs I by member
fig_q1_shift.png     who shifts, and when? load - reference by member and hour (I and C)
Colours are fixed per model in every figure: R grey, I orange, C blue (colour-blind safe).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.community_model import budget, settlement, solve
from src.data_loader import load_community

OUT = Path(__file__).resolve().parent / "results" / "Q1"
COL = {"R": "#6b7280", "I": "#e08a00", "C": "#0b6fb8"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})
H = np.arange(24)


def fig_profile(com, R, I, C):
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    cap = C.cap
    a.fill_between(H, 0, 1, where=C.P_imp >= cap - 1e-6, transform=a.get_xaxis_transform(), color=COL["C"], alpha=0.07, step="mid", lw=0)
    for s, k in ((R, "R"), (I, "I"), (C, "C")):
        a.step(H, s.P_imp, where="mid", color=COL[k], lw=2, label=f"{k}: peak {s.P_imp.max():.1f} kW, exceedance {s.exceedance.sum():.1f} kWh")
    a.step(H, cap, where="mid", color="k", ls="--", lw=1.2, label=f"cap {cap[0]:.2f} kW")
    a.set_ylabel("Aggregated import (kW)")
    a.set_ylim(top=38)
    a.set_title("Only the community contract C keeps the import under the cap (shaded: hours where C binds)", loc="left", fontsize=10.5)
    a.legend(frameon=False, loc="upper right", fontsize=9)
    disc = com.energy_price + (1 - com.contract.tariff_discount_ratio) * com.import_tariff
    b.step(H, C.lam, where="mid", color=COL["C"], lw=2, label=r"internal price $\lambda_t$ of C")
    b.step(H, disc, where="mid", color="k", ls=":", lw=1.2, label="day-ahead price + discounted import tariff")
    b.set_ylabel("DKK/kWh")
    b.set_xlabel("Hour")
    b.set_title(r"The gap between $\lambda_t$ and the import price is the cap rent $\mu_t$", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=8.5, loc="upper left")
    b.set_xticks(range(0, 24, 3))
    fig.savefig(OUT / "fig_q1_profile.png")
    plt.close(fig)


def fig_map(com):
    betas, fs = np.linspace(0, 1, 9), np.round(np.arange(0.9, 1.51, 0.1), 2)
    gI, gC = np.zeros((len(fs), len(betas))), np.zeros((len(fs), len(betas)))
    for j, f in enumerate(fs):
        sc = f / 1.3
        cR = solve(com, "R").objective
        for i, b in enumerate(betas):
            gI[j, i] = cR - solve(com, "I", beta=b, cap_scale=sc).objective
            gC[j, i] = cR - solve(com, "C", beta=b, cap_scale=sc).objective
    np.savetxt(OUT / "map_gain_I.csv", gI, delimiter=",")
    np.savetxt(OUT / "map_gain_C.csv", gC, delimiter=",")
    lim = max(abs(gI).max(), abs(gC).max())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, g, name in zip(axs, (gI, gC), ("I", "C")):
        im = ax.pcolormesh(betas, fs, g, cmap="RdBu", norm=norm, shading="nearest")
        ax.contour(betas, fs, g, levels=[0], colors="k", linewidths=1.5)
        ax.plot(0.75, 1.3, marker="o", mfc="none", mec="k", ms=11)
        ax.set_title(f"{name}: gain over R" + (" (never positive)" if (g <= 0).all() else ""), loc="left", fontsize=10.5, color=COL[name])
        ax.set_xlabel(r"tariff discount $\beta^{DSO}$")
        ax.grid(False)
    axs[0].set_ylabel("cap energy factor $f$ (1.3 = base)")
    fig.colorbar(im, ax=axs, label="Gain over R (DKK/day); blue = worth signing", shrink=0.9)
    fig.suptitle("A contract pays only for a generous discount, and pooling (C) lowers the discount needed; circle = base case", x=0.01, ha="left", fontsize=10.5)
    fig.savefig(OUT / "fig_q1_map.png")
    plt.close(fig)


def fig_members(com, R, I, C):
    st = settlement(com, C)
    cR = np.array([sum(R.cost_member[c].values()) for c in st.index])
    cI = np.array([sum(I.cost_member[c].values()) for c in st.index])
    gR, gI = cR - st.cost_C.values, cI - st.cost_C.values
    order = np.argsort(gR)
    lab = [f"{st.index[k]} ({com.member(st.index[k]).archetype})" for k in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 6.2))
    ax.barh(y + 0.2, gR[order], height=0.38, color=COL["R"], label="vs R (not signing)")
    ax.barh(y - 0.2, gI[order], height=0.38, color=COL["I"], label="vs I (signing alone)")
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8.5)
    ax.set_xlabel("Gain from the dual settlement of C (DKK/day, positive = better off)")
    b = budget(C)
    ax.set_title(f"Under the dual settlement {int((gR < -1e-6).sum())} of 16 members are worse off than in R;\n"
                 f"the manager keeps a rent of {b['surplus']:.0f} DKK/day (R is {R.objective:.0f}, C total {C.objective:.0f})",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0)
    fig.savefig(OUT / "fig_q1_members.png")
    plt.close(fig)


def fig_shift(com, I, C):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    lim = max(np.abs(I.load - I.ref).max(), np.abs(C.load - C.ref).max())
    for ax, s, k in zip(axs, (I, C), ("I", "C")):
        im = ax.imshow(s.load - s.ref, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{k}: load minus reference", loc="left", color=COL[k], fontsize=10.5)
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(False)
    axs[0].set_yticks(range(16))
    axs[0].set_yticklabels([f"{m.consumer_id}" for m in com.members], fontsize=8)
    fig.colorbar(im, ax=axs, label="kWh/h (red = consumes more, blue = less)", shrink=0.9)
    fig.suptitle("Under C the low-slope members (C1, C2) do the shifting; under I every member shifts, even the stiff C4 (at a high cost)", x=0.01, ha="left", fontsize=10.5)
    fig.savefig(OUT / "fig_q1_shift.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    com = load_community("base")
    R, I, C = solve(com, "R"), solve(com, "I"), solve(com, "C")
    fig_profile(com, R, I, C)
    fig_members(com, R, I, C)
    fig_shift(com, I, C)
    fig_map(com)
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.png")))


if __name__ == "__main__":
    main()
