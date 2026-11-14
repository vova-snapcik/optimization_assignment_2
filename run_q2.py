"""Question 2 reproduction:  python run_q2.py [--skip-sweep] [--time-limit 300]

1. base case   DP^cap -> DP^BB -> DP^BB+IR  against R, I, C (Q1): community cost, value recovered, flexibility, budget,
               members below their benchmark (blocking members) and the posted prices against the duals of C
2. sensitivity tariff discount beta^DSO (cap energy factor fixed at the base value): feasibility of each step, efficiency
               cost and blocking members
Outputs go to results/Q2/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.community_model import solve
from src.data_loader import load_community
from src.dp_model import solve_dp

OUT = Path(__file__).resolve().parent / "results" / "Q2"
STEPS = [("DP^cap", dict()), ("DP^BB", dict(bb=True)), ("DP^BB+IR", dict(bb=True, ir="best"))]


def member_costs(sol):
    return np.array([sum(sol.cost_member[c].values()) for c in sol.members])


def row(r, C, bench_I, bench_R, com) -> dict:
    shift = float(np.abs(r.load - r.ref).sum())
    best = np.minimum(bench_I, bench_R)
    gain = best - r.member_cost
    return {"status": r.status, "cost": r.cost, "loss_vs_C": r.cost - C.objective,
            "recovered_vs_I": (bench_I.sum() - r.cost) / (bench_I.sum() - C.objective) if bench_I.sum() > C.objective else np.nan,
            "exceedance_kWh": float(r.exceedance.sum()), "peak_kW": float(r.P_imp.max()), "shifted_kWh": shift,
            "manager_profit": r.budget, "ir_slack_total": float(r.slack.sum()),
            "n_worse_than_best": int((gain < -1e-4).sum()), "worst_member_loss": float(max(0.0, -gain.min())),
            "blockers": ",".join(m for m, g in zip(r.members, gain) if g < -1e-4), "gap": r.mip_gap, "solve_s": r.solve_time}


def base_case(com, tl):
    OUT.mkdir(parents=True, exist_ok=True)
    R, I, C = solve(com, "R"), solve(com, "I"), solve(com, "C")
    cR, cI = member_costs(R), member_costs(I)
    rows, res = {}, {}
    for name, kw in STEPS:
        r = solve_dp(com, time_limit=tl, **kw)
        res[name] = r
        rows[name] = row(r, C, cI, cR, com)
        print(name, r.status, round(r.cost, 2), flush=True)
    fx = solve_dp(com, bb=True, ir="best", objective="slack", fix_load=res["DP^cap"].load, time_limit=tl, name="DP^BB+IR (C loads)")
    res[fx.variant] = fx
    rows[fx.variant] = row(fx, C, cI, cR, com)
    print(fx.variant, fx.status, round(fx.cost, 2), "slack", round(float(fx.slack.sum()), 2), "gap", round(fx.mip_gap, 3), flush=True)
    rows["C (Q1 duals)"] = {"status": "OPTIMAL", "cost": C.objective, "loss_vs_C": 0.0, "exceedance_kWh": float(C.exceedance.sum()),
                            "peak_kW": float(C.P_imp.max()), "shifted_kWh": float(np.abs(C.load - C.ref).sum())}
    tab = pd.DataFrame(rows).T
    tab.to_csv(OUT / "base_case_steps.csv")
    print("\n[base case]\n", tab.to_string(), flush=True)
    mem = pd.DataFrame({"archetype": [com.member(c).archetype for c in R.members], "cost_R": cR, "cost_I": cI})
    for name, r in res.items():
        mem[f"cost_{name}"] = r.member_cost
        mem[f"gain_vs_best_{name}"] = np.minimum(cR, cI) - r.member_cost
    mem.index = R.members
    mem.to_csv(OUT / "members_by_step.csv")
    pr = pd.DataFrame({"hour": range(24), "lambda_C": C.lam, "mu_cap": C.mu_cap})
    for name, r in res.items():
        pr[f"x_{name}"] = r.x
    pr.to_csv(OUT / "prices.csv", index=False)
    return res, (R, I, C)


def sweep_beta(com, tl, betas=(0.25, 0.5, 0.6, 0.75, 0.9, 1.0)):
    rows = []
    for b in betas:
        R, I, C = solve(com, "R", beta=b), solve(com, "I", beta=b), solve(com, "C", beta=b)
        cR, cI = member_costs(R), member_costs(I)
        for name, kw in STEPS:
            r = solve_dp(com, beta=b, time_limit=tl, **kw)
            d = row(r, C, cI, cR, com)
            d.update(beta=b, step=name, cost_C=C.objective, cost_R=R.objective, cost_I=I.objective)
            rows.append(d)
            print(f"beta={b} {name}: {r.status} cost {r.cost:.1f} (C {C.objective:.1f}, R {R.objective:.1f}) slack {r.slack.sum():.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep_beta.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60, help="per solve in the sweep (s)")
    ap.add_argument("--base-time-limit", type=float, default=120, help="per solve in the base case (s)")
    a = ap.parse_args()
    com = load_community("base")
    base_case(com, a.base_time_limit)
    if not a.skip_sweep:
        sweep_beta(com, a.time_limit)


if __name__ == "__main__":
    main()
