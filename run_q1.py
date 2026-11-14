"""Question 1 reproduction:  python run_q1.py [--skip-sweep]

1. validation      R against the reference numbers of dso_contract_params.json; C with no cap / no discount / member
                   metering reproduces R; C <= sum_i I_i; settlement identity (sum of member bills = C cost + rent)
2. base case       R, I, C side by side (community and DSO metrics)
3. settlement      dual settlement of C, individual rationality against I and R by member, budget of the manager
4. sensitivity     discount beta x cap tightness (f/1.3 scaling): community cost of R, I, C and DSO metrics
Outputs go to results/Q1/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.community_model import budget, settlement, solve
from src.data_loader import load_community

OUT = Path(__file__).resolve().parent / "results" / "Q1"


def summary_row(sol) -> dict:
    d = sol.dso_metrics()
    d["community_cost_DKK"] = sol.objective
    d["solve_s"] = sol.solve_time
    return d


def validate(com) -> None:
    ref = com.contract.reference_model_R
    R = solve(com, "R")
    print("[validation] R vs file:  cost %.2f (file %.2f) | import %.1f (%.1f) | peak %.1f (%.1f) | exceedance %.1f (%.1f)" % (
        R.objective, ref["cost_DKK"], R.P_imp.sum(), ref["daily_import_kWh"], R.P_imp.max(), ref["peak_import_kW"],
        R.exceedance.sum(), ref["exceedance_kWh"]))
    V = solve(com, "C", beta=0.0, alpha=0.0, cap_scale=1e3, metering="member")
    print("[validation] C(no cap, no discount, member metering) = %.4f  vs R = %.4f" % (V.objective, R.objective))
    assert abs(V.objective - R.objective) < 1e-6
    I, C = solve(com, "I"), solve(com, "C")
    print("[validation] C = %.2f <= sum I = %.2f ; R = %.2f" % (C.objective, I.objective, R.objective))
    assert C.objective <= I.objective + 1e-6
    st = settlement(com, C)
    b = budget(C)
    print("[validation] member bills %.2f = C cost %.2f + surplus %.2f -> %s" % (
        st.cost_C.sum(), C.objective, b["surplus"], "OK" if abs(st.cost_C.sum() - C.objective - b["surplus"]) < 1e-6 else "MISMATCH"))
    print("[validation] manager surplus %.2f = sum_t mu_t cap_t %.2f" % (b["surplus"], b["cap_rent_mu_x_cap"]))


def base_case(com) -> tuple:
    R, I, C = solve(com, "R"), solve(com, "I"), solve(com, "C")
    tab = pd.DataFrame({"R": summary_row(R), "I": summary_row(I), "C": summary_row(C)})
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "base_case_summary.csv")
    print("\n[base case]\n", tab.round(2).to_string())

    # members: own cost in R and I; cost under the dual settlement of C
    cost_R = {c: sum(v.values()) for c, v in R.cost_member.items()}
    cost_I = {c: sum(v.values()) for c, v in I.cost_member.items()}
    st = settlement(com, C)
    df = pd.DataFrame({"archetype": [com.member(c).archetype for c in st.index], "cost_R": pd.Series(cost_R), "cost_I": pd.Series(cost_I),
                       "cost_C_settled": st.cost_C, "payment_lam": st.settlement_payment})
    df["gain_vs_R"] = df.cost_R - df.cost_C_settled
    df["gain_vs_I"] = df.cost_I - df.cost_C_settled
    df["IR_vs_R"] = df.gain_vs_R >= -1e-6
    df["IR_vs_I"] = df.gain_vs_I >= -1e-6
    df.to_csv(OUT / "settlement_by_member.csv")
    print("\n[settlement of C by member] (DKK/day; gain = benchmark cost - cost under settlement)\n", df.round(2).to_string())
    b = budget(C)
    print("\n[manager budget] collected %.1f, connection cost %.1f, surplus (congestion rent) %.1f" % (b["collected"], b["connection_cost"], b["surplus"]))
    print("members worse off than R: %d, than I: %d" % ((~df.IR_vs_R).sum(), (~df.IR_vs_I).sum()))
    pd.DataFrame({"hour": range(24), "lam": C.lam, "mu_cap": C.mu_cap, "P_imp": C.P_imp, "cap": C.cap, "R_import": R.P_imp,
                  "I_import": I.P_imp, "price": com.energy_price}).to_csv(OUT / "hourly_C.csv", index=False)
    return R, I, C


def sweep(com) -> None:
    rows = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        for f in (0.9, 1.0, 1.1, 1.2, 1.3, 1.5):
            sc = f / 1.3
            R = solve(com, "R", cap_scale=sc)
            I = solve(com, "I", beta=beta, cap_scale=sc)
            C = solve(com, "C", beta=beta, cap_scale=sc)
            rows.append({"beta": beta, "f": f, "cost_R": R.objective, "cost_I": I.objective, "cost_C": C.objective,
                         "exc_R": R.exceedance.sum(), "exc_I": I.exceedance.sum(), "exc_C": C.exceedance.sum(),
                         "peak_R": R.P_imp.max(), "peak_I": I.P_imp.max(), "peak_C": C.P_imp.max(),
                         "shift_I": I.dso_metrics()["shifted_kWh"], "shift_C": C.dso_metrics()["shifted_kWh"],
                         "C_beats_R": C.objective < R.objective, "I_beats_R": I.objective < R.objective})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sweep_beta_cap.csv", index=False)
    print("\n[sweep beta x f]\n", df.round(1).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()
    com = load_community("base")
    validate(com)
    base_case(com)
    if not args.skip_sweep:
        sweep(com)


if __name__ == "__main__":
    main()
