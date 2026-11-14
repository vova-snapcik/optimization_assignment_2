"""Question 1: the models R, I and C as ONE parametrized class.

    R  no contract           every member alone, full tariffs on its own flows, no cap, no penalty
    I  individual contracts  every member alone, discounted tariffs on its own flows, own cap kappa_i * cap_t,
                             penalty alpha on its own exceedance
    C  community contract    one manager dispatches all members; one meter; discounted tariffs on the aggregated
                             P_imp / P_exp; one cap and one penalty on the aggregate; hourly balance behind the meter

Member problem (identical in R, I, C; all linear)
    flexible load   l_lo <= l_t <= l_hi,  sum_t l_t >= E_min,  disutility c_L * (dp_t + dm_t),  l_t - ref_t = dp_t - dm_t
    PV              0 <= g_t <= PV_t,  marginal cost c_PV (curtailable)
    battery         soc_t = soc_{t-1} + eta_ch ch_t - dis_t / eta_dis,  0 <= soc <= S, cyclic: soc_T = soc_0
    net consumption q_t = l_t + ch_t - dis_t - g_t

Modelling choices (stated in the report):
    * the tariff of the contract applies to ALL imported kWh (also above the cap); the penalty is paid on top
      (wording of dso_contract_params.json);
    * the end-of-day battery state equals the initial one (cyclic), as in Assignment 1 question 3;
    * the cap is soft: exceedance e_t >= P_imp_t - cap_t, e_t >= 0, penalised at alpha.

Switches: ``mode`` ("R" | "I" | "C"), ``beta``, ``alpha``, ``cap_scale`` (multiplies the cap profile and the individual caps),
``metering`` ("community" | "member", only for C: "member" charges tariffs on every member's own flows, used as
validation limit case: C with no cap, no discount and member metering must reproduce R).

Dual convention (C): ``lam_t`` is the dual of the hourly balance written as  P_imp - P_exp - sum_i q_it = 0,
i.e. the marginal cost (DKK/kWh) of one more kWh of net consumption of the community in hour t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData, InputData


@dataclass
class Solution:
    mode: str
    params: dict
    objective: float                       # DKK, the community cost that the model minimises
    members: list[str]
    load: np.ndarray                       # [i, t]
    ref: np.ndarray
    pv: np.ndarray
    ch: np.ndarray
    dis: np.ndarray
    soc: np.ndarray
    q: np.ndarray                          # net consumption of member i (kWh/h), + = buys
    q_imp: np.ndarray                      # gross import of the member (R, I) or net positive part (C)
    q_exp: np.ndarray
    P_imp: np.ndarray                      # aggregate at the connection point (R, I: sum of the members' gross flows)
    P_exp: np.ndarray
    cap: np.ndarray
    cost_member: dict[str, dict[str, float]]   # components per member (own-cost view, see below)
    lam: np.ndarray | None = None          # internal price (C), DKK/kWh
    mu_cap: np.ndarray | None = None       # multiplier of the cap (C), DKK/kWh
    solve_time: float = 0.0
    n_vars: int = 0
    n_cons: int = 0
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ DSO view (same definition in every model)
    @property
    def exceedance(self) -> np.ndarray:
        return np.maximum(self.P_imp - self.cap, 0.0)

    def dso_metrics(self) -> dict[str, float]:
        return {
            "import_kWh": float(self.P_imp.sum()),
            "peak_kW": float(self.P_imp.max()),
            "exceedance_kWh": float(self.exceedance.sum()),
            "max_exceedance_kW": float(self.exceedance.max()),
            "hours_above_cap": int((self.exceedance > 1e-6).sum()),
            "hours_binding": int((self.P_imp >= self.cap - 1e-6).sum()),
            "shifted_kWh": float(np.abs(self.load - self.ref).sum()),
        }

    def total_cost(self) -> float:
        return float(sum(sum(c.values()) for c in self.cost_member.values()))


def _caps(com: CommunityData, cap_scale: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    k = com.contract
    return cap_scale * np.asarray(k.capacity_limit_kW, float), {i: cap_scale * np.asarray(v, float) for i, v in k.individual_capacity_limit_kW.items()}


def solve(com: CommunityData, mode: str, beta: float | None = None, alpha: float | None = None, cap_scale: float = 1.0,
          metering: str = "community", verbose: bool = False) -> Solution:
    """Build and solve R, I or C. ``beta`` / ``alpha`` default to the contract file; R ignores them."""
    assert mode in ("R", "I", "C") and metering in ("community", "member")
    k = com.contract
    beta = k.tariff_discount_ratio if beta is None else beta
    alpha = k.penalty_DKK_per_kWh if alpha is None else alpha
    if mode == "R":
        beta, alpha = 0.0, 0.0
    cap, cap_i = _caps(com, cap_scale)

    mem: list[InputData] = com.members
    ids = [m.consumer_id for m in mem]
    N, T = range(len(mem)), range(24)
    p = com.energy_price
    t_imp, t_exp = com.import_tariff, com.export_tariff
    disc = 1.0 - beta

    m = gp.Model(f"{mode}")
    m.Params.OutputFlag = 1 if verbose else 0

    load = m.addVars(N, T, lb=-GRB.INFINITY, name="load")
    dev_p = m.addVars(N, T, lb=0.0, name="dev_p")
    dev_m = m.addVars(N, T, lb=0.0, name="dev_m")
    pv = m.addVars(N, T, lb=0.0, name="pv")
    ch = m.addVars(N, T, lb=0.0, name="ch")
    dis = m.addVars(N, T, lb=0.0, name="dis")
    soc = m.addVars(N, T, lb=0.0, name="soc")

    q = {}
    for i, d in enumerate(mem):
        for t in T:
            m.addConstr(load[i, t] >= d.load_lower_kWh[t])
            m.addConstr(load[i, t] <= d.load_upper_kWh[t])
            m.addConstr(load[i, t] - d.reference_load[t] == dev_p[i, t] - dev_m[i, t])
            m.addConstr(pv[i, t] <= d.pv_available[t])
            if d.has_battery:
                prev = d.battery_initial_soc_kWh if t == 0 else soc[i, t - 1]
                m.addConstr(soc[i, t] == prev + d.battery_charging_efficiency * ch[i, t] - dis[i, t] / d.battery_discharging_efficiency)
                m.addConstr(soc[i, t] <= d.battery_capacity_kWh)
                m.addConstr(ch[i, t] <= d.battery_max_charge_kW)
                m.addConstr(dis[i, t] <= d.battery_max_discharge_kW)
            else:
                m.addConstr(ch[i, t] == 0)
                m.addConstr(dis[i, t] == 0)
                m.addConstr(soc[i, t] == 0)
            q[i, t] = load[i, t] + ch[i, t] - dis[i, t] - pv[i, t]
        m.addConstr(gp.quicksum(load[i, t] for t in T) >= d.min_daily_energy_kWh)
        if d.has_battery:
            m.addConstr(soc[i, 23] == d.battery_initial_soc_kWh)          # cyclic end of day

    # member-private cost (disutility + PV production cost), common to the three models
    own = {i: gp.quicksum(mem[i].linear_disutility * (dev_p[i, t] + dev_m[i, t]) for t in T) for i in N}
    pvc = {i: gp.quicksum(mem[i].pv_marginal_cost * pv[i, t] for t in T) for i in N}

    metered_by_member = mode in ("R", "I") or metering == "member"
    qi = qe = e_i = P_imp = P_exp = e = bal = capc = None
    if metered_by_member:
        qi = m.addVars(N, T, lb=0.0, name="qimp")
        qe = m.addVars(N, T, lb=0.0, name="qexp")
        for i in N:
            for t in T:
                m.addConstr(qi[i, t] - qe[i, t] == q[i, t])
        energy = {i: gp.quicksum(p[t] * (qi[i, t] - qe[i, t]) for t in T) for i in N}
        tariff = {i: gp.quicksum(disc * (t_imp * qi[i, t] + t_exp * qe[i, t]) for t in T) for i in N}
        penalty = {i: gp.LinExpr() for i in N}
        if mode == "I":
            e_i = m.addVars(N, T, lb=0.0, name="exc")
            for i, d in enumerate(mem):
                for t in T:
                    m.addConstr(e_i[i, t] >= qi[i, t] - cap_i[d.consumer_id][t])
            penalty = {i: gp.quicksum(alpha * e_i[i, t] for t in T) for i in N}
        if mode == "C":                      # limit-case validation: aggregate cap/penalty on top of member metering
            e = m.addVars(T, lb=0.0, name="exc")
            for t in T:
                m.addConstr(e[t] >= gp.quicksum(qi[i, t] for i in N) - cap[t])
            penalty[0] = gp.quicksum(alpha * e[t] for t in T)
        total = gp.quicksum(energy[i] + tariff[i] + penalty[i] + own[i] + pvc[i] for i in N)
    else:
        P_imp = m.addVars(T, lb=0.0, name="Pimp")
        P_exp = m.addVars(T, lb=0.0, name="Pexp")
        e = m.addVars(T, lb=0.0, name="exc")
        bal = {t: m.addConstr(P_imp[t] - P_exp[t] - gp.quicksum(q[i, t] for i in N) == 0, name=f"bal{t}") for t in T}
        capc = {t: m.addConstr(e[t] - P_imp[t] >= -cap[t], name=f"cap{t}") for t in T}
        total = gp.quicksum(p[t] * (P_imp[t] - P_exp[t]) + disc * (t_imp * P_imp[t] + t_exp * P_exp[t]) + alpha * e[t] for t in T) \
            + gp.quicksum(own[i] + pvc[i] for i in N)

    m.setObjective(total, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{mode}: status {m.Status}")

    def arr(v):
        return np.array([[v[i, t].X for t in T] for i in N])

    L, G, CH, DIS, SOC = arr(load), arr(pv), arr(ch), arr(dis), arr(soc)
    ref = np.array([d.reference_load for d in mem])
    Q = L + CH - DIS - G
    if metered_by_member:
        QI, QE = arr(qi), arr(qe)
        PI, PE = QI.sum(0), QE.sum(0)
    else:
        QI, QE = np.maximum(Q, 0), np.maximum(-Q, 0)         # members' own net position (not metered)
        PI, PE = np.array([P_imp[t].X for t in T]), np.array([P_exp[t].X for t in T])

    cost_member = {}
    for i, d in enumerate(mem):
        dis_i = float((mem[i].linear_disutility * np.abs(L[i] - ref[i])).sum())
        pv_i = float(mem[i].pv_marginal_cost * G[i].sum())
        c = {"disutility": dis_i, "pv_cost": pv_i}
        if metered_by_member:
            c["energy"] = float((p * (QI[i] - QE[i])).sum())
            c["tariff"] = float((disc * (t_imp * QI[i] + t_exp * QE[i])).sum())
            c["penalty"] = float(alpha * np.maximum(QI[i] - cap_i[d.consumer_id], 0).sum()) if mode == "I" else 0.0
        cost_member[d.consumer_id] = c

    sol = Solution(mode=mode, params=dict(beta=beta, alpha=alpha, cap_scale=cap_scale, metering=metering),
                   objective=m.ObjVal, members=ids, load=L, ref=ref, pv=G, ch=CH, dis=DIS, soc=SOC, q=Q, q_imp=QI, q_exp=QE,
                   P_imp=PI, P_exp=PE, cap=cap, cost_member=cost_member, solve_time=m.Runtime,
                   n_vars=m.NumVars, n_cons=m.NumConstrs)
    if mode == "C" and not metered_by_member:
        sol.lam = np.array([bal[t].Pi for t in T])
        sol.mu_cap = np.array([capc[t].Pi for t in T])       # >= 0: value of one more kWh of cap in hour t
        sol.extra["penalty_DKK"] = float(alpha * sol.exceedance.sum())
        sol.extra["connection_cost_DKK"] = float((p * (PI - PE)).sum() + (disc * (t_imp * PI + t_exp * PE)).sum() + alpha * sol.exceedance.sum())
    return sol


def settlement(com: CommunityData, sol_C: Solution) -> pd.DataFrame:
    """Dual settlement of model C: member i pays lam_t * q_it (net consumption, + buys / - sells), plus its private costs.

    Returns one row per member with the cost under the settlement, and the manager's budget (sum of what members pay
    minus the connection-point cost) in ``sol.extra``-free form via ``budget``.
    """
    lam = sol_C.lam
    rows = []
    for i, cid in enumerate(sol_C.members):
        c = sol_C.cost_member[cid]
        pay = float((lam * sol_C.q[i]).sum())
        rows.append({"member": cid, "settlement_payment": pay, "private_cost": c["disutility"] + c["pv_cost"],
                     "cost_C": pay + c["disutility"] + c["pv_cost"]})
    df = pd.DataFrame(rows).set_index("member")
    return df


def budget(sol_C: Solution) -> dict[str, float]:
    """Manager's budget: collected from members at lam minus the payments to the DSO / market (= congestion rent >= 0 if lam sets the price)."""
    collected = float((sol_C.lam * (sol_C.P_imp - sol_C.P_exp)).sum())
    cost = sol_C.extra["connection_cost_DKK"]
    return {"collected": collected, "connection_cost": cost, "surplus": collected - cost,
            "cap_rent_mu_x_cap": float((sol_C.mu_cap * sol_C.cap).sum())}
