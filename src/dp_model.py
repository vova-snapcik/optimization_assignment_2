"""Question 2: the dynamic-pricing model DP^x as ONE parametrized MILP.

Bilevel structure (optimistic / strong-Stackelberg solution concept)
    upper level  manager chooses hourly prices x_t (same for all members) and, anticipating the members' reactions, the
                 community's exchange P_imp, P_exp, e  (P_imp - P_exp = sum_i q_it, e_t >= P_imp_t - cap_t)
                 objective: community cost = connection-point cost + members' private costs (payments are transfers)
    lower level  member i: min  sum_t x_t q_it + c^L_i sum_t (dp_it + dm_it) + c^PV g_it       (an LP, q buys AND sells at x_t)
                 s.t. 0<=dp<=hi-ref, 0<=dm<=ref-lo, sum(dp-dm) >= Emin - sum(ref), PV, battery (cyclic)
                 (l = ref + dp - dm is eliminated; q = l + ch - dis - g)

Single-level MILP
    1. lower level replaced by its KKT conditions: primal feasibility, dual feasibility (reduced costs r_j = c_j(x) - A_j'u >= 0),
       complementarity y_j r_j = 0 and u_k s_k = 0, written as SOS1 pairs (no big-M needed; bounds on u, r only tighten);
    2. the bilinear payment x_t q_it is replaced by strong duality: member i's total cost  sum_t x_t q_it + private_i
       = sum_t x_t ref_t + b'u_i  (linear in x and the duals);  hence  payment_i = sum_t x_t ref_t + b'u_i - private_i.
Variants (switches):  bb (budget balance: sum_i payment_i = connection-point cost),  ir in {None,"R","I","best"} (member cost
under DP <= its cost in the benchmark; "best" = min of R and I), ir_soft (slack s_i >= 0 penalised by rho, so that the model stays
feasible and the members that cannot be satisfied are identified by s_i > 0).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .community_model import solve as solve_base
from .data_loader import CommunityData


@dataclass
class DPResult:
    variant: str
    status: str
    cost: float                      # community cost = connection cost + private costs (the welfare measure, DKK/day)
    x: np.ndarray                    # posted prices
    members: list[str]
    load: np.ndarray
    ref: np.ndarray
    q: np.ndarray
    P_imp: np.ndarray
    P_exp: np.ndarray
    cap: np.ndarray
    member_cost: np.ndarray          # cost under DP (payment + private)
    payment: np.ndarray
    private: np.ndarray
    bench: np.ndarray | None
    slack: np.ndarray                # IR violation (DKK/day), 0 if met
    conn_cost: float
    mip_gap: float
    solve_time: float
    n_bin_pairs: int
    n_vars: int
    n_cons: int
    extra: dict = field(default_factory=dict)

    @property
    def budget(self) -> float:
        """Manager's profit = payments collected - connection-point cost."""
        return float(self.payment.sum() - self.conn_cost)

    @property
    def exceedance(self) -> np.ndarray:
        return np.maximum(self.P_imp - self.cap, 0.0)


def benchmark_costs(com: CommunityData, which: str, beta=None, alpha=None, cap_scale=1.0) -> np.ndarray:
    R = solve_base(com, "R", beta, alpha, cap_scale)
    I = solve_base(com, "I", beta, alpha, cap_scale)
    cR = np.array([sum(R.cost_member[c].values()) for c in R.members])
    cI = np.array([sum(I.cost_member[c].values()) for c in I.members])
    return {"R": cR, "I": cI, "best": np.minimum(cR, cI)}[which]


def solve_dp(com: CommunityData, bb: bool = False, ir: str | None = None, ir_soft: bool = True, rho: float = 1e3,
             beta: float | None = None, alpha: float | None = None, cap_scale: float = 1.0,
             x_bounds: tuple[float, float] = (0.0, 13.0), objective: str = "welfare", max_total_slack: float | None = None, time_limit: float = 600, mip_gap: float = 1e-5,
             threads: int = 0, mip_focus: int = 0, mccormick: bool = True, fix_load: np.ndarray | None = None, verbose: bool = False, name: str | None = None) -> DPResult:
    k = com.contract
    beta = k.tariff_discount_ratio if beta is None else beta
    alpha = k.penalty_DKK_per_kWh if alpha is None else alpha
    cap = cap_scale * np.asarray(k.capacity_limit_kW, float)
    p, ti, te, disc = com.energy_price, com.import_tariff, com.export_tariff, 1 - beta
    mem = com.members
    N, T = range(len(mem)), range(24)
    bench = benchmark_costs(com, ir, beta, alpha, cap_scale) if ir else None

    m = gp.Model("DP")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.Threads = threads
    m.Params.MIPFocus = mip_focus
    X = m.addVars(T, lb=x_bounds[0], ub=x_bounds[1], name="x")
    Pimp = m.addVars(T, lb=0, name="Pimp")
    Pexp = m.addVars(T, lb=0, name="Pexp")
    E = m.addVars(T, lb=0, name="exc")

    Y: dict = {}            # (i, name, t) -> primal var
    q_expr, priv, dual_obj, const_x = {}, {}, {}, {}
    npairs = 0
    BIGU = 40.0

    for i, d in enumerate(mem):
        ref, lo, hi = d.reference_load, d.load_lower_kWh, d.load_upper_kWh
        c0: dict = {}                       # cost of var (excluding x)
        cx: dict = {}                       # coefficient of x_t in the cost of var
        y: dict = {}
        def newvar(nm, t, ub, c, coef):
            v = m.addVar(lb=0, ub=GRB.INFINITY if ub is None else ub, name=f"{nm}_{i}_{t}")
            y[(nm, t)] = v
            c0[(nm, t)] = c
            cx[(nm, t)] = coef
            return v
        for t in T:
            newvar("dp", t, float(hi[t] - ref[t]), d.linear_disutility, +1)
            newvar("dm", t, float(ref[t] - lo[t]), d.linear_disutility, -1)
            if d.has_pv:
                newvar("g", t, float(d.pv_available[t]), d.pv_marginal_cost, -1)
            if d.has_battery:
                newvar("ch", t, float(d.battery_max_charge_kW), 0.0, +1)
                newvar("dis", t, float(d.battery_max_discharge_kW), 0.0, -1)
                newvar("soc", t, float(d.battery_capacity_kWh), 0.0, 0)
        if fix_load is not None:            # members' flexible-load schedule imposed (e.g. that of C): dp, dm fixed
            for t in T:
                dev = float(fix_load[i][t] - ref[t])
                y[("dp", t)].lb = y[("dp", t)].ub = max(dev, 0.0)
                y[("dm", t)].lb = y[("dm", t)].ub = max(-dev, 0.0)
        # constraints: (coeffs {key: a}, sense, rhs)
        rows = []
        for t in T:
            rows.append(({("dp", t): -1.0}, ">=", -(hi[t] - ref[t])))
            rows.append(({("dm", t): -1.0}, ">=", -(ref[t] - lo[t])))
            if d.has_pv:
                rows.append(({("g", t): -1.0}, ">=", -d.pv_available[t]))
            if d.has_battery:
                rows.append(({("ch", t): -1.0}, ">=", -d.battery_max_charge_kW))
                rows.append(({("dis", t): -1.0}, ">=", -d.battery_max_discharge_kW))
                rows.append(({("soc", t): -1.0}, ">=", -d.battery_capacity_kWh))
                eq = {("soc", t): 1.0, ("ch", t): -d.battery_charging_efficiency, ("dis", t): 1.0 / d.battery_discharging_efficiency}
                if t > 0:
                    eq[("soc", t - 1)] = -1.0
                rows.append((eq, "=", d.battery_initial_soc_kWh if t == 0 else 0.0))
        rows.append(({**{("dp", t): 1.0 for t in T}, **{("dm", t): -1.0 for t in T}}, ">=", d.min_daily_energy_kWh - float(ref.sum())))
        if d.has_battery:
            rows.append(({("soc", 23): 1.0}, "=", d.battery_initial_soc_kWh))

        # primal feasibility, multipliers, complementarity on constraints
        u_list, colsum = [], {key: gp.LinExpr() for key in y}
        for kk, (coef, sense, rhs) in enumerate(rows):
            lhs = gp.quicksum(a * y[key] for key, a in coef.items())
            if sense == "=":
                u = m.addVar(lb=-BIGU, ub=BIGU, name=f"u_{i}_{kk}")
                m.addConstr(lhs == rhs)
            else:
                u = m.addVar(lb=0, ub=BIGU, name=f"u_{i}_{kk}")
                s = m.addVar(lb=0, name=f"s_{i}_{kk}")
                m.addConstr(lhs - s == rhs)
                m.addSOS(GRB.SOS_TYPE1, [u, s], [1, 2])
                npairs += 1
            u_list.append((u, rhs))
            for key, a in coef.items():
                colsum[key] += a * u
        # dual feasibility + complementarity on variables
        for key, v in y.items():
            t = key[1]
            r = m.addVar(lb=0, ub=BIGU, name=f"r_{i}_{key[0]}_{t}")
            m.addConstr(r == c0[key] + cx[key] * X[t] - colsum[key])
            m.addSOS(GRB.SOS_TYPE1, [v, r], [1, 2])
            npairs += 1
        priv[i] = gp.quicksum(c0[key] * v for key, v in y.items())
        dual_obj[i] = gp.quicksum(rhs * u for u, rhs in u_list)
        const_x[i] = gp.quicksum(X[t] * ref[t] for t in T)
        q_expr[i] = {t: ref[t] + y[("dp", t)] - y[("dm", t)]
                     + (y[("ch", t)] - y[("dis", t)] if d.has_battery else 0)
                     - (y[("g", t)] if d.has_pv else 0) for t in T}
        Y[i] = y

    # upper level
    for t in T:
        m.addConstr(Pimp[t] - Pexp[t] == gp.quicksum(q_expr[i][t] for i in N))
        m.addConstr(E[t] >= Pimp[t] - cap[t])
    conn = gp.quicksum(p[t] * (Pimp[t] - Pexp[t]) + disc * (ti * Pimp[t] + te * Pexp[t]) + alpha * E[t] for t in T)
    member_cost = {i: const_x[i] + dual_obj[i] for i in N}                 # strong duality
    if mccormick:
        # valid tightening: strong duality  sum_t x_t q_it + private_i = const_x_i + b'u_i  with the bilinear x_t q_it
        # replaced by its McCormick envelope (x_t and q_it are bounded); cuts off relaxations whose dual value lies below the primal value
        xl, xu = x_bounds
        for i, d in enumerate(mem):
            ql = d.load_lower_kWh - (d.battery_max_discharge_kW if d.has_battery else 0.0) - (d.pv_available if d.has_pv else 0.0)
            qu = d.load_upper_kWh + (d.battery_max_charge_kW if d.has_battery else 0.0)
            ws = []
            for t in T:
                w = m.addVar(lb=-GRB.INFINITY, name=f"w_{i}_{t}")
                q = q_expr[i][t]
                m.addConstr(w >= xl * q + X[t] * ql[t] - xl * ql[t])
                m.addConstr(w >= xu * q + X[t] * qu[t] - xu * qu[t])
                m.addConstr(w <= xu * q + X[t] * ql[t] - xu * ql[t])
                m.addConstr(w <= xl * q + X[t] * qu[t] - xl * qu[t])
                ws.append(w)
            m.addConstr(gp.quicksum(ws) + priv[i] == member_cost[i])
    payment = {i: member_cost[i] - priv[i] for i in N}
    if bb:
        m.addConstr(gp.quicksum(payment[i] for i in N) == conn)
    slack = {}
    if ir:
        for i in N:
            slack[i] = m.addVar(lb=0, ub=(GRB.INFINITY if ir_soft else 0.0), name=f"irslack_{i}")
            m.addConstr(member_cost[i] <= float(bench[i]) + slack[i])
    welfare_cost = conn + gp.quicksum(priv[i] for i in N)
    if ir and max_total_slack is not None:
        m.addConstr(gp.quicksum(slack.values()) <= max_total_slack)
    if ir and objective == "slack":            # stage 1: smallest total IR violation
        m.setObjective(gp.quicksum(slack.values()) + 1e-4 * welfare_cost, GRB.MINIMIZE)
    else:                                       # welfare, IR slack penalised by rho (or capped by max_total_slack)
        m.setObjective(welfare_cost + (rho * gp.quicksum(slack.values()) if ir and max_total_slack is None else 0), GRB.MINIMIZE)

    t0 = time.time()
    m.optimize()
    wall = time.time() - t0
    label = name or ("DP" + ("^{cap}" if not bb and not ir else "^{" + "+".join(([ "BB"] if bb else []) + ([f"IR-{ir}"] if ir else [])) + "}"))
    if m.SolCount == 0:
        status = {GRB.INFEASIBLE: "INFEASIBLE", GRB.TIME_LIMIT: "TIME_LIMIT_NO_SOL", GRB.INF_OR_UNBD: "INF_OR_UNBD"}.get(m.Status, str(m.Status))
        return DPResult(label, status, np.nan, np.full(24, np.nan), [d.consumer_id for d in mem], np.zeros((16, 24)), np.zeros((16, 24)),
                        np.zeros((16, 24)), np.zeros(24), np.zeros(24), cap, np.zeros(16), np.zeros(16), np.zeros(16), bench,
                        np.zeros(16), np.nan, np.nan, wall, npairs, m.NumVars, m.NumConstrs)
    ev = lambda e: e.getValue() if hasattr(e, "getValue") else float(e)
    Q = np.array([[ev(q_expr[i][t]) for t in T] for i in N])
    ref = np.array([d.reference_load for d in mem])
    load = np.array([[ref[i, t] + Y[i][("dp", t)].X - Y[i][("dm", t)].X for t in T] for i in N])
    mc = np.array([ev(member_cost[i]) for i in N])
    pay = np.array([ev(payment[i]) for i in N])
    prv = np.array([ev(priv[i]) for i in N])
    xs = np.array([X[t].X for t in T])
    conn_val = ev(conn)
    return DPResult(label, "OPTIMAL" if m.Status == GRB.OPTIMAL else f"SUBOPT({m.Status})", ev(welfare_cost), xs, [d.consumer_id for d in mem],
                    load, ref, Q, np.array([Pimp[t].X for t in T]), np.array([Pexp[t].X for t in T]), cap, mc, pay, prv, bench,
                    np.array([slack[i].X for i in N]) if ir else np.zeros(len(mem)), conn_val, m.MIPGap, wall, npairs, m.NumVars, m.NumConstrs)
