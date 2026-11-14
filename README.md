# 46750 - Assignment 2 (Questions 1 and 2): energy community with a capacity-limitation contract

Code for Assignment 2 of *46750 - Optimization in Modern Power Systems* (DTU). It builds on the starter package
(`data/`, `src/data_loader.py`, `src/scenario_generation.py` are the provided files, unchanged). Question 3 is not
part of this folder yet. Run every command from this folder (`assignment2/`).

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate     # or reuse the repository's .venv
pip install -r requirements.txt                     # gurobipy, numpy, pandas, matplotlib
```
A full Gurobi licence is required (an academic licence works). The restricted licence is too small for Question 2.

## 2. Reproduce every result (one command per experiment)

| Experiment | Command | Writes to `results/` | Time |
|---|---|---|---|
| Q1 validation, base case R / I / C, dual settlement, discount x cap sweep | `python run_q1.py` | `Q1/*.csv` | < 1 min |
| Q1 figures | `python plot_q1.py` | `Q1/fig_q1_*.png` | < 1 min |
| Q2 base case (cap -> budget balance -> IR) and discount sweep | `python run_q2.py` | `Q2/*.csv` | about 25 min |
| Q2 figures (read the CSVs above) | `python plot_q2.py` | `Q2/fig_q2_*.png` | seconds |

`run_q2.py` options: `--skip-sweep`, `--time-limit` (per solve in the sweep, default 60 s), `--base-time-limit`
(per solve in the base case, default 120 s). The MILP time limits are part of the results: see Section 5.

## 3. Files

```
src/community_model.py   Q1: models R, I, C as ONE parametrized class (solve(com, mode, beta, alpha, cap_scale, metering));
                         settlement() and budget() for the dual settlement of C
src/dp_model.py          Q2: dynamic-pricing MILP DP^x as ONE parametrized model (solve_dp(com, bb, ir, ...))
run_q1.py, plot_q1.py    Q1 experiments and figures
run_q2.py, plot_q2.py    Q2 experiments and figures
data/, src/data_loader.py, src/scenario_generation.py     provided starter files (unchanged)
```

## 4. Modelling choices (state them in the report)

Member problem (all models): flexible load within hourly bounds and a daily minimum, linear disutility on the deviation
from the reference, curtailable PV with marginal cost, battery with cyclic end-of-day state of charge (SoC_24 = SoC_0).

* **Contract tariff:** the discounted tariff applies to *all* imported kWh and the penalty is paid on top (wording of
  `dso_contract_params.json`). The cap is soft: exceedance e_t >= P_imp_t - cap_t, e_t >= 0, penalty alpha.
* **R, I:** each member has its own meter; R pays the full tariffs and has no cap; I pays discounted tariffs, the penalty
  on its own exceedance and has the cap kappa_i x cap_t. R and I are solved as one separable LP per model.
* **C:** one meter, hourly balance P_imp - P_exp = sum_i q_it; the internal price lam_t is the dual of that balance
  (marginal cost of one more kWh of community net consumption). Dual settlement: member i pays lam_t q_it.
* **Validation limit case:** `solve(com, "C", beta=0, alpha=0, cap_scale=1e3, metering="member")` reproduces R exactly.
  R also reproduces the cost, import and peak of `dso_contract_params.json` (cost 398.24, peak 27.6, import 173.4;
  exceedance 66.3 against 66.5 in the file - see Section 5).
* **DP (Q2):** optimistic (strong Stackelberg) solution concept. The members' LPs are replaced by their KKT conditions
  (complementarity as SOS1 pairs); the bilinear payment x_t q_it is replaced by strong duality, so payment_i and member
  cost are linear in x and the dual variables. Prices are bounded to [0, 13] DKK/kWh (13 > alpha + maximum price + tariff).
  Variants: `bb=True` budget balance (payments collected = connection-point cost); `ir="best"` individual rationality
  against the better of R and I per member (`"R"`, `"I"` also possible); IR is *soft* (slack s_i >= 0, penalty rho = 1000)
  so the model stays feasible and the blocking members are the ones with s_i > 0. `fix_load` imposes C's load schedule.
* **Metrics:** community cost = connection-point cost + members' private costs (disutility, PV cost); payments are
  transfers. DSO view in every model: aggregated gross import against the cap (exceedance, peak, hours above cap).
  Value recovered by DP = (cost_I - cost_DP) / (cost_I - cost_C).

## 5. Known limitations (read before quoting numbers)

* **MILP gaps.** DP^cap is proven optimal in about 1 s. DP^BB is proven optimal for some discounts (0.25, 0.9, 1.0)
  and within 0.14 DKK of C at the others (time limit, gap <= 6e-5 in the base case). **DP^BB+IR does not converge**: the
  solver stops at the time limit with a gap near 97% (17% when C's loads are fixed). Its cost and total IR violation are
  *upper bounds* from the best incumbent, and the blocking members are the ones in that incumbent. Tightening attempts
  (variable and dual bounds, McCormick strong-duality cuts, solver focus) did not close the gap.
* **Non-unique solutions.** With linear disutility a member's best response is a set. The manager's prices are not
  unique (e.g. the cap-only manager profit varies between runs, 164 to 171 DKK, and the DP^BB price reaches the upper
  bound of 13 DKK/kWh in hours 18-19). Aggregate metrics under indifference can differ between equally optimal solutions;
  this may explain the small exceedance gap of R against the file (66.3 vs 66.5 kWh).
* **IR benchmark** is the better of R and I per member. Sweep: only the discount beta^DSO is varied (cap energy factor
  1.3, penalty 10). Cap and penalty sweeps exist only for Q1 (`run_q1.py`).
