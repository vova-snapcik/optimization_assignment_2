# Question 2 - Can price signals achieve what the central dispatch does? (draft)

Draft of the report section for Question 2, plus an appendix that lists every step taken. Numbers come from
`results/Q2/` (`python run_q2.py`); figures from `python plot_q2.py`. **Every number marked (UB) is the best solution
found by a MILP that did not converge, i.e. an upper bound, not a proven optimum** (see Section 6).
Notation of Assignment 1 and Question 1 is kept; only what is new is defined here.

## 0. Answer in four lines

1. **Value recovered.** One hourly price recovers the whole value of $\mathcal C$: $\mathcal{DP}^{cap}$ reproduces the cost
   of $\mathcal C$ exactly (372.16 DKK/day, no cap exceedance, same energy shifted) at every discount tested (proven optimal).
2. **Budget balance is free.** Imposing it costs at most 0.14 DKK/day (proven 0 at $\beta=0.25, 0.9, 1.0$). It removes
   the manager's rent of 164-195 DKK/day, but the rent is recycled mostly to net-exporting members, not to the losers.
3. **Individual rationality cannot be met with one price** (best found, UB: total violation 12.7 DKK/day at the base
   contract, 4.6 at $\beta=1$) and costs the community at least about 10 DKK/day (UB) in welfare.
4. **Blocking members:** the stiff, asset-free consumers (C3, C4, and mostly C2) block at every discount.

## 1. Models

### 1.1 Setting and notation (new symbols)

| Symbol | Meaning |
|---|---|
| $x_t$ | internal price posted by the manager in hour $t$, DKK/kWh, the same for all members, $x_t\in[\underline x,\bar x]=[0,13]$ |
| $\delta^+_{it},\delta^-_{it}\ge0$ | upward / downward deviation of member $i$'s load from $\ell^{ref}_{it}$ (so $\ell_{it}=\ell^{ref}_{it}+\delta^+_{it}-\delta^-_{it}$) |
| $g_{it}, c_{it}, d_{it}, s_{it}$ | PV production, battery charge, discharge, state of charge |
| $q_{it}=\ell_{it}+c_{it}-d_{it}-g_{it}$ | net consumption of member $i$ (positive = buys, negative = sells, both at $x_t$) |
| $\phi_i=\sum_t c^L_i(\delta^+_{it}+\delta^-_{it})+c^{PV}\sum_t g_{it}$ | private cost of member $i$ (disutility and PV cost) |
| $\Gamma$ | manager's connection-point cost $\sum_t [p_t(P^{imp}_t-P^{exp}_t)+(1-\beta)(\tau^{imp}P^{imp}_t+\tau^{exp}P^{exp}_t)+\alpha e_t]$ |
| $\pi_i=\sum_t x_t q_{it}$ | payment of member $i$ to the manager (negative = it is paid) |
| $\hat C_i=\min(C^{\mathcal R}_i,C^{\mathcal I}_i)$ | benchmark cost of member $i$: its better alternative to the community contract |
| $s_i\ge0$ | IR violation of member $i$ (DKK/day) |

### 1.2 Lower level: the member's best response to posted prices

Given $x$, member $i$ solves the linear program (for members without PV or battery the corresponding variables are absent)

$$
\min_{\delta,g,c,d,s\ge0}\; C_i(x)=\sum_t x_t q_{it}+\phi_i
$$
subject to, with the dual variable of each constraint in brackets,

- $\delta^+_{it}\le \bar\ell_{it}-\ell^{ref}_{it}\;[\bar u^+_{it}]$, $\;\delta^-_{it}\le \ell^{ref}_{it}-\underline\ell_{it}\;[\bar u^-_{it}]$ (hourly bounds of Assignment 2);
- $\sum_t(\delta^+_{it}-\delta^-_{it})\ge E^{min}_i-\sum_t\ell^{ref}_{it}\;[\sigma_i]$ (daily minimum energy; the right-hand side is 0 in the base case);
- $g_{it}\le PV_{it}$, $c_{it}\le\bar P^{ch}$, $d_{it}\le\bar P^{dis}$, $s_{it}\le\bar S$ (upper bounds, duals $\ge0$);
- $s_{it}=s_{i,t-1}+\eta^{ch}c_{it}-d_{it}/\eta^{dis}$, with $s_{i,0}$ = initial state and $s_{i,24}=s_{i,0}$ (cyclic end of day; free duals).

**What changes from Assignment 1.** (i) The member buys *and sells* at the same price $x_t$: there is no
import/export spread, no tariff and no cap in its own problem; those are paid by the manager only. (ii) The price
$x_t$ multiplies the *net* consumption, so for a PV or battery owner it is simultaneously the price of buying and the
revenue from selling: it will export whenever $x_t$ exceeds its marginal cost $c^{PV}$ (or the battery's opportunity
value). (iii) Because the objective is linear in $(\delta,g,c,d)$ the response is a *set*, not a point, at certain prices.

**Stationarity (reduced costs).** For a variable $y_j\ge0$ with cost $c_j(x)$, the reduced cost is
$r_j=c_j(x)-\sum_k A_{kj}u_k\ge0$. For the load deviations:

$$
r^{\delta^+}_{it}=c^L_i+x_t+\bar u^+_{it}-\sigma_i\ge0,\qquad
r^{\delta^-}_{it}=c^L_i-x_t+\bar u^-_{it}+\sigma_i\ge0 .
$$

With complementarity ($y_jr_j=0$, $u_k\cdot\text{slack}_k=0$) this reproduces the price ladder of Assignment 1: the
multiplier $\sigma_i$ is member $i$'s internal value of a kWh of its daily need. If $|x_t-\sigma_i|<c^L_i$ the member
keeps its reference in hour $t$; if $x_t<\sigma_i-c^L_i$ it consumes more (up to the upper bound); if $x_t>\sigma_i+c^L_i$
it consumes less (down to the lower bound); at $|x_t-\sigma_i|=c^L_i$ it is **indifferent** and the response is a set
(Section 5). Battery and PV have the analogous conditions (PV: produce if $x_t>c^{PV}$).

### 1.3 Upper level: the manager

The manager chooses $x$ and the community's metered exchange anticipating the members' responses:

$$
\mathcal{DP}:\quad \min_{x,\,P,\,e,\,y}\; W=\Gamma+\sum_i\phi_i
\quad\text{s.t.}\quad P^{imp}_t-P^{exp}_t=\sum_i q_{it},\;\; e_t\ge P^{imp}_t-\bar P^{DSO}_t,\;\; e_t,P^{imp}_t,P^{exp}_t\ge0,\;\; y_i\in\arg\min C_i(x).
$$

*Objective and its implication.* Following the assignment, the manager maximizes the community's welfare, i.e.
minimizes $W$ = connection-point cost plus the members' private costs. Payments $\pi_i$ are transfers inside the
community and cancel in $W$. The implication: **the manager is indifferent between all price vectors that induce the same
dispatch**, so $x$ is in general not unique and its distributional effect (who pays what) is not decided by the
objective. Requirements on the prices (budget balance, IR) are what select among them.

*Solution concept.* Optimistic (strong Stackelberg): when a member's best response is not unique, the member picks the
element the manager prefers. This is the standard single-level MPEC reading and is the one for which $\mathcal C$ can be
reproduced (Section 5 discusses what is lost when it fails).

### 1.4 Single-level MILP (steps of the reformulation)

1. **Eliminate $\ell$:** substitute $\ell_{it}=\ell^{ref}_{it}+\delta^+_{it}-\delta^-_{it}$ so that every variable is $\ge0$ and the member LP is in standard form $\min c(x)^\top y$, $Ay\ge b$ (equalities for the battery).
2. **Replace each member LP by its KKT conditions:** primal feasibility; dual feasibility $r_j(x)=c_j(x)-A_j^\top u\ge0$ (linear in $x$ since $x$ only enters the cost vector); complementarity $y_jr_j=0$ and $u_k(A_ky-b_k)=0$. Each complementarity is an **SOS1 pair** (no big-M constants to tune; bounds $u\le40$, $r\le40$, and variable bounds only tighten the relaxation).
3. **Linearize the payments by strong duality.** The product $x_tq_{it}$ is bilinear, but at the KKT point the member's primal and dual objectives coincide:
   $$\sum_t x_t q_{it}+\phi_i=\sum_t x_t\ell^{ref}_{it}+b_i^\top u_i
   \;\Rightarrow\;
   C_i(x)=\sum_t x_t\ell^{ref}_{it}+b_i^\top u_i,\quad \pi_i=C_i(x)-\phi_i ,$$
   which is linear in $x$, $u$ and the primal variables. This is what allows budget balance and IR to be written as linear constraints.
4. **Tighten (optional, `mccormick=True`):** add $\sum_t w_{it}+\phi_i=C_i(x)$ with $w_{it}$ the McCormick envelope of $x_tq_{it}$ over $[\underline x,\bar x]\times[q^{min}_{it},q^{max}_{it}]$ (valid linear cuts).
5. Result: a **mixed-integer linear program** (SOS1 pairs). Size at the base case (16 members, with the McCormick cuts): 6,856 variables, 4,888 constraints, 3,088 complementarity pairs; it grows linearly in the number of members and hours.

### 1.5 Variants of the pricing problem (one parametrized model `solve_dp`)

| Variant | Added to $\mathcal{DP}$ | Switch |
|---|---|---|
| $\mathcal{DP}^{cap}$ | nothing (cap only, the three-step study starts here) | `bb=False, ir=None` |
| $\mathcal{DP}^{BB}$ | budget balance: $\sum_i\pi_i=\Gamma$ (equivalently $\sum_tx_t(P^{imp}_t-P^{exp}_t)=\Gamma$) | `bb=True` |
| $\mathcal{DP}^{BB+IR}$ | IR: $C_i(x)\le\hat C_i+s_i$, $s_i\ge0$, objective $W+\rho\sum_is_i$, $\rho=1000$ | `bb=True, ir="best"` |

*Choice for infeasibility.* Uniform prices plus BB plus IR can be infeasible, so IR is **soft**: the slack $s_i$ is the
amount (DKK/day) that member $i$ stays below its benchmark. A large $\rho$ makes the model meet IR whenever possible
(near-lexicographic), keeps the problem always feasible, and makes the blocking members and the amount to be given up
directly readable ($s_i>0$). Alternatives considered and not implemented: per-member price discrimination (it abandons
the uniform-price premise of the question) and relaxing BB to $\ge$ (it lets the manager keep a rent).
*IR benchmark:* the better of $\mathcal R$ and $\mathcal I$ per member ($\hat C_i$), since a member can always choose either alternative.

## 2. Hypotheses (stated before the runs)

- **H1.** $\mathcal{DP}^{cap}=\mathcal C$: the duals of $\mathcal C$ are supporting prices (KKT of $\mathcal C$ decomposes member by member, and the only coupling is the balance, whose dual is the uniform price), so one price per hour recovers $\mathcal C$ in the optimistic sense.
- **H2.** Budget balance is cheap: prices only have to lower the collected amount by the rent, and prices have slack in the hours where members are indifferent or at bounds.
- **H3.** IR cannot be met by one price: it requires every member's cost to fall below its own benchmark, but the same price is paid by flexible members (who must be paid to shift) and stiff ones (who just pay). We expected the violation to shrink as the discount grows (more total surplus to share) but not to vanish. *This was refined by the data: the aggregate gain is positive at base ($\sum_i\hat C_i=392.5>W^*=372.2$), so IR is not infeasible for lack of total value, only for lack of a uniform price.*
- **H4.** The blocking members are those whose response is least elastic (large $c^L$, no assets).

## 3. Numerical results

Base case ($\beta=0.75$, $\alpha=10$, cap $f=1.3$, day d1). Costs in DKK/day; $W^*=372.16$, $C^{\mathcal R}=398.24$, $C^{\mathcal I}=480.67$, $\sum_i\hat C_i=392.45$.

| Step | $W$ | loss vs $\mathcal C$ | exceed. (kWh) | shifted (kWh) | manager profit | members below $\hat C_i$ | total IR violation | MIP gap |
|---|---|---|---|---|---|---|---|---|
| $\mathcal C$ (Q1 duals) | 372.16 | 0 | 0 | 111.3 | 163.8 (rent) | 13 | - | 0 |
| $\mathcal{DP}^{cap}$ | 372.16 | 0 | 0 | 111.3 | 171.2 | 13 | 0 | 0 (proven) |
| $\mathcal{DP}^{BB}$ | 372.18 | 0.02 | 0 | 114.2 | 0 | 8 | 0 | 6e-5 |
| $\mathcal{DP}^{BB+IR}$ (UB) | 382.81 | 10.65 | 1.37 | 108.0 | 0 | 7 | 11.1 | 0.97 |
| $\mathcal{DP}^{BB+IR}$, C's loads fixed (UB) | 430.95 | 58.8 | 6.0 | 111.3 | 0 | 12 | 48.9 | 0.17 |

*Discount sweep* (cap and penalty at base, `fig_q2_sweep.png`):

| $\beta^{DSO}$ | 0.25 | 0.5 | 0.6 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| $W^{\mathcal C}$ (= $\mathcal{DP}^{cap}$) | 419.9 | 396.7 | 387.0 | 372.2 | 357.3 | 347.3 |
| $C^{\mathcal R}$ | 398.2 | 398.2 | 398.2 | 398.2 | 398.2 | 398.2 |
| $\mathcal{DP}^{BB}$ loss vs $\mathcal C$ | 0 | 0.14 | 0.10 | 0.02 | 0 | 0 |
| $\mathcal{DP}^{BB+IR}$ loss vs $\mathcal C$ (UB) | 10.5 | 10.9 | 10.7 | 10.5 | 10.4 | 25.7 |
| IR violation left (UB) | 55.5 | 18.1 | 15.0 | 12.7 | 10.2 | 4.6 |
| members below benchmark, $\mathcal{DP}^{BB}$ | 8 | 8 | 8 | 8 | 7 | 7 |

**Reading (by question).**

- *How much of $\mathcal C$ is recovered, and what explains it?* All of it ($\mathcal{DP}^{cap}$ equals $\mathcal C$ in cost, exceedance and shifted energy at every $\beta$). Two properties explain it: the members' problems are linear, so $\mathcal C$'s dual prices make every member's $\mathcal C$-dispatch a best response; and the only coupling between members is the hourly balance, whose dual is a single price. Fig. `fig_q2_prices.png`: where the cap binds (hours 8-15) the posted price coincides with $\lambda_t$ of $\mathcal C$ (up to 5.4 DKK/kWh at hour 11, against about 1.4-1.7 for day-ahead plus discounted tariff); elsewhere several prices induce the same dispatch.
- *Can the same prices redistribute (BB, IR)?* BB: yes, at (near) zero welfare cost. Mechanism (consistent with the data, not separately proven): the community is a net exporter in hours 18-20 (5.3, 12.0, 4.7 kW in $\mathcal C$), and a high price there pays exporters without changing the dispatch (they are already at their limits). The $\mathcal{DP}^{BB}$ price reaches the upper bound 13 in hours 18-19 for this reason, i.e. **the rent is recycled to net exporters (PV and battery owners), not to the importing losers**, which is why the number of members below their benchmark falls only from 13 to 8 (`fig_q2_members.png`). IR: no, not with one price, and the best solutions found (UB) change the dispatch (peak 10.2 kW, 1.4 kWh of exceedance) at a cost of about 10 DKK/day.
- *What does the requirement cost the community?* BB: about nothing; IR: at least about 10.5 DKK/day in the best incumbent, i.e. 2.8% of the community cost and 52% of the aggregate pooling surplus over the better alternatives ($\sum\hat C_i-W^*=20.3$) (UB).
- *Who blocks?* The same group in every incumbent: C3 and C4 at all discounts, C2 at all but $\beta=1$ (then C2_B), plus several of their battery/PV versions. They have the highest slopes (1-2 DKK/kWh) and no assets: they cannot shift, they pay the high congestion price at the peak, and their benchmark ($\mathcal R$, 27.1 DKK) is cheap for them (they face no cap penalty alone in $\mathcal R$). Raising the discount shrinks the violation (55.5 -> 4.6) but does not remove it. When $C$'s loads are imposed the violation is at least about 40 DKK/day (gap 0.17), so the unavoidable cost of IR is mostly the cost of changing the dispatch.
- *Who pays for congestion and who is paid for flexibility?* Under $\mathcal{DP}^{cap}$ the congestion rent (171 DKK/day) accrues to the manager and is paid by the importers (stiff, asset-free members lose 21-23 DKK/day). Under $\mathcal{DP}^{BB}$ importers still pay the congestion price in hours 8-15, but the rent is returned through high prices to exporters: PV+battery owners gain 17-27 DKK/day against their benchmark, the asset-free stiff members still lose about 21-23. Neither single-purpose price structure rewards flexibility specifically; it rewards net exports.

## 4. Recommendations (for the Q2 part)

- **Community manager:** use $\mathcal{DP}^{BB}$: it recovers $\mathcal C$ at no welfare cost and returns the rent; do not promise IR to all members with one uniform price.
- **DSO:** a higher discount helps the community and the manager (the sweep) but does not change the structural result; the cap is met in all three steps except the IR-compliant incumbent (1.4 kWh exceedance).
- **Members:** the stiff asset-free members should expect to be worse off than staying alone under a uniform price; a side payment or personalized (discriminated) prices would be needed to bring them in.
- **On uniform prices:** suitable for efficiency and budget balance, not suitable if every member must be individually better off.

## 5. Discussion: indifference (no implementation required)

*When is the best response not unique?* A member's problem is an LP, and the ladder of Section 1.2 shows that at
$|x_t-\sigma_i|=c^L_i$ (kink of the disutility), at $x_t=c^{PV}$ for PV, or at the battery's value-of-storage ties, the
objective is flat in that variable, so the optimal response is a set (e.g. any load between the reference and a bound).
The prices that make $\mathcal C$'s dispatch a best response sit exactly on such kinks wherever the cap, not a member
bound, sets the price: the cap multiplier is positive only in hours where a marginal member is indifferent.

*Optimistic vs pessimistic manager.* The model above is optimistic: among equally good responses the member picks the
one the manager likes. That is why $\mathcal{DP}^{cap}$ reproduces $\mathcal C$ exactly. A pessimistic manager assumes
the worst response in the set: the guarantee then is only attained in the limit (the supremum of the price perturbations
is not attained), so the exact $\mathcal C$ outcome cannot be guaranteed.

*How it exposes the manager in practice.* (i) A tiny deviation from the supporting price, or any forecasting or
metering error, tips every marginal member to one side: at the cap-binding hours the community could import more than
the cap, and every kWh above the cap costs $\alpha=10$ DKK/kWh. (ii) Members have no incentive to follow the
manager's preferred element; at indifference they are free, and a strategic or heterogeneous community could pick the
other element. (iii) The same non-uniqueness makes the manager's *profit* and the *distribution* arbitrary (we observe the
cap-only profit changing between 164 and 171 DKK/day across equally optimal solutions, and a price at the 13 DKK/kWh
bound in hours 18-19). Remedies: a small margin $\varepsilon$ on the prices that make the wanted response strictly
optimal (cost: the price premium paid to every marginal member), or an explicit tie-breaking rule in the contract.
*Not done:* counting the indifferent member-hours in the solutions (needs the duals $\sigma_i$ printed from the MILP).

## 6. Limitations to state in the report

- $\mathcal{DP}^{cap}$ is solved to proven optimality in about 1 s. $\mathcal{DP}^{BB}$ is proven optimal for $\beta=0.25, 0.9, 1.0$ and otherwise stopped at the time limit (60-120 s) within 0.14 DKK of $\mathcal C$.
- $\mathcal{DP}^{BB+IR}$ is **not solved to optimality** (gap about 0.97). Its cost, violation and blocking members are those of the best incumbent. With $\mathcal C$'s loads fixed the gap is 0.17 and the violation is at least about 40 DKK/day. The loss of about 10.5 DKK/day is nearly constant across $\beta$ because the solver returns the same dispatch (peak 10.2 kW, 1.37 kWh exceedance); at $\beta=1$ the incumbent is poorer (25.7), probably a solver artefact.
- Price range $[0,13]$ is an assumption (excludes negative prices).
- One sensitivity dimension only (discount), as allowed by the assignment; cap tightness and penalty are only swept in Question 1.

---

## Appendix A - Steps taken to solve the problem

1. **Read the material.** Assignment 2 text (Question 2, tables of sensitivities and metrics), the Assignment 1 feedback (main errors: same price for import and export in the objective; results that should have prompted checks; duals not shown), and the starter data and loader.
2. **Question 1 models first** (needed as benchmark and as the base of $\mathcal{DP}$): $\mathcal R$, $\mathcal I$, $\mathcal C$ as one parametrized LP (`src/community_model.py`). Choices stated: tariff on all imports plus penalty, cyclic battery, soft cap, curtailable PV.
3. **Validation of Q1.** $\mathcal R$ reproduces cost 398.24, peak 27.6, import 173.4 of the contract file; $\mathcal C$ with no cap, no discount and member-level metering equals $\mathcal R$; $\mathcal C\le\sum_i\mathcal I_i$; sum of member bills equals $\mathcal C$ cost plus the manager's surplus, and the surplus equals the cap multipliers times the cap.
4. **Derived the member's KKT system** (Section 1.2) and the price ladder with the multiplier of the daily minimum.
5. **Built the manager's MILP** (Section 1.4): KKT with SOS1 pairs, bilinear payments linearized by strong duality, budget balance and soft IR written on top (`src/dp_model.py`).
6. **First verification:** $\mathcal{DP}^{cap}$ gives exactly $\mathcal C$'s cost (372.16) in 1.2 s, and its prices coincide with $\mathcal C$'s duals in the cap-binding hours.
7. **Variants on the base case.** $\mathcal{DP}^{BB}$ and $\mathcal{DP}^{BB+IR}$ solved with a time limit. The first attempt (300 s per solve, output buffered by a pipe) showed no progress and was stopped; I reran with unbuffered logs and a 120 s limit.
8. **Convergence problem of the IR variant (gap about 97%)** and what was tried, in order: (a) tighter variable and dual bounds ($u,r\le40$, bounds on every primal variable, price range $[0,13]$): no improvement; (b) a stage-1 problem that minimizes only the total IR violation (the second stage, welfare with the violation capped, was implemented as `max_total_slack` but not run because stage 1 did not converge): gap 0.996, incumbent 8.6; (c) McCormick envelope of $x_tq_{it}$ tied to strong duality: no improvement (violation 12.7, gap 0.997); (d) solver focus on the bound (`MIPFocus=3`): worse incumbent (12.7); (e) imposing $\mathcal C$'s load schedule (`fix_load`): solves with gap 0.17 and gives a certified-ish bound (violation $\ge$ about 40 DKK/day) but a different statement.
9. **Decision:** report the full $\mathcal{DP}^{BB+IR}$ as best-incumbent upper bounds with their gaps, and the fixed-load result as the evidence about efficient dispatch. (This was agreed in the discussion; no further tightening was attempted.)
10. **Sensitivity study:** tariff discount $\beta\in\{0.25,0.5,0.6,0.75,0.9,1.0\}$ at the base cap, three steps per value, 60 s per solve. Chosen because it moves $\mathcal C$ across break-even against $\mathcal R$ (about 0.5 at the base cap in Q1), where IR is most likely to change.
11. **Figures and documentation:** `plot_q2.py` (prices, members, sweep), `README.md` (one command per experiment, limitations). Checked each figure visually; changed the sweep figure from one mixed axis to three panels.
12. **Check of a surprising result:** the $\mathcal{DP}^{BB}$ price at the 13 DKK/kWh bound in hours 18-19 was compared with $\mathcal C$'s export profile: the community exports in hours 18-20 (5.3, 12.0, 4.7 kW), which supports the explanation in Section 3 (not separately verified on the $\mathcal{DP}^{BB}$ dispatch).

## Appendix B - Solver settings

Gurobi 13.0.3 (academic licence), MIPGap $10^{-5}$ requested; time limits 120 s (base case) and 60 s (sweep); default threads; SOS1 pairs (no manual big-M); bounds $u\le40$, $r\le40$, price in $[0,13]$; $\rho=1000$.
Reproduction: `python run_q2.py && python plot_q2.py` (about 25 min).
