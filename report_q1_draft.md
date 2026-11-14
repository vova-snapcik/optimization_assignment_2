# Question 1 - Is the community worth forming? (draft)

Draft of the report section for Question 1, plus an appendix that lists every step taken. Numbers come from
`results/Q1/` (`python run_q1.py`); figures from `python plot_q1.py`. All Q1 models are linear programs solved to
proven optimality in under 0.02 s each, so none of the numbers below is an incumbent or a bound.
Notation of Assignment 1 is kept; only what is new is defined here.

## 0. Answer in five lines

1. **Flexibility the DSO can extract.** With the base contract the community contract $\mathcal C$ removes the whole
   congestion: the aggregated peak falls from 27.6 kW ($\mathcal R$) to 9.4 kW and the exceedance of the cap from 66.3 kWh to 0.
   Individual contracts $\mathcal I$ leave a peak of 11.1 kW and 4.3 kWh of exceedance.
2. **Worth for the community as a whole.** $\mathcal C$ costs 372.2 DKK/day against 398.2 for $\mathcal R$ (gain 26.1) and 480.7 for
   $\sum_i\mathcal I_i$ (gain 108.5). Signing *individually* never pays on the grid tested; signing as a community pays only above a
   discount of about 0.5 at the base cap (below, $\mathcal R$ is cheaper).
3. **Why.** $\mathcal C\le\sum_i\mathcal I_i$ always (pooling), but nothing bounds $\mathcal C$ by $\mathcal R$: the contract
   is worth signing together when the tariff discount outweighs the cost of shifting load, and not otherwise.
4. **Who gains under the dual settlement of $\mathcal C$.** Not everyone: only 3 of 16 members (the flexible PV owners C1_PV, C1_PVB, C2_PVB)
   are better off than in $\mathcal R$; 13 are worse off. Against $\mathcal I$, 8 are worse off. The asset-free and battery-only members lose
   most.
5. **Budget.** The manager does not break even: it collects 163.8 DKK/day more than it pays (the congestion rent
   $\sum_t\mu_t\bar P_t$), 44% of the community's cost. The settlement is not budget balanced and not individually rational,
   which is the problem that Question 2 addresses.

## 1. Models

### 1.1 Setting and notation (new symbols)

| Symbol | Meaning |
|---|---|
| $\bar P^{DSO}_t$ | hourly cap on the import at the connection point (9.39 kW in every hour, base case) |
| $\alpha^{DSO}$, $\beta^{DSO}$ | penalty (DKK/kWh above the cap) and discount on the grid tariffs of the metered exchange (10 and 0.75 in the base case) |
| $\kappa_i$ | share of member $i$ in the cap of model $\mathcal I$, $\sum_i\kappa_i=1$, proportional to its net energy imported in $\mathcal R$ ($\approx0.02$ for PV owners, 0.10 for members without PV) |
| $q^{imp}_{it},q^{exp}_{it}\ge0$ | gross import and export of member $i$ (as in Assignment 1) |
| $P^{imp}_t,P^{exp}_t\ge0$ | aggregated gross import / export of the community at the connection point (model $\mathcal C$) |
| $e_{it}$, $e_t\ge0$ | exceedance of the individual cap (model $\mathcal I$) / of the community cap (model $\mathcal C$) |
| $\lambda_t$, $\mu_t$ | dual of the community balance (internal price) and of the cap, DKK/kWh |
| $\delta^+_{it},\delta^-_{it}$, $g_{it}$, $c_{it}$, $d_{it}$, $s_{it}$ | load deviation up/down, PV production, battery charge, discharge, state of charge |
| $q_{it}=\ell_{it}+c_{it}-d_{it}-g_{it}$ | net consumption of member $i$ |
| $\phi_i=c^L_i\sum_t(\delta^+_{it}+\delta^-_{it})+c^{PV}\sum_tg_{it}$ | private cost of member $i$ (disutility and PV production cost) |

### 1.2 Member constraints (common to $\mathcal R$, $\mathcal I$, $\mathcal C$)

For every member $i$ (the PV and battery variables are absent for members without those assets):

- load $\ell_{it}=\ell^{ref}_{it}+\delta^+_{it}-\delta^-_{it}$, with $\underline\ell_{it}\le\ell_{it}\le\bar\ell_{it}$ (within $\Delta=1$ kWh of the reference and $[0,L^{max}]$) and $\sum_t\ell_{it}\ge E^{min}_i$ (daily minimum = daily reference energy);
- PV: $0\le g_{it}\le PV_{it}$ (curtailable, marginal cost $c^{PV}=1.41$ DKK/kWh);
- battery: $s_{it}=s_{i,t-1}+\eta^{ch}c_{it}-d_{it}/\eta^{dis}$, $0\le s_{it}\le\bar S$, $0\le c_{it}\le\bar P^{ch}$, $0\le d_{it}\le\bar P^{dis}$, initial state $s_{i,0}=0.5\bar S$ and **cyclic end of day** $s_{i,24}=s_{i,0}$ (our choice: no free energy at the end of the horizon; the same as in Assignment 1 question 3).

### 1.3 The three models

**$\mathcal R$ - no contract (full tariffs, own meter, no cap).** Each member solves

$$
\min\; \sum_t\big[p_t(q^{imp}_{it}-q^{exp}_{it})+\tau^{imp}q^{imp}_{it}+\tau^{exp}q^{exp}_{it}\big]+\phi_i
\quad\text{s.t. } q^{imp}_{it}-q^{exp}_{it}=q_{it},\; \text{Section 1.2}.
$$
(The day-ahead price applies to both imports and exports, with opposite signs; the tariffs are paid on both flows. This
is the point where the Assignment 1 code had a sign error, now covered by the validation of Section 3.1.)

**$\mathcal I$ - individual contracts (own meter, discounted tariffs, own cap).** Each member solves

$$
\min\; \sum_t\big[p_t(q^{imp}_{it}-q^{exp}_{it})+(1-\beta)(\tau^{imp}q^{imp}_{it}+\tau^{exp}q^{exp}_{it})+\alpha e_{it}\big]+\phi_i
\quad\text{s.t. } e_{it}\ge q^{imp}_{it}-\kappa_i\bar P^{DSO}_t,\; e_{it}\ge0,\; \text{as in }\mathcal R.
$$
The penalty $\alpha\max(0,q^{imp}_{it}-\kappa_i\bar P^{DSO}_t)$ is convex piecewise linear, hence linearized with $e_{it}$
(exact, since $\alpha>0$ makes $e_{it}$ hit its lower bound).

**$\mathcal C$ - community contract, central dispatch (one meter, one cap).** The manager dispatches all members:

$$
\min\; \sum_t\big[p_t(P^{imp}_t-P^{exp}_t)+(1-\beta)(\tau^{imp}P^{imp}_t+\tau^{exp}P^{exp}_t)+\alpha e_t\big]+\sum_i\phi_i
$$
$$
\text{s.t. } P^{imp}_t-P^{exp}_t=\sum_iq_{it}\;[\lambda_t],\qquad e_t\ge P^{imp}_t-\bar P^{DSO}_t\;[\mu_t],\qquad e_t,P^{imp}_t,P^{exp}_t\ge0,\;\;\text{Section 1.2 for every } i.
$$
This maximizes the community's welfare (payments between members are transfers and cancel), and it is one switch of the same
class as $\mathcal R$ and $\mathcal I$ (`solve(com, mode, beta, alpha, cap_scale, metering)`), reused in Question 2.

*Problem class and consequence for duality.* All three are linear programs (the disutility is linear, the penalty is linearized,
everything else is linear): strong duality holds, the KKT conditions are necessary and sufficient, and the dual variables are
marginal values. The consequences: (i) $\lambda_t$ and $\mu_t$ have an economic meaning (below); (ii) the primal optimum is
**not unique** in general (members with a linear disutility are indifferent at kinks) and the duals are not unique under
degeneracy, so the settlement and aggregate metrics read from one optimal solution are one choice among equals; (iii) there is no
complementarity or integer variable to handle in Question 1 (that appears in Question 2).

*Interpretation of the duals of $\mathcal C$.* From stationarity in $P^{imp}_t$, $P^{exp}_t$ and $e_t$:

$$
\lambda_t=p_t+(1-\beta)\tau^{imp}+\mu_t\;\text{ if the community imports},\qquad
\lambda_t=p_t-(1-\beta)\tau^{exp}\;\text{ if it exports},\qquad 0\le\mu_t\le\alpha,\;\; \mu_t>0\Rightarrow P^{imp}_t=\bar P^{DSO}_t .
$$
So $\lambda_t$ is the marginal cost of one more kWh of net consumption: day-ahead price plus discounted tariff plus,
in congested hours, the cap rent $\mu_t$ (the value of one more kW of cap).

### 1.4 Settlement and metrics

*Dual settlement of $\mathcal C$.* Member $i$ pays $\lambda_tq_{it}$ for its net consumption in each hour (negative = it is paid), plus its
private cost $\phi_i$: $C^{\mathcal C}_i=\sum_t\lambda_tq_{it}+\phi_i$. The manager receives $\sum_t\lambda_t(P^{imp}_t-P^{exp}_t)$ and pays the
connection-point cost $\Gamma$. With $e_t=0$ and $\mu_t>0$ only where $P^{imp}_t=\bar P^{DSO}_t$, the surplus is

$$
\text{surplus}=\sum_t\lambda_t(P^{imp}_t-P^{exp}_t)-\Gamma=\sum_t\mu_tP^{imp}_t=\sum_t\mu_t\bar P^{DSO}_t\;\;(\text{congestion rent}),
$$
and $\sum_iC^{\mathcal C}_i=W^{\mathcal C}+\text{surplus}$ with $W^{\mathcal C}$ the optimal cost of $\mathcal C$. Both identities are checked numerically.

*Metrics (reference situation stated for each).*

| Metric | Definition | Reference |
|---|---|---|
| Community cost | $\sum_i$ (energy + discounted tariff + penalty + $\phi_i$), DKK/day | $\mathcal R$ |
| Value of a contract | $\text{Cost}^{\mathcal R}-\text{Cost}^{X}$, $X\in\{\sum_i\mathcal I_i,\mathcal C\}$ (positive = worth signing) | $\mathcal R$ |
| Value of pooling | $\sum_i\text{Cost}^{\mathcal I_i}-\text{Cost}^{\mathcal C}$ | $\mathcal I$ |
| DSO: exceedance, peak, hours above cap | $\sum_t\max(0,P^{imp}_t-\bar P^{DSO}_t)$, $\max_tP^{imp}_t$, number of hours with exceedance; for $\mathcal R$ and $\mathcal I$, $P^{imp}_t=\sum_iq^{imp}_{it}$ (same definition in all models) | $\mathcal R$ |
| Flexibility extracted | $\sum_{i,t}|\ell_{it}-\ell^{ref}_{it}|$ (kWh shifted), by slope class | $\mathcal R$ (14.6 kWh) |
| Member gain | $C^{\mathcal R}_i-C^{\mathcal C}_i$ and $C^{\mathcal I}_i-C^{\mathcal C}_i$ (IR test: $\ge0$) | $\mathcal R$, $\mathcal I$ |
| Manager's budget | payments collected minus $\Gamma$ | 0 (balanced) |

## 2. Expectations (the results are read against them)

- **E1.** $\mathcal C\le\sum_i\mathcal I_i$ for every contract (pooling is a relaxation of the individual problems), but no such inequality against $\mathcal R$.
- **E2.** Individual contracts force every member to shift (each has a tiny cap $\kappa_i\bar P^{DSO}_t$ and a penalty of 10 DKK/kWh), so $\mathcal I$ may cost more than $\mathcal R$. Pooling lets the cheap flexibility serve everybody.
- **E3.** Under $\mathcal C$ the flexible (low-slope) members shift most and the stiff ones little; the cap rent rises in the hours where the reference peak is highest.
- **E4.** The dual settlement over-collects (rent) and leaves some members worse off than alone.
- **E5.** The contract pays for the community when the discount is large and the cap not too tight; break-even in the discount around the point where tariff savings equal the cost of shifting.

## 3. Numerical results

### 3.1 Validation

- $\mathcal R$ against the reference numbers of `dso_contract_params.json`: cost 398.24 (file 398.24), import 173.4 (173.4), peak 27.6 (27.6); exceedance 66.3 against 66.5 in the file (small difference, see Section 6). The cap is exceeded in 9 hours.
- $\mathcal C$ with no cap, no discount and member-level metering equals $\mathcal R$ exactly (398.2417, difference below $10^{-6}$).
- $\mathcal C\le\sum_i\mathcal I_i$ (372.16 $\le$ 480.67). Member bills equal $W^{\mathcal C}$ plus the surplus (535.93 = 372.16 + 163.77); the surplus equals $\sum_t\mu_t\bar P^{DSO}_t$ (163.77).

### 3.2 Base case: what the contract extracts (day d1, $\beta=0.75$, $\alpha=10$, $f=1.3$)

| | $\mathcal R$ | $\sum_i\mathcal I_i$ | $\mathcal C$ |
|---|---|---|---|
| Community cost (DKK/day) | 398.2 | 480.7 | **372.2** |
| Peak import (kW) | 27.6 | 11.1 | 9.4 |
| Exceedance of the cap (kWh) | 66.3 | 4.3 | 0 |
| Hours above the cap | 9 | 11 | 0 |
| Hours at the cap | 9 | 11 | 18 |
| Energy shifted (kWh) | 14.6 | 117.2 | 111.3 |
| Energy imported (kWh) | 173.4 | 167.5 | 180.1 |

Cost decomposition (DKK/day): in $\mathcal R$ the disutility is 3.7 and the tariffs 93.9; in $\mathcal I$ the **disutility is 107.1 and the penalty
42.9** while the tariffs fall to 22.4 (discount); in $\mathcal C$ the disutility is 47.4 with no penalty. This is why $\mathcal I$ is the most
expensive: every member is pushed to shift, with the penalty as the stick.

*Who shifts (Fig. `fig_q1_shift.png`).* Under $\mathcal I$ every slope class shifts about the same energy (32.2, 28.4, 28.3, 28.3 kWh for
slope classes 1-4), even the stiff class (2 DKK/kWh) that is a poor provider of flexibility. Under $\mathcal C$ the flexibility is called in merit
order of slope: 56.1, 44.4, 10.3 and 0.4 kWh. The cap binds in 18 of 24 hours; the aggregated import follows the cap and
is zero in hours 18-21 when the price is highest (`fig_q1_profile.png`). The community exports 21.9 kWh (18.0 in $\mathcal R$).

*Duals of $\mathcal C$ (`fig_q1_profile.png`, lower panel).* $\lambda_t$ equals the day-ahead price plus the discounted import tariff plus $\mu_t$.
The cap rent is largest in hour 11 ($\mu_{11}=4.19$, $\lambda_{11}=5.41$ DKK/kWh) when the reference peak is 27.6 kW, about 2 to 4.4 times the
day-ahead-plus-tariff price in hours 9-14, and zero from hour 17. In the export hours 18-20 $\lambda_t=p_t-(1-\beta)\tau^{exp}$ (2.0, 2.4, 2.1), as the formula predicts.

### 3.3 Sensitivity study: discount $\beta^{DSO}$ and cap tightness $f$

Design: one study over the two contract terms that decide whether the contract pays, on a grid $\beta\in\{0,0.25,0.5,0.75,1\}$
by $f\in\{0.9,1.0,1.1,1.2,1.3,1.5\}$ (the cap is $f\,E^{\mathcal R}/24$, scaled from the base profile, with the shares $\kappa_i$ fixed);
`fig_q1_map.png` uses a finer grid (9 values of $\beta$, 7 of $f$). Reference: $\mathcal R$ (independent of the contract). Penalty $\alpha=10$ is fixed.

- **Community point of view.** $\mathcal I$ is never cheaper than $\mathcal R$ on the whole grid. $\mathcal C$ is cheaper than $\mathcal R$ above a
  break-even curve: at $f=1.3$ from $\beta\approx0.5$ (396.7 against 398.2); at $f=1.2$ from $\beta=0.75$; at $f=1.1$ only at $\beta=1$; at $f=1.0$ never (409.4
  at $\beta=1$). A tighter cap raises the cost of every contract (e.g. $\mathcal C$ at $\beta=0.75$: 358 at $f=1.5$, 372 at $f=1.3$, 430 at $f=1.0$)
  because more load has to be shifted.
- **DSO point of view.** $\mathcal C$ leaves zero exceedance everywhere; $\mathcal I$ leaves 2.5-9.6 kWh. The peak under $\mathcal C$ equals the cap
  (6.5 kW at $f=0.9$ up to 10.8 kW at $f=1.5$), against 27.6 kW without contract: the DSO's gain does not depend on the discount, only on the cap.
  The discount is a transfer from the DSO to the community that does not change the flexibility extracted (energy shifted is almost flat in $\beta$:
  110.8-114.2 kWh at $f=1.3$).
- **Break-even is where the contract pays for both.** At the base cap the DSO gets full congestion relief for any $\beta$, and the community
  gains from $\beta\gtrsim0.5$; the pairs ($\beta$, $f$) above the curve are the contracts "both sides gain".

### 3.4 Settlement of $\mathcal C$ (base case): individual rationality and the manager's budget

Gain of each member under the dual settlement, in DKK/day (positive = better off than the alternative):

| Slope | gain vs $\mathcal R$: none | PV | battery | PV+bat. | gain vs $\mathcal I$: none | PV | battery | PV+bat. |
|---|---|---|---|---|---|---|---|---|
| L1 (0.25) | -13.6 | +2.1 | -9.3 | +6.5 | -8.4 | +2.9 | -13.2 | +6.3 |
| L2 (0.5) | -16.2 | -0.5 | -12.1 | +3.6 | -8.8 | +1.4 | -13.8 | +4.0 |
| L3 (1.0) | -21.2 | -5.5 | -17.1 | -1.4 | -8.4 | -0.4 | -14.6 | +0.4 |
| L4 (2.0) | -23.2 | -7.5 | -19.1 | -3.4 | +0.2 | +4.0 | -8.3 | +1.4 |

- **Individual rationality.** Against $\mathcal R$: 3 members gain (C1_PV, C1_PVB, C2_PVB), 13 lose. Against $\mathcal I$: 8 lose (C1, C2, C3, C1_B, C2_B, C3_B, C4_B and C3_PV).
  By asset class the total gain against $\mathcal R$ is $-74.1$ (no asset), $-11.4$ (PV), $-57.6$ (battery) and $+5.3$ (PV and battery); by slope class $-14.3$, $-25.1$, $-45.1$, $-53.1$.
- **Budget.** Members' bills total 535.9, the manager pays 179.0 at the connection point and keeps 163.8 (the congestion rent): the
  settlement is **not budget balanced**. The aggregate gain over $\mathcal R$ is positive (26.1) and over $\mathcal I$ is positive (108.5), so lump-sum transfers
  could in principle make every member better off than alone; the dual prices alone do not.
- **Why the types diverge.** Importers without PV pay $\lambda_t$ during the congested hours (up to 5.4 DKK/kWh) and, being mostly stiff, cannot avoid it: the asset-free
  members pay 36.8-49.4 DKK/day for their energy, a total bill of 40.3-50.3 DKK/day including disutility, against 26.7-27.1 alone. PV owners are net sellers in
  several congested hours (e.g. C1_PV in hours 8-9 and 12-14) and are paid $\lambda_t$ (1.9-3.4 DKK/kWh there), so they gain. Battery-only members are better off in $\mathcal I$
  than in $\mathcal R$ (e.g. C1_B 21.7 against 25.5 DKK/day) because the tariff discount makes their arbitrage cheaper. In $\mathcal C$ they still arbitrage
  (C1_B charges at $\lambda_t$ of 1.4-1.9 and discharges at 2.1-5.4), but having no PV they also pay the congested $\lambda_t$ on their net load, so their bill (34.9)
  exceeds both alternatives and they would rather sign alone. The stiff members C4 and C4_PV are about indifferent against $\mathcal I$ (+0.2, +4.0) because under $\mathcal I$ they are forced to shift
  at 2 DKK/kWh, but they prefer not to sign at all ($\mathcal R$: 27.1 and 24.1 DKK against 50.3 and 31.6).

## 4. Recommendations

- **DSO:** the community contract is the effective instrument: it removes all exceedance and cuts the peak by two thirds, for any discount. Individual contracts
  leave exceedance and make members worse off than no contract. A discount of about 0.5 or more (at $f=1.3$) is needed to get the community to sign.
- **Community manager:** form the community, dispatch centrally, but do not charge the dual prices as they are: they over-collect 164 DKK/day and push 13 of 16
  members below their no-contract cost. The rent has to be returned and the allocation designed (Question 2).
- **Members:** PV owners and flexible members gain; asset-free, battery-only and stiff members should not sign unless compensated. A member that does not sign
  should prefer $\mathcal R$ to $\mathcal I$ (individual contracts raise the community cost by 82 DKK/day).

## 5. Discussion

*Why $\mathcal C\le\sum_i\mathcal I_i$.* Take any optimal dispatch of the individual problems and implement it in $\mathcal C$. The energy payments are
identical. The tariff is paid on the aggregated flows $P^{imp}_t=(\sum_iq_{it})^+$, which are no larger than the sum of the gross flows (netting inside the
community pays no tariff). The community exceedance satisfies $\max(0,\sum_ia_i-\sum_ib_i)\le\sum_i\max(0,a_i-b_i)$ because $\sum_i\kappa_i=1$, so the penalty cannot
be larger. $\mathcal C$ can then only improve, hence the bound. Against $\mathcal R$ no dispatch is shared: $\mathcal R$ has no cap and the full tariffs, so the contract can make
the community worse off when moving load costs more than the discount and the avoided penalty save; this is the case at $\beta<0.5$ at the base cap.
*Why a contract can be worth signing together and not alone:* pooling lets the cheap flexibility (low slopes, batteries) serve the whole cap, while $\mathcal I$ makes every member pay
its own shifting and its own penalty.

*Uniqueness.* The cost of $\mathcal C$ is unique (LP), but the dispatch and the duals are not in the hours where members are indifferent; the settlement therefore
depends on which optimal solution the solver returns. This also affects the exceedance of $\mathcal R$ (66.3 against 66.5 kWh in the file).

## 6. Limitations to state in the report

- Only the contract terms $\beta$ and the cap factor $f$ are swept; the penalty $\alpha$ is fixed at 10 DKK/kWh (not swept), and the shares $\kappa_i$ follow the base rule (net energy in $\mathcal R$);
  the other split rules of the contract file (peak, gross energy, hourly, capacity) are not tested.
- The sweep CSV rounds $\beta=0.75$ to 0.8 in its printout; the values are those of 0.75.
- Fairness metrics (dispersion of gains, benefit per kWh of flexibility) are not computed; the analysis is by consumer type (slope and asset class).
- Tariff above the cap, cyclic battery and curtailable PV are modelling choices (Section 1); a different choice (e.g. tariff only below the cap) changes the numbers slightly but not the ordering.
- The exceedance of $\mathcal R$ differs from the contract file by 0.2 kWh (likely tie-breaking among equally optimal member schedules); not verified.

---

## Appendix A - Steps taken to solve the problem

1. **Read the material:** Assignment 2 text (Question 1, the base case and contract terms, tables of sensitivities and metrics), the Assignment 1 feedback (sign error in the objective of the code, no price-ladder comparison with import/export prices, duals plot missing) and the starter data and loader (`InputData`, `ContractData`, the contract file with $E^{\mathcal R}$, cap and shares).
2. **Fixed the modelling choices** (Section 1): tariff on all imported kWh plus penalty (as in the contract file), soft cap, cyclic battery, curtailable PV with marginal cost, load deviation variables $\delta^\pm$.
3. **Built $\mathcal R$, $\mathcal I$, $\mathcal C$ as one parametrized class** (`src/community_model.py`): `mode`, `beta`, `alpha`, `cap_scale`, `metering`. $\mathcal R$ and $\mathcal I$ are solved as one separable LP each (each member's optimum is independent); $\mathcal C$ has the aggregate balance and cap.
4. **Validation:** $\mathcal R$ against cost, import and peak of the contract file; $\mathcal C$ in its limit case (no cap, no discount, member metering) equals $\mathcal R$ to $10^{-6}$; $\mathcal C\le\sum\mathcal I_i$; settlement identities (bills = cost + surplus; surplus = $\sum\mu_t\bar P_t$); dual sign checked on slack and binding hours ($\lambda_t=p_t+(1-\beta)\tau^{imp}+\mu_t$).
5. **Base-case run and settlement** (`run_q1.py`): R/I/C table, cost decomposition, shifted energy by class, dual settlement, IR against $\mathcal R$ and $\mathcal I$ by member, manager's budget.
6. **Sensitivity study:** discount $\times$ cap tightness; the contract terms that decide whether the contract pays, chosen after seeing that $\mathcal C$ crosses $\mathcal R$ in this range.
7. **Figures** (`plot_q1.py`): hourly import against the cap with $\lambda_t$, map of gain over $\mathcal R$ in $(\beta,f)$, gain by member, shifted load by member and hour. Checked each figure visually; fixed overlapping legends and a title that said the stiff members stay put (under $\mathcal I$ they shift too).
8. **Writing this draft:** numbers read from `results/Q1/*.csv` and from re-runs of the models for the cost decomposition and per-class shifted energy (not stored in the CSVs).

## Appendix B - Points to compare with a teammate's Question 1

The results depend on a few modelling choices; differences in the following would explain mismatches: (i) tariff on all imports vs only below the cap; (ii) battery end-of-day state (cyclic here); (iii) PV curtailment with marginal cost $c^{PV}$ in the objective; (iv) the same day-ahead price with opposite signs for imports and exports; (v) which optimal solution the solver returns under indifference (aggregate import profile, duals); (vi) which shares $\kappa_i$ and which cap profile are used. Reference check: $\mathcal R$ cost 398.24, peak 27.6 kW, import 173.4 kWh; $\mathcal I$ 480.67; $\mathcal C$ 372.16; rent 163.77.

Reproduction: `python run_q1.py && python plot_q1.py` (under one minute). Gurobi 13.0.3, default settings; all models LP.
