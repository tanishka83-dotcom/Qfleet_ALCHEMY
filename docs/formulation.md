# QFleet Phase 2: Mathematical Formulation & Fleet Optimizer

**Problem Statement:** SIH26138 — AI/ML & Quantum-Inspired Optimisation for Maritime Fleet Fuel Efficiency and Emissions.

---

## 1. Problem Overview

The Phase 2 Fleet Optimizer solves the multi-objective maritime fleet dispatch, vessel assignment, route speed, and fuel-selection problem across a network of scheduled trade routes $\mathcal{R}$.

Given a set of scheduled trade routes $\mathcal{R}$ with known distances, cargo demands (in TEU), and delivery deadlines, the optimizer assigns an active vessel $v \in \mathcal{V}$, an operational speed $s \in \mathcal{S}_v$, and an eligible bunker fuel type $f \in \mathcal{F}_v$ to each route $r \in \mathcal{R}$.

> [!IMPORTANT]
> **Surrogate Evaluation via Trained ML Model (Phase 1)**
> The fuel consumption function $\text{FuelConsumed}(r, v, s, f)$ is **NOT** evaluated via the simple cubic physics baseline formula. 
> Instead, it is evaluated directly using the **Phase 1 Physics + XGBoost residual trained surrogate model**, which accounts for non-linear physical interactions, hull fouling degradation over vessel age, engine part-load curves, and class-specific speed exponents.

---

## 2. Mathematical Formulation

### 2.1 Sets and Indices
- $r \in \mathcal{R} = \{1, 2, \dots, R\}$: Set of shipping routes / voyages to be fulfilled ($R = 6$).
- $v \in \mathcal{V} = \{1, 2, \dots, V\}$: Set of available vessel classes in the fleet ($V = 4$: Feeder, Panamax, Post-Panamax, ULCV).
- $s \in \mathcal{S}_v = \{0.60, 0.70, 0.80, 0.90, 1.00\} \times v_{\text{design}}$: Discrete operational speed options (knots).
- $f \in \mathcal{F}_v$: Set of compatible bunker fuel types for vessel $v$ (HFO, VLSFO, MGO, LNG, Methanol).

### 2.2 Parameters & Quotas
- $D_r$: Nautical distance of route $r$ (nm).
- $Q_r$: Cargo demand for route $r$ (TEU).
- $T_r^{\max}$: Delivery deadline for route $r$ (days).
- $C_v^{\text{cap}}$: Nominal container capacity of vessel $v$ (TEU).
- $N_v^{\max}$: **Fleet Allocation Quota** — maximum routes assigned to vessel class $v$ ($\text{Feeder}: 1, \text{Panamax}: 2, \text{PostPanamax}: 2, \text{ULCS}: 2$).
- $P_f$: Bunker fuel price for fuel type $f$ (\$/tonne).
- $I_f^{\text{WTW}}$: Well-to-Wake (WTW) lifecycle GHG intensity factor for fuel $f$ ($\text{g CO}_2\text{eq}/\text{MJ}$).
- $\text{LHV}_f$: Lower heating value of fuel $f$ ($\text{MJ}/\text{kg}$).
- $\text{Cap}_{\text{CO}_2}$: Total allowable fleet lifecycle emissions cap across all routes ($35,000\text{ t }\text{CO}_2\text{eq}$).
- $w_1 = 1.0$: Weight on bunker fuel cost (\$).
- $w_2 = 100.0$: Shadow carbon price (\$/t $\text{CO}_2\text{eq}$, EU ETS proxy).
- $w_3 = 100,000.0$: Heavy late-delivery penalty (\$/day past deadline).

### 2.3 Decision Variables
For discrete formulations (MILP / combinatorial):
$$x_{r, v, s, f} \in \{0, 1\} \quad \forall r \in \mathcal{R}, v \in \mathcal{V}, s \in \mathcal{S}_v, f \in \mathcal{F}_v$$
where $x_{r, v, s, f} = 1$ if route $r$ is serviced by vessel $v$ running at speed $s$ on fuel $f$, and $0$ otherwise.

---

## 3. Objective Function

The objective minimizes total composite fleet operational cost, carbon penalty, and schedule delay:

$$\min \quad J = w_1 \cdot \text{Cost}_{\text{fuel}} + w_2 \cdot \text{Emissions}_{\text{WTW CO}_2} + w_3 \cdot \text{Penalty}_{\text{schedule}} + \sum_{k} \text{Penalty}_k$$

### 3.1 Components
- **Fuel Cost**:
  $$\text{Cost}_{\text{fuel}} = \sum_{r, v, s, f} x_{r, v, s, f} \cdot \mathcal{M}_{\text{fuel}}(r, v, s, f) \cdot P_f$$
- **Lifecycle WTW Emissions**:
  $$\text{Emissions}_{\text{WTW CO}_2} = \sum_{r, v, s, f} x_{r, v, s, f} \cdot \left[ \mathcal{M}_{\text{fuel}}(r, v, s, f) \times 10^3 \times \text{LHV}_f \times I_f^{\text{WTW}} \times 10^{-6} \right]$$
- **Schedule Delay**:
  $$\Delta_r = \max\left(0, \frac{D_r}{24 s} - T_r^{\max}\right), \quad \text{Penalty}_{\text{schedule}} = \sum_{r \in \mathcal{R}} \Delta_r$$

---

## 4. Constraints & Status

1. **Route Assignment (Hard)**:
   $$\sum_{v \in \mathcal{V}} \sum_{s \in \mathcal{S}_v} \sum_{f \in \mathcal{F}_v} x_{r, v, s, f} = 1 \quad \forall r \in \mathcal{R}$$

2. **Cargo Capacity Constraint (Hard)**:
   $$\sum_{s \in \mathcal{S}_v} \sum_{f \in \mathcal{F}_v} x_{r, v, s, f} \cdot C_v^{\text{cap}} \ge Q_r \quad \forall r \in \mathcal{R}$$

3. **Fleet-Size Allocation Quota (Hard)**:
   Couples routes across the fleet:
   $$\sum_{r \in \mathcal{R}} \sum_{s \in \mathcal{S}_v} \sum_{f \in \mathcal{F}_v} x_{r, v, s, f} \le N_v^{\max} \quad \forall v \in \mathcal{V}$$

4. **Fleet Emissions Cap (Non-Binding)**:
   $$\text{Emissions}_{\text{WTW CO}_2} \le 35,000 \text{ t}$$
   *Status:* **Non-Binding**. Optimal fleet lifecycle emissions are $\approx 12,571.30\text{ t}$, well below the $35,000\text{ t}$ cap.

5. **Schedule Feasibility (Hard Deadline)**:
   A solution is defined as feasible if and only if $\text{Delay} = 0.00\text{ days}$, cargo demand is satisfied, quotas are respected, and emissions are within cap.

---

## 5. MILP Piecewise-Linear Grid Formulation

- Exact evaluation over the candidate grid: $6\text{ routes} \times 4\text{ vessels} \times 5\text{ speeds} \times 5\text{ fuels} = 600\text{ binary decision variables}$.
- Solved via `scipy.optimize.milp` using the HiGHS branch-and-cut solver.
