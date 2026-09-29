# QFleet Phase 1: Maritime Fuel Consumption Prediction & Quantum-Inspired Feature Selection

**Problem Statement:** SIH26138 — AI/ML & Quantum-Inspired Optimisation for Maritime Fleet Fuel Efficiency and Emissions.

---

> [!WARNING]
> **SYNTHETIC DATA DISCLAIMER**
> All voyage data in this phase is generated using a parameterised physics model with controlled stochastic noise and deliberate non-linearities (class-specific speed exponents, hull fouling factors, engine part-load curves, and wave interaction effects).
> **Prediction results and model evaluation metrics reflect the generator's synthetic dynamics and are not reflective of real-world fleet performance.**

---

## Architecture & Project Structure

```
qfleet/
├── config.py                  # Single source of truth: constants, vessel specs, fuels, routes, TODO_VERIFY
├── db.py                      # SQLite ORM via SQLAlchemy (vessels, fuels, routes, metrics)
├── data/
│   ├── generate_data.py       # Synthetic voyage generator (5,000+ voyages -> parquet)
│   └── voyages.parquet        # Generated dataset (with DATA_IS_SYNTHETIC metadata)
├── models/
│   ├── fuel_model.py          # 5 Prediction Models (OLS, LR, RF, XGBoost, Physics+XGBoost)
│   └── qi_feature_select.py   # Quantum-Inspired Evolutionary Algorithm (QI-EA) for feature & hyperparameter tuning
├── tests/
│   └── test_phase1.py         # Pytest suite testing DB, data generation, reproducibility & models
├── results/
│   ├── prediction_metrics.csv # 5-model test evaluation metrics
│   └── qi_comparison.csv      # 5-seed benchmark comparing Plain XGBoost, Random Search, and QI-EA
└── README.md                  # Project documentation and TODO_VERIFY registry
```

---

## Execution Instructions

All commands use the existing Python virtual environment (`.venv`):

### 1. Initialize & Seed Database
```powershell
.venv\Scripts\python.exe db.py
```

### 2. Generate Synthetic Voyage Data
```powershell
.venv\Scripts\python.exe data\generate_data.py
```

### 3. Train & Evaluate Fuel Models
```powershell
.venv\Scripts\python.exe models\fuel_model.py
```

### 4. Run Quantum-Inspired Feature Selection Benchmark
```powershell
.venv\Scripts\python.exe models\qi_feature_select.py
```

### 5. Run Test Suite
```powershell
.venv\Scripts\python.exe -m pytest tests\test_phase1.py -v
```

---

## Models & Methodology

### 1. Physics Baseline & Residual Learning
The physics baseline calculates VLSFO-equivalent fuel consumption according to the standard cubic law:
$$\text{duration} = \frac{\text{distance}}{\text{speed} \times 24}$$
$$\text{VLSFO\_equiv} = k \times \left(\frac{\text{speed}}{\text{design\_speed}}\right)^3 \times \text{duration} \times \text{load\_factor} \times \text{weather\_factor}$$
$$\text{fuel\_mass} = \text{VLSFO\_equiv} \times \left(\frac{\text{LHV}_{\text{VLSFO}}}{\text{LHV}_{\text{fuel}}}\right) \times \frac{1}{\text{engine\_eff\_ratio}}$$

The **Physics + XGBoost** hybrid model computes this analytic baseline and trains an XGBoost model specifically on the residuals to capture higher-order physical dynamics (hull fouling, speed exponent variations, part-load efficiency).

### 2. Quantum-Inspired Evolutionary Algorithm (QI-EA)
- **Representation:** 11 Qubit angles $\theta \in [0, \pi/2]$ ($6$ for feature groups, $5$ for XGBoost hyperparameters).
- **Gate Update:** Quantum rotation gate based on Narayanan & Moore (1996).
- **Evaluation:** Strict cross-validation MAE on training split only; final evaluation on untouched $20\%$ test split across 5 independent random seeds with equal evaluation budgets.

---

## Verified Sources & TODO_VERIFY Registry

Constants are sourced where official regulatory standards exist (IMO GHG Studies, FuelEU Maritime Regulation (EU) 2023/1805, ISO 8217). All unverified or proxy values are flagged with `TODO_VERIFY`:

| Item ID | Parameter / Constant | Current Value / Proxy | Reason / Action Needed |
| :--- | :--- | :--- | :--- |
| **TV-01** | LHV_BIO_METHANOL | 19.9 MJ/kg | Verify specific bio-feedstock LHV against FuelEU Annex II |
| **TV-02** | LHV_E_METHANOL | 19.9 MJ/kg | Verify e-fuel synthesis standard LHV |
| **TV-03** | LHV_E_DIESEL | 42.7 MJ/kg | Verify Fischer-Tropsch e-diesel LHV against EN 15940 |
| **TV-04** | CO2_WTW_BIO_METHANOL | 15.0 gCO2eq/MJ | Default RED II proxy; requires RED II certified pathway |
| **TV-05** | CO2_WTW_E_METHANOL | 5.0 gCO2eq/MJ | Renewable electricity upstream footprint dependent |
| **TV-06** | CO2_WTW_E_DIESEL | 6.0 gCO2eq/MJ | Direct air capture + green hydrogen pathway dependent |
| **TV-07** | ENG_EFF_LNG_OTTO | 0.98 | Lean-burn Otto vs Diesel cycle engine efficiency |
| **TV-08** | ENG_EFF_BIO_METHANOL | 0.96 | Dual-fuel 2-stroke pilot ignition efficiency |
| **TV-09** | ENG_EFF_E_METHANOL | 0.96 | Dual-fuel 2-stroke pilot ignition efficiency |
| **TV-10** | ENG_EFF_E_DIESEL | 1.00 | Assumed drop-in equivalent to conventional MGO |
| **TV-11** | SPEC_FEEDER | ref_daily_fuel=32 t/d | Proxy based on 1000-2000 TEU average; verify with sea trials |
| **TV-12** | SPEC_PANAMAX | ref_daily_fuel=65 t/d | Proxy based on 4000-5000 TEU average; verify with noon reports |
| **TV-13** | SPEC_POST_PANAMAX | ref_daily_fuel=115 t/d | Proxy based on 8000-10000 TEU average; verify with noon reports |
| **TV-14** | SPEC_ULCV | ref_daily_fuel=165 t/d | Proxy based on 18000-24000 TEU average; verify with noon reports |
| **TV-15** | EXP_FEEDER | speed_exp=2.85 | Synthetic generator hidden speed exponent |
| **TV-16** | EXP_PANAMAX | speed_exp=3.10 | Synthetic generator hidden speed exponent |
| **TV-17** | EXP_POST_PANAMAX | speed_exp=2.95 | Synthetic generator hidden speed exponent |
| **TV-18** | EXP_ULCV | speed_exp=3.20 | Synthetic generator hidden speed exponent |
| **TV-19** | ROUTE_SHANGHAI_ROTTERDAM | 10,500 nm | Verify actual waypoint distance via Suez Canal |
| **TV-20** | ROUTE_SINGAPORE_ROTTERDAM| 8,300 nm | Verify actual waypoint distance via Suez Canal |
| **TV-21** | ROUTE_BUSAN_LA | 5,200 nm | Verify great-circle vs rhumb-line nautical distance |
| **TV-22** | ROUTE_ROTTERDAM_NY | 3,400 nm | Verify North Atlantic track nautical distance |
| **TV-23** | ROUTE_JEBEL_ALI_SINGAPORE| 3,600 nm | Verify Arabian Sea / Malacca Strait nautical distance |
| **TV-24** | ROUTE_SHANGHAI_LA | 5,800 nm | Verify trans-Pacific container shipping lane distance |
| **TV-25** | GEN_FOULING_RATE | 0.015 / year | Verify empirical hull degradation coefficient |
| **TV-26** | GEN_PART_LOAD_COEFF | 0.25 | Verify SFOC bathtub curve part-load penalty |
| **TV-27** | GEN_WEATHER_SPEED_K | 0.40 | Verify wave resistance vs speed non-linear coupling |
| **TV-28** | GEN_MEASUREMENT_NOISE | 0.03 | Bunker measurement uncertainty std-dev |
