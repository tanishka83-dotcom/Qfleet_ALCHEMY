"""
QFleet Phase 1 — Configuration and Constants
=============================================
SIH Problem ID: SIH26138
Maritime fuel prediction — container ships.

POLICY
------
  Every constant below carries a # SOURCE: comment citing the primary
  regulatory or engineering document from which the value was drawn.
  Any value that could not be independently verified against an authoritative
  primary source is prefixed with # TODO_VERIFY and is also listed in
  TODO_VERIFY_ITEMS at the bottom of this file.

UNIT CONVENTIONS (used throughout this codebase)
-------------------------------------------------
  LHV                  : MJ / kg_fuel
  TTW emission factor  : g CO2  / g fuel         (tank-to-wake, combustion)
  WTW emission factor  : g CO2eq / MJ            (well-to-wake, life-cycle)
  Design / op. speed   : knots (kn)
  Distance             : nautical miles (nm)
  Fuel mass            : metric tonnes (t)
  k (ref daily cons.)  : t VLSFO / day  at design speed, design load
  Duration             : hours (h) and days (day = h/24)
  Time                 : UTC datetime unless noted
"""

from __future__ import annotations
import pathlib

# ---------------------------------------------------------------------------
# 1.  Physical constants
# ---------------------------------------------------------------------------

SEAWATER_DENSITY_KG_M3: float = 1025.0
# SOURCE: ISO 15016:2015 — Ships and marine technology — Guidelines for the
#         assessment of speed and power performance by analysis of speed trial
#         data, Clause 4.2. Standard reference seawater density.

GRAVITY_M_S2: float = 9.80665
# SOURCE: BIPM SI Brochure, 9th edition (2019), p. 154.
#         Standard acceleration of free fall.

ADMIRALTY_SPEED_EXPONENT: float = 3.0
# SOURCE: Admiralty Coefficient method; see Carlton, J. (2007) "Marine
#         Propellers and Propulsion", 2nd ed., Butterworth-Heinemann, Ch. 16.
#         IMO CII Guidelines (MEPC.338(76)) use the cube-law approximation
#         for the relationship between fuel consumption and speed.
# NOTE:   The synthetic data generator uses vessel-class-specific exponents
#         (range 2.7–3.3) as a hidden effect NOT captured by this baseline.
#         See VESSEL_CLASSES[*]["_gen_speed_exp"] and data/generate_data.py.

HOURS_PER_DAY: float = 24.0


# ---------------------------------------------------------------------------
# 2.  Fuel lower heating values (LHV)
#     Unit: MJ / kg
# ---------------------------------------------------------------------------

LHV_MJ_PER_KG: dict[str, float] = {
    # ----- Conventional fossil fuels ----------------------------------------
    "HFO":      40.2,
    # SOURCE: IMO MEPC.1/Circ.684 (2009) "Interim Guidelines on the Method of
    #         Calculation of the EEDI for New Ships", Annex, Table 1.

    "VLSFO":    40.5,
    # SOURCE: IMO MEPC.212(63) (2011) Appendix IX, Table 1 lists "LFO" at
    #         40.5 MJ/kg. VLSFO (≤0.5% sulphur) is compositionally close.
    # TODO_VERIFY TV-01: VLSFO is not a named category in MEPC.212(63);
    #   using the LFO proxy (40.5 MJ/kg) pending a dedicated regulatory entry.

    "MGO":      42.7,
    # SOURCE: IMO MEPC.1/Circ.684 (2009), Annex, Table 1 (MDO/MGO).

    # ----- Alternative fuels ------------------------------------------------
    "LNG":      48.0,
    # SOURCE: IMO 4th GHG Study (2020), MEPC 75/7/15, Table 5.1.

    "METHANOL": 19.9,
    # SOURCE: IMO 4th GHG Study (2020), Table 5.1.
    # TODO_VERIFY TV-04: FuelEU Maritime Delegated Regulation (EU) 2023/1640,
    #   Annex I lists 19.93 MJ/kg; difference is <0.16%. Using 19.9 from IMO.
    #   Confirm which value the applicable standard mandates.

    "AMMONIA":  18.6,
    # SOURCE: IMO 4th GHG Study (2020), Table 5.1.
    # TODO_VERIFY TV-35: Confirm the ammonia LHV used for the green-fuel
    #   proxy against the applicable IMO fuel-quality specification.

    "LH2":     119.9,
    # SOURCE: IMO 4th GHG Study (2020), Table 5.1 (gaseous H2 LHV;
    #         liquefaction changes density, not LHV).
    # TODO_VERIFY TV-34: Confirm the LH2 LHV basis against ISO 6976 and an
    #   IMO-recognized marine fuel specification before operational use.

    "AMMONIA_GREEN": 18.6,
    # TODO_VERIFY TV-35: Proxy copied from ammonia LHV; verify against an
    #   IMO-recognized marine fuel specification and account for fuel purity.

    "LH2_GREEN": 119.9,
    # TODO_VERIFY TV-34: Proxy copied from hydrogen LHV; verify against
    #   ISO 6976 and an IMO-recognized marine fuel specification.
}

# Reference fuel for all VLSFO-equivalent normalisation
REFERENCE_FUEL: str = "VLSFO"
LHV_VLSFO_MJ_PER_KG: float = LHV_MJ_PER_KG["VLSFO"]

# Fuels that appear in the simulation (subset of all defined fuels)
SIMULATION_FUELS: list[str] = ["HFO", "VLSFO", "MGO", "LNG", "METHANOL"]
EXTENDED_FUELS: list[str] = ["LH2_GREEN", "AMMONIA_GREEN"]


# ---------------------------------------------------------------------------
# 3.  CO2 emission factors — Tank-to-Wake (TTW)
#     Unit: g CO2 / g fuel  (dimensionless ratio; multiply fuel mass in t
#           to get CO2 mass in t)
# ---------------------------------------------------------------------------

CO2_TTW_G_PER_G_FUEL: dict[str, float] = {
    "HFO":      3.114,
    # SOURCE: IMO MEPC.212(63) (2011) Appendix IX, Table 1.

    "VLSFO":    3.151,
    # SOURCE: IMO MEPC.212(63) Appendix IX, Table 1 (listed as "LFO").
    # TODO_VERIFY TV-02: VLSFO has no dedicated entry in MEPC.212(63);
    #   using LFO value (3.151 g/g) as proxy.

    "MGO":      3.206,
    # SOURCE: IMO MEPC.212(63) Appendix IX, Table 1.

    "LNG":      2.750,
    # SOURCE: IMO MEPC.212(63) Appendix IX, Table 1.

    "METHANOL": 1.375,
    # SOURCE: IMO MEPC.212(63) Appendix IX, Table 1.

    "AMMONIA":  0.000,
    # No carbon content. SOURCE: IMO 4th GHG Study (2020), Section 5.
    # TODO_VERIFY TV-36: Confirm zero TTW CO2 against IMO MEPC.212(63)
    #   carbon-content conventions and account separately for N2O emissions.

    "LH2":      0.000,
    # No carbon content. SOURCE: IMO 4th GHG Study (2020), Section 5.
    # TODO_VERIFY TV-37: Confirm zero TTW CO2 against IMO MEPC.212(63)
    #   carbon-content conventions for the selected hydrogen pathway.

    "AMMONIA_GREEN": 0.000,
    # TODO_VERIFY TV-36: Proxy based on ammonia containing no carbon; verify
    #   against IMO MEPC.212(63) and account separately for N2O emissions.

    "LH2_GREEN": 0.000,
    # TODO_VERIFY TV-37: Proxy based on hydrogen containing no carbon; verify
    #   against IMO MEPC.212(63) for the selected hydrogen pathway.
}


# ---------------------------------------------------------------------------
# 4.  CO2 emission factors — Well-to-Wake (WTW)
#     Unit: g CO2eq / MJ  (multiply fuel energy content in MJ to get CO2eq)
#     Emission formula:
#       energy_mj = fuel_t * 1e3 * LHV_MJ_PER_KG   (t→kg via 1e3, MJ/kg)
#       co2_wtw_t = energy_mj * CO2_WTW_G_CO2EQ_PER_MJ * 1e-6   (g→t)
# ---------------------------------------------------------------------------

CO2_WTW_G_CO2EQ_PER_MJ: dict[str, float] = {
    "HFO":              86.2,
    # SOURCE: FuelEU Maritime Delegated Regulation (EU) 2023/1640, Annex I,
    #         "Default WTW GHG intensity values", fossil HFO pathway.

    "VLSFO":            86.2,
    # SOURCE: No dedicated VLSFO entry in FuelEU Maritime Annex I (2023).
    #         Using HFO as proxy (same crude-oil origin, similar WTW chain).
    # TODO_VERIFY TV-03: Confirm VLSFO WTW factor against a regulatory source;
    #   actual value may differ slightly from HFO.

    "MGO":              89.0,
    # SOURCE: FuelEU Maritime Delegated Regulation (EU) 2023/1640, Annex I,
    #         fossil MGO pathway.

    "LNG":              75.7,
    # SOURCE: FuelEU Maritime Delegated Regulation (EU) 2023/1640, Annex I,
    #         fossil LNG pathway (without upstream methane-slip correction).

    "METHANOL":        105.8,
    # SOURCE: FuelEU Maritime Delegated Regulation (EU) 2023/1640, Annex I,
    #         fossil methanol (natural-gas-based) pathway.

    "METHANOL_GREEN":    6.7,
    # SOURCE: FuelEU Maritime Delegated Regulation (EU) 2023/1640, Annex I,
    #         e-methanol (electrolytic hydrogen + captured CO2) pathway.
    # TODO_VERIFY TV-05: Strongly pathway-dependent; 6.7 g/MJ assumes >90%
    #   renewable electricity for electrolysis. Grid mix determines actual value.

    "AMMONIA_FOSSIL":  120.7,
    # TODO_VERIFY TV-06: FuelEU Maritime Annex I (2023) does not list a fossil
    #   ammonia entry at time of writing. Value from ICCT Working Paper 2022-28
    #   "Opportunities for hydrogen and ammonia as zero-carbon shipping fuels."
    #   Not a regulatory document.

    "AMMONIA_GREEN":     3.5,
    # TODO_VERIFY TV-07: Verify against FuelEU Maritime Annex I and a
    #   pathway-specific JEC Well-to-Wheels lifecycle assessment.
    #   Range is 2–6 g CO2eq/MJ depending on H2 production pathway per ICCT
    #   Working Paper 2022-28. Using midpoint estimate.

    "LH2_GREEN":         6.1,
    # TODO_VERIFY TV-08: Verify against FuelEU Maritime Annex I and a
    #   pathway-specific JEC/IEA lifecycle assessment, including liquefaction.
    #   Proxy informed by IRENA (2022), Fig. 3.2.
}


# ---------------------------------------------------------------------------
# 5.  Fuel-specific engine efficiency ratios (relative to VLSFO = 1.0)
#     Unit: dimensionless
#
#     Interpretation:
#       fuel_t = vlsfo_equiv_t * (LHV_VLSFO / LHV_fuel) / efficiency_ratio
#
#     A ratio < 1.0 means more physical fuel mass is needed for the same work
#     output compared with VLSFO.  A ratio > 1.0 means less fuel mass needed.
#
# TODO_VERIFY TV-09 through TV-14: All values below. Engine efficiency ratio
#   is engine-type, load-point, and retrofit-specific. No single IMO document
#   mandates these values. Ranges drawn from:
#     - DNV GL (2019) "Comparison of alternative marine fuels", Report
#       No. 2019-0567 (public version).
#     - MAN Energy Solutions product guides: ME-GI (LNG), ME-LGIM (methanol).
#     - DSME/MAN ES ammonia engine development reports (2023, not published).
# ---------------------------------------------------------------------------

ENGINE_EFFICIENCY_RATIO: dict[str, float] = {
    "HFO":      0.99,
    # TODO_VERIFY TV-09: Marginally lower LHV → marginally more fuel mass
    #   per unit energy; typical two-stroke SFC on HFO vs VLSFO ~1% higher.
    #   Reference: DNV GL 2019-0567, Table 5.

    "VLSFO":    1.00,
    # Reference fuel; ratio defined as 1.0 by convention.

    "MGO":      1.02,
    # TODO_VERIFY TV-10: Higher LHV and cleaner combustion slightly improve
    #   thermal efficiency on a dedicated diesel engine.
    #   Reference: DNV GL 2019-0567, Table 5.

    "LNG":      0.95,
    # TODO_VERIFY TV-11: Dual-fuel MEGI (two-stroke) engines incur a ~2–5%
    #   thermal efficiency penalty vs HFO due to methane slip and pilot oil
    #   consumption. MAN ES ME-GI Project Guide (2022), Table 1.

    "METHANOL": 0.85,
    # TODO_VERIFY TV-12: Lower energy density requires higher volumetric fuel
    #   flow; dual-fuel operation in ME-LGIM engines quoted at ~15% higher SFC
    #   per unit energy than VLSFO baseline.
    #   Reference: MAN ES ME-LGIM Project Guide (2023), Section 3.

    "AMMONIA":  0.82,
    # TODO_VERIFY TV-13: Pre-commercial; significant pilot-fuel requirement
    #   and combustion challenges lower net thermal efficiency.
    #   Reference: MAN ES ammonia engine development press releases (2023);
    #   DSME ammonia dual-fuel retrofit study (2023, confidential summary).

    "LH2":      0.70,
    # TODO_VERIFY TV-14: Hydrogen combustion in adapted diesel/Otto-cycle
    #   engines still experimental at commercial scale. Broad range in
    #   literature. IMO 4th GHG Study (2020), Chapter 4 technology assessment,
    #   reports conversion efficiencies of 45–65% for H2 combustion engines.

    "AMMONIA_GREEN": 0.82,
    # TODO_VERIFY TV-41: Verify net engine efficiency and pilot-fuel demand
    #   against MAN ES ammonia engine specifications and sea-trial data.

    "LH2_GREEN": 0.70,
    # TODO_VERIFY TV-40: Verify net engine efficiency against IMO 4th GHG
    #   Study technology data and marine hydrogen engine sea trials.
}


# ---------------------------------------------------------------------------
# 6.  Vessel class specifications
#
# TODO_VERIFY TV-15 through TV-18: All DWT, TEU capacity, design speed, and
#   reference daily fuel consumption (k) values are REPRESENTATIVE of typical
#   market vessels. Sources:
#     - Alphaliner fleet statistics (2024 summary)
#     - Clarksons Research "Container Ship Fleet Register" (2023)
#     - MAN Energy Solutions engine selection guides (public)
#   IMO DCS (MEPC.282(70)) collects per-vessel data but does not publish
#   standardised per-class specifications. ALL values require confirmation
#   against a single authoritative source before production use.
#
# TODO_VERIFY TV-19: Speed exponents (key "_gen_speed_exp") are SYNTHETIC
#   GENERATOR parameters only. They are NOT used in the physics baseline.
#   Literature range for the speed-fuel exponent is 2.7–3.3 (Carlton 2007).
# ---------------------------------------------------------------------------

VESSEL_CLASSES: dict[str, dict] = {
    "Feeder": {
        # Representative 1,000-TEU feeder vessel
        "dwt_t":              12_000,   # TODO_VERIFY TV-15
        "teu_capacity":        1_000,   # TODO_VERIFY TV-15
        "design_speed_kn":      17.0,   # TODO_VERIFY TV-15
        # k = reference daily fuel consumption in t VLSFO/day at design speed
        # and full design load.  Derived from ~15 MW installed power × 200
        # g/kWh SFOC × 24 h × 1e-6 t/g ≈ 72 t/day; slow-steamed feeder
        # reference ~28 t/day.  TODO_VERIFY TV-15.
        "ref_daily_fuel_t":     28.0,   # t VLSFO/day
        "age_range_yr":       (0, 25),  # Operational age range for generator
        # ---- SYNTHETIC GENERATOR ONLY — not used in physics baseline --------
        "_gen_speed_exp":        2.9,   # TODO_VERIFY TV-19
    },
    "Panamax": {
        # Original Panamax (beam limited to 32.2 m by Panama Canal old locks)
        "dwt_t":              55_000,   # TODO_VERIFY TV-16
        "teu_capacity":        5_000,   # TODO_VERIFY TV-16
        "design_speed_kn":      22.0,   # TODO_VERIFY TV-16
        "ref_daily_fuel_t":    100.0,   # t VLSFO/day  TODO_VERIFY TV-16
        "age_range_yr":       (0, 25),
        "_gen_speed_exp":        3.1,   # TODO_VERIFY TV-19
    },
    "PostPanamax": {
        # Neo-Panamax / post-Panamax, 10,000–14,999 TEU range
        "dwt_t":              90_000,   # TODO_VERIFY TV-17
        "teu_capacity":       10_000,   # TODO_VERIFY TV-17
        "design_speed_kn":      23.0,   # TODO_VERIFY TV-17
        "ref_daily_fuel_t":    160.0,   # t VLSFO/day  TODO_VERIFY TV-17
        "age_range_yr":       (0, 25),
        "_gen_speed_exp":        3.0,   # TODO_VERIFY TV-19
    },
    "ULCS": {
        # Ultra-Large Container Ship, ≥18,000 TEU
        "dwt_t":             200_000,   # TODO_VERIFY TV-18
        "teu_capacity":       22_000,   # TODO_VERIFY TV-18
        "design_speed_kn":      23.0,   # TODO_VERIFY TV-18
        "ref_daily_fuel_t":    260.0,   # t VLSFO/day  TODO_VERIFY TV-18
        "age_range_yr":       (0, 25),
        "_gen_speed_exp":        2.8,   # TODO_VERIFY TV-19
    },
}


# ---------------------------------------------------------------------------
# 7.  Reference routes
#
# TODO_VERIFY TV-23 through TV-28: All route distances (nm).
#   Distances are approximate great-circle or rhumb-line estimates.
#   Actual operational distances depend on routing, weather avoidance, and
#   canal transit.  Sources attempted:
#     - Sea-Distances.org (online tool; not a primary regulatory source)
#     - Port Authority published sailing distances where available
#   ALL distances must be verified against Lloyd's List Intelligence, Intertanko
#   voyage calculator, or equivalent authoritative maritime source before
#   production use.
# ---------------------------------------------------------------------------

ROUTES: list[dict] = [
    {
        "name":               "Singapore–Rotterdam (via Suez)",
        "origin_port":        "Singapore",
        "dest_port":          "Rotterdam",
        "distance_nm":         9_900,
        # TODO_VERIFY TV-23: Sea-Distances.org gives ~9,870–9,950 nm.
        "typical_laden_pct":    0.75,   # TODO_VERIFY: industry estimate
        "cargo_demand_teu":    8_000,
        "deadline_days":           26,
    },
    {
        "name":               "Shanghai–Los Angeles (Trans-Pacific)",
        "origin_port":        "Shanghai",
        "dest_port":          "Los Angeles",
        "distance_nm":         5_600,
        # TODO_VERIFY TV-24: Great-circle ~5,470 nm; operational routing ~5,600 nm.
        "typical_laden_pct":    0.90,
        "cargo_demand_teu":   10_000,
        "deadline_days":           18,
    },
    {
        "name":               "Rotterdam–New York (North Atlantic)",
        "origin_port":        "Rotterdam",
        "dest_port":          "New York",
        "distance_nm":         3_450,
        # TODO_VERIFY TV-25: Great-circle ~3,340 nm; operational routing ~3,450 nm.
        "typical_laden_pct":    0.80,
        "cargo_demand_teu":    6_000,
        "deadline_days":           12,
    },
    {
        "name":               "Singapore–Sydney (South Pacific)",
        "origin_port":        "Singapore",
        "dest_port":          "Sydney",
        "distance_nm":         3_350,
        # TODO_VERIFY TV-26: Sea-Distances.org ~3,280–3,400 nm.
        "typical_laden_pct":    0.70,
        "cargo_demand_teu":    3_000,
        "deadline_days":           11,
    },
    {
        "name":               "Shanghai–Singapore (Intra-Asia)",
        "origin_port":        "Shanghai",
        "dest_port":          "Singapore",
        "distance_nm":         1_650,
        # TODO_VERIFY TV-27: Sea-Distances.org ~1,600–1,700 nm.
        "typical_laden_pct":    0.65,
        "cargo_demand_teu":    2_000,
        "deadline_days":            6,
    },
    {
        "name":               "Hamburg–Busan (via Suez)",
        "origin_port":        "Hamburg",
        "dest_port":          "Busan",
        "distance_nm":        11_200,
        # TODO_VERIFY TV-28: Sea-Distances.org ~11,100–11,300 nm.
        "typical_laden_pct":    0.78,
        "cargo_demand_teu":    9_000,
        "deadline_days":           29,
    },
]


# ---------------------------------------------------------------------------
# 8.  Synthetic data generator parameters
#     These control HIDDEN EFFECTS in generate_data.py that are NOT present
#     in the analytic physics baseline.  Clearly separated to avoid confusion.
# ---------------------------------------------------------------------------

GENERATOR: dict = {
    # Random seed for all numpy RNG operations in data generation
    "RANDOM_SEED": 42,

    # Number of voyages to generate (must be ≥ 5,000)
    "N_VOYAGES": 5_100,

    # ---- Effect 1: speed exponent per vessel class --------------------------
    # See VESSEL_CLASSES[*]["_gen_speed_exp"]; physics baseline uses 3.0.

    # ---- Effect 2: Hull fouling multiplicative penalty by vessel age --------
    # fuel_multiplier = 1 + FOULING_RATE_PER_YEAR * age_years
    # TODO_VERIFY TV-20: 0.5%/year is a commonly cited figure in fleet
    #   performance studies (Dalgaard et al. 2011, JMSE). Actual range is
    #   0.3–1.0% depending on anti-fouling coating type and dry-dock cycle.
    "FOULING_RATE_PER_YEAR": 0.005,

    # ---- Effect 3: Engine part-load efficiency penalty ----------------------
    # penalty_mult = 1 + PART_LOAD_COEFF * (1 - speed/design_speed)^2
    # TODO_VERIFY TV-21: Polynomial form is a simplification of real SFOC
    #   curves which are engine-specific (MAN ES shop test data, not public).
    "PART_LOAD_COEFF": 0.08,

    # ---- Effect 4: Nonlinear weather–speed interaction ----------------------
    # effective_weather = weather_factor
    #   * (1 + WEATHER_SPEED_K * (weather_factor - 1) * speed/design_speed)
    # Adverse weather has a larger relative impact at higher vessel speeds.
    # TODO_VERIFY TV-22: Coefficient derived from simplified wave-resistance
    #   reasoning; not directly sourced from wave resistance literature.
    "WEATHER_SPEED_K": 0.40,
}


# ---------------------------------------------------------------------------
# 9.  Modelling parameters
# ---------------------------------------------------------------------------

MODEL_RANDOM_SEED: int = 42
TRAIN_TEST_SPLIT:  float = 0.80   # 80% train, 20% test — fixed; no data leakage

XGBOOST_DEFAULT_PARAMS: dict = {
    "n_estimators":     200,
    "max_depth":          5,
    "learning_rate":    0.05,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "random_state":    MODEL_RANDOM_SEED,
    "n_jobs":            -1,
    "tree_method":   "hist",   # Required for XGBoost ≥ 2.0 on CPU
    "verbosity":          0,
}

QIEA_PARAMS: dict = {
    "population_size":       20,
    "n_generations":         20,
    "n_cv_folds":             3,   # CV folds used in fitness evaluation
    "n_seeds":                5,   # Outer seeds for mean/std reporting
    # Δθ for rotation-gate update (radians)
    # Narayanan & Moore (1996) suggest small values; 0.05π ≈ 9°
    "rotation_delta":  0.05 * 3.141592653589793,
    # Cap n_estimators in the inner CV loop for runtime; final test
    # evaluation uses the fully decoded (up to 300) value
    "inner_n_estimators": 100,
}


# ---------------------------------------------------------------------------
# 10. Phase 2 Fleet Optimization Parameters
# ---------------------------------------------------------------------------

# Bunker fuel market prices (USD / metric tonne) — 2024 indicative averages
# TODO_VERIFY TV-29 through TV-33: Sourced from Ship & Bunker Rotterdam indices /
# Platts maritime fuel assessments; highly variable over time.
BUNKER_PRICES_USD_PER_TONNE: dict[str, float] = {
    "HFO":              480.0,
    "VLSFO":            620.0,
    "MGO":              820.0,
    "LNG":              750.0,
    "METHANOL":         890.0,
    "METHANOL_GREEN":  1600.0,
    "AMMONIA_FOSSIL":   700.0,
    "AMMONIA_GREEN":   1300.0,  # TODO_VERIFY TV-39: IRENA/IEA green ammonia cost data and port bunker quotes.
    "LH2_GREEN":       2800.0,  # TODO_VERIFY TV-38: IEA hydrogen cost data and liquefied-H2 port bunker quotes.
}

FUEL_COST_PROXY_USD_PER_GJ: dict[str, float] = {
    fuel: BUNKER_PRICES_USD_PER_TONNE[fuel] / LHV_MJ_PER_KG[fuel]
    for fuel in EXTENDED_FUELS
}
# TODO_VERIFY TV-38/TV-39: Energy-normalized costs inherit the uncertainty of
#   the fuel price proxies and LHV values above.

# Multi-objective weights for composite objective:
# J = w1 * FuelCost ($) + w2 * LifecycleCO2 ($/tonne) + w3 * Delay ($/day)
OPTIMIZATION_WEIGHTS: dict[str, float] = {
    "w1_fuel_cost":            1.0,     # Weight on direct bunker cost (USD)
    "w2_co2_emission":       100.0,     # Shadow carbon price ($/t CO2eq; e.g. EU ETS proxy)
    "w3_schedule_penalty": 100000.0,   # Late delivery penalty ($/day past deadline — high to prevent late delivery)
}

# Constraint penalty multipliers (applied when constraints are violated)
CONSTRAINT_PENALTIES: dict[str, float] = {
    "cargo_capacity_per_teu":    200.0,   # $/TEU shortfall
    "emissions_cap_per_tonne":  2000.0,   # $/tonne over emissions cap (ensures feasible fleet compliance beats violation)
    "vessel_overlap":         500000.0,   # $ per fleet scheduling conflict
}

# Default fleet vessel availability limits (maximum routes each vessel in the fleet can be assigned)
DEFAULT_FLEET_VESSEL_LIMITS: dict[str, int] = {
    "Feeder Reference":      1,
    "Panamax Reference":     2,
    "PostPanamax Reference": 2,
    "ULCS Reference":        2,
}

# Default fleet lifecycle emissions cap for the demo scenario (tonnes WTW CO2eq)
DEFAULT_FLEET_EMISSIONS_CAP_T: float = 35_000.0

# Phase 2 Algorithm Configurations (Equal 1,200 Evaluation Budget per Method)
PHASE2_QI_PARAMS: dict = {
    "population_size": 30,    # 30 pop x 40 gen = 1,200 evaluations
    "n_generations":   40,
    "rotation_delta":  0.05 * 3.141592653589793,
}

PHASE2_SQA_PARAMS: dict = {
    "n_trotters":   24,       # 24 trotters x 50 steps = 1,200 evaluations
    "n_steps":      50,
    "gamma_init":    2.0,
    "gamma_final":   0.01,
    "temperature":   1.0,
}

PHASE2_GA_PARAMS: dict = {
    "population_size": 30,    # 30 pop x 40 gen = 1,200 evaluations
    "n_generations":   40,
    "crossover_rate":   0.80,
    "mutation_rate":    0.15,
}

PHASE2_MILP_PARAMS: dict = {
    "speed_bins":    5,      # 5 discrete speed points (60%, 70%, 80%, 90%, 100% design speed)
    "time_limit_s": 120,     # 2 minutes max execution time-box
}


# ---------------------------------------------------------------------------
# 11. Filesystem paths (all relative to project root via pathlib)
# ---------------------------------------------------------------------------

PROJECT_ROOT:      pathlib.Path = pathlib.Path(__file__).parent
DATA_DIR:          pathlib.Path = PROJECT_ROOT / "data"
RESULTS_DIR:       pathlib.Path = PROJECT_ROOT / "results"
VOYAGES_PARQUET:   pathlib.Path = DATA_DIR / "voyages.parquet"
METRICS_CSV:       pathlib.Path = RESULTS_DIR / "prediction_metrics.csv"
QI_COMPARISON_CSV: pathlib.Path = RESULTS_DIR / "qi_comparison.csv"
DB_PATH:           pathlib.Path = DATA_DIR / "qfleet.db"


# ---------------------------------------------------------------------------
# TODO_VERIFY — Master List
# All items that could not be confirmed against a single authoritative primary
# source.  Resolve each before using this codebase for production decisions.
# ---------------------------------------------------------------------------

TODO_VERIFY_ITEMS: list[str] = [
    # Fuels
    "TV-01  VLSFO LHV (40.5 MJ/kg): No dedicated IMO MEPC entry. Using LFO "
        "proxy from MEPC.212(63) Appendix IX, Table 1.",
    "TV-02  VLSFO TTW CO2 factor (3.151 g/g): No dedicated MEPC.212(63) "
        "entry. Using LFO proxy.",
    "TV-03  VLSFO WTW CO2eq factor (86.2 g/MJ): No FuelEU Maritime Annex I "
        "entry. Using HFO as proxy.",
    "TV-04  Methanol LHV (19.9 MJ/kg): FuelEU Maritime Annex I (2023) gives "
        "19.93 MJ/kg. Difference <0.2%; using IMO 4th GHG Study value.",
    "TV-05  Green e-methanol WTW factor (6.7 g CO2eq/MJ): Strongly pathway- "
        "dependent; assumes >90% renewable electricity for electrolysis.",
    "TV-06  Fossil ammonia WTW factor (120.7 g CO2eq/MJ): Sourced from ICCT "
        "Working Paper 2022-28, not a regulatory document.",
    "TV-07  Green ammonia WTW factor (3.5 g CO2eq/MJ): Production-pathway- "
        "dependent (range 2–6 per ICCT 2022-28).",
    "TV-08  Green LH2 WTW factor (6.1 g CO2eq/MJ): Depends on liquefaction "
        "energy penalty and H2 production route; sourced from IRENA 2022.",
    "TV-34  Green LH2 LHV (119.9 MJ/kg): Verify against ISO 6976 and an "
        "IMO-recognized marine fuel specification.",
    "TV-35  Green ammonia LHV (18.6 MJ/kg): Verify against an IMO-recognized "
        "marine fuel specification and fuel-purity data.",
    "TV-36/37  Hydrogen/ammonia TTW CO2 factors (0 g/g): Verify against IMO "
        "MEPC.212(63); assess ammonia N2O separately.",
    "TV-38  Green LH2 price ($2,800/t): Verify against IEA hydrogen cost "
        "data and actual liquefied-hydrogen port bunker quotes.",
    "TV-39  Green ammonia price ($1,300/t): Verify against IRENA/IEA green "
        "ammonia cost data and actual port bunker quotes.",
    "TV-40/41  Green LH2/ammonia engine-efficiency ratios (0.70/0.82): "
        "Verify against marine engine specifications and sea-trial data.",
    # Engine efficiency ratios
    "TV-09  HFO engine efficiency ratio (0.99): Indicative; DNV GL 2019-0567.",
    "TV-10  MGO engine efficiency ratio (1.02): Indicative; DNV GL 2019-0567.",
    "TV-11  LNG engine efficiency ratio (0.95): MAN ES ME-GI Project Guide "
        "(2022); actual methane slip and pilot oil fraction vary per vessel.",
    "TV-12  Methanol engine efficiency ratio (0.85): MAN ES ME-LGIM Project "
        "Guide (2023); two-stroke specific.",
    "TV-13  Ammonia engine efficiency ratio (0.82): Pre-commercial; MAN ES "
        "ammonia engine development reports (2023).",
    "TV-14  LH2 engine efficiency ratio (0.70): Experimental at commercial "
        "scale; IMO 4th GHG Study 2020, Chapter 4.",
    # Vessel classes
    "TV-15  Feeder: DWT 12,000 t, TEU 1,000, speed 17 kn, k=28 t/day — "
        "representative only; no IMO DCS standard defines class specs.",
    "TV-16  Panamax: DWT 55,000 t, TEU 5,000, speed 22 kn, k=100 t/day — "
        "representative only.",
    "TV-17  PostPanamax: DWT 90,000 t, TEU 10,000, speed 23 kn, k=160 t/day "
        "— representative only.",
    "TV-18  ULCS: DWT 200,000 t, TEU 22,000, speed 23 kn, k=260 t/day — "
        "representative only.",
    # Generator hidden-effect coefficients
    "TV-19  Speed exponents per class (2.9, 3.1, 3.0, 2.8): SYNTHETIC "
        "GENERATOR ONLY. Literature range 2.7–3.3 (Carlton 2007).",
    "TV-20  Hull fouling rate (0.5%/year): Dalgaard et al. (2011, JMSE); "
        "actual range 0.3–1.0% depending on coating and dry-dock cycle.",
    "TV-21  Part-load efficiency coefficient (0.08): Simplified polynomial; "
        "real SFOC curves are engine-specific (MAN ES shop test data).",
    "TV-22  Weather–speed interaction coefficient (0.40): Simplified model; "
        "not directly sourced from wave resistance literature.",
    # Routes
    "TV-23  Singapore–Rotterdam distance (9,900 nm): Sea-Distances.org; not "
        "a regulatory source.",
    "TV-24  Shanghai–Los Angeles distance (5,600 nm): Approximate operational "
        "routing.",
    "TV-25  Rotterdam–New York distance (3,450 nm): Approximate.",
    "TV-26  Singapore–Sydney distance (3,350 nm): Approximate.",
    "TV-27  Shanghai–Singapore distance (1,650 nm): Approximate.",
    "TV-28  Hamburg–Busan distance (11,200 nm): Approximate.",
    # Phase 3 Benchmark Instance Assumptions
    "TV-29  Medium instance emissions cap (58,000 t CO2eq): Synthetic benchmark constraint "
        "calibrated to enforce binding low-carbon fuel substitution (unconstrained ~62k t, min ~55k t).",
    "TV-30  Large instance emissions cap (122,000 t CO2eq): Synthetic benchmark constraint "
        "calibrated to enforce binding low-carbon fuel substitution (unconstrained ~128k t, min ~116.5k t).",
]

# ---------------------------------------------------------------------------
# Phase 3 Benchmark Configuration
# ---------------------------------------------------------------------------

BENCHMARK_SEEDS: list[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

BENCHMARK_EVAL_BUDGETS: dict[str, int] = {
    "Small": 1200,
    "Medium": 3000,
    "Large": 6000,
}

BENCHMARK_INSTANCE_PARAMS: dict[str, dict] = {
    "Small": {
        "name": "Small_4V_6R",
        "num_vessels": 4,
        "num_routes": 6,
        "emissions_cap_t": 35000.0,
        "eval_budget": 1200,
        "ga_params": {"pop_size": 30, "n_generations": 40},   # 30 * 40 = 1200
        "sqa_params": {"n_trotters": 24, "n_steps": 49},      # 24 + 24*49 = 1200
        "qiea_params": {"pop_size": 30, "n_generations": 40}, # 30 * 40 = 1200
    },
    "Medium": {
        "name": "Medium_8V_15R",
        "num_vessels": 8,
        "num_routes": 15,
        "emissions_cap_t": 58000.0,  # Binding cap (unconstrained ~62k t, min ~55k t)
        "eval_budget": 3000,
        "ga_params": {"pop_size": 50, "n_generations": 60},   # 50 * 60 = 3000
        "sqa_params": {"n_trotters": 30, "n_steps": 99},      # 30 + 30*99 = 3000
        "qiea_params": {"pop_size": 50, "n_generations": 60}, # 50 * 60 = 3000
    },
    "Large": {
        "name": "Large_15V_30R",
        "num_vessels": 15,
        "num_routes": 30,
        "emissions_cap_t": 122000.0,  # Binding cap (unconstrained ~128k t, min ~116.5k t)
        "eval_budget": 6000,
        "ga_params": {"pop_size": 60, "n_generations": 100},  # 60 * 100 = 6000
        "sqa_params": {"n_trotters": 40, "n_steps": 149},     # 40 + 40*149 = 6000
        "qiea_params": {"pop_size": 60, "n_generations": 100},# 60 * 100 = 6000
    },
}

BENCHMARK_RESULTS_CSV: pathlib.Path = RESULTS_DIR / "benchmark_results.csv"

