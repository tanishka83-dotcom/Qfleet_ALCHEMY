"""
QFleet Phase 1 — Synthetic Voyage Data Generator
==================================================
Generates 5 000+ synthetic container-ship voyages and writes them to
data/voyages.parquet.

⚠  DATA IS SYNTHETIC
---------------------
All records are produced by a parameterised physics model with controlled
noise and intentionally hidden non-linearities.  Results obtained from models
trained on this data reflect how well those models capture the *generator*
behaviour — not real-fleet operational accuracy.

The Parquet file carries the file-level metadata key:
  DATA_IS_SYNTHETIC = "true"

Physics formula (VLSFO-equivalent basis)
-----------------------------------------
  duration_day   = distance_nm / speed_kn / HOURS_PER_DAY
                   Unit: [nm] / [kn] / [h/day] = [day]

  vlsfo_equiv_t  = k * (speed_kn / design_speed_kn)^3 * duration_day
                   * load_factor * weather_factor
                   Unit: [t/day] * [1]^3 * [day] = [t]  ✓

  fuel_t         = vlsfo_equiv_t * (LHV_VLSFO / LHV_fuel) / engine_eff_ratio
                   Unit: [t] * [MJ/kg / MJ/kg] / [1] = [t]  ✓

  TTW CO2: fuel_t [t] * co2_ttw_factor [g/g = t/t] = co2_ttw_t [t]
  WTW CO2: energy_mj [MJ] * co2_wtw_factor [g/MJ] * 1e-6 [t/g] = co2_wtw_t [t]
    where energy_mj = fuel_t [t] * 1e3 [kg/t] * LHV_fuel [MJ/kg]

HIDDEN EFFECTS in the generator (NOT in the analytic physics baseline)
-----------------------------------------------------------------------
  1. Speed exponent varies by vessel class (2.7–3.3) not a fixed 3.
  2. Hull-fouling multiplier: 1 + FOULING_RATE_PER_YEAR * vessel_age_yr
  3. Engine part-load efficiency penalty:
       1 + PART_LOAD_COEFF * (1 - speed/design_speed)^2
  4. Nonlinear weather–speed interaction replaces linear weather_factor:
       eff_weather = weather_factor
                    * (1 + WEATHER_SPEED_K * (weather_factor - 1)
                         * speed / design_speed)

These create systematic residuals that the physics baseline model cannot
capture, giving ML models meaningful signal to learn.
"""

from __future__ import annotations

import sys
import pathlib

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone

import config


# ---------------------------------------------------------------------------
# Hidden-effect helpers (all receive rng for full determinism)
# ---------------------------------------------------------------------------

def _hull_fouling_mult(age_yr: float) -> float:
    """
    Multiplicative fuel penalty from hull fouling.

    formula : 1 + FOULING_RATE_PER_YEAR * age_yr
    unit    : dimensionless
    """
    return 1.0 + config.GENERATOR["FOULING_RATE_PER_YEAR"] * age_yr


def _part_load_mult(speed_kn: float, design_speed_kn: float) -> float:
    """
    Multiplicative fuel penalty from engine part-load inefficiency.

    formula : 1 + PART_LOAD_COEFF * (1 - speed / design_speed)^2
    unit    : dimensionless
    Applies when ship operates well below design speed.
    """
    ratio = speed_kn / design_speed_kn
    return 1.0 + config.GENERATOR["PART_LOAD_COEFF"] * (1.0 - ratio) ** 2


def _effective_weather(weather_raw: float, speed_kn: float,
                        design_speed_kn: float) -> float:
    """
    Nonlinear weather–speed interaction factor.

    formula : weather_raw * (1 + WEATHER_SPEED_K * (weather_raw - 1)
                                 * speed / design_speed)
    unit    : dimensionless (≥ 1)

    Rationale: adverse weather (weather_raw > 1) has a larger relative
    impact at higher vessel speeds due to wave-resistance scaling.
    The physics baseline uses a plain linear weather_raw instead of this.
    """
    k = config.GENERATOR["WEATHER_SPEED_K"]
    ratio = speed_kn / design_speed_kn
    return weather_raw * (1.0 + k * (weather_raw - 1.0) * ratio)


# ---------------------------------------------------------------------------
# Single voyage generator
# ---------------------------------------------------------------------------

def _generate_voyage(
    voyage_id: int,
    vessel_class: str,
    fuel_type: str,
    route: dict,
    rng: np.random.Generator,
) -> dict:
    """Generate one synthetic voyage record.  All randomness from `rng`."""

    spec         = config.VESSEL_CLASSES[vessel_class]
    design_speed = spec["design_speed_kn"]          # kn
    k            = spec["ref_daily_fuel_t"]          # t VLSFO/day at design speed
    speed_exp    = spec["_gen_speed_exp"]            # HIDDEN: class-specific exponent

    lhv_fuel  = config.LHV_MJ_PER_KG[fuel_type]    # MJ/kg
    lhv_vlsfo = config.LHV_VLSFO_MJ_PER_KG          # MJ/kg
    eff_ratio = config.ENGINE_EFFICIENCY_RATIO[fuel_type]   # dimensionless

    co2_ttw_factor = config.CO2_TTW_G_PER_G_FUEL[fuel_type]   # g CO2/g fuel
    co2_wtw_factor = config.CO2_WTW_G_CO2EQ_PER_MJ[fuel_type] # g CO2eq/MJ

    # ----- Sample voyage parameters ----------------------------------------
    # Operational speed: 60–100% of design speed
    speed_kn    = float(rng.uniform(0.60 * design_speed, 1.00 * design_speed))
    # Load factor: fraction of TEU capacity carrying cargo
    load_factor = float(rng.uniform(0.30, 1.00))
    # Raw weather difficulty: 1.0 (calm) to 1.30 (rough); used in physics
    # baseline as-is; generator applies nonlinear interaction on top
    weather_raw = float(rng.uniform(1.00, 1.30))
    # Vessel age drawn from class age range
    age_yr      = float(rng.uniform(*spec["age_range_yr"]))
    distance_nm = float(route["distance_nm"])

    # ----- Hidden effects (NOT in physics baseline) -------------------------
    fouling_m   = _hull_fouling_mult(age_yr)
    partload_m  = _part_load_mult(speed_kn, design_speed)
    eff_weather = _effective_weather(weather_raw, speed_kn, design_speed)

    # ----- Voyage duration --------------------------------------------------
    duration_h   = distance_nm / speed_kn           # hours; [nm/kn = h]
    duration_day = duration_h / config.HOURS_PER_DAY

    # ----- VLSFO-equivalent fuel (with HIDDEN speed exponent) ---------------
    # Unit: [t/day] × [1]^exp × [day] = [t]
    vlsfo_equiv_t = (
        k
        * (speed_kn / design_speed) ** speed_exp   # HIDDEN exponent
        * duration_day
        * load_factor
        * eff_weather                               # HIDDEN nonlinear weather
        * fouling_m                                 # HIDDEN hull fouling
        * partload_m                                # HIDDEN part-load penalty
    )

    # ----- Convert to actual fuel mass ----------------------------------------
    # fuel_t = vlsfo_equiv_t × (LHV_VLSFO / LHV_fuel) / engine_eff_ratio
    # Unit: [t] × [MJ/kg / MJ/kg] / [1] = [t]  ✓
    fuel_clean = vlsfo_equiv_t * (lhv_vlsfo / lhv_fuel) / eff_ratio
    # Add Gaussian noise: σ = 3% of clean value (measurement / stowage
    # uncertainty; same σ-fraction as typical bunker survey uncertainty)
    fuel_t = float(rng.normal(fuel_clean, 0.03 * fuel_clean))
    fuel_t = max(fuel_t, 1e-3)   # prevent non-positive values

    # ----- Emissions --------------------------------------------------------
    # TTW: fuel_t [t] × g/g ratio = co2_ttw_t [t]
    co2_ttw_t = fuel_t * co2_ttw_factor

    # WTW: energy_mj = fuel_t [t] × 1e3 [kg/t] × LHV [MJ/kg]
    #       co2_wtw_t [t] = energy_mj × g CO2eq/MJ × 1e-6 [t/g]
    energy_mj = fuel_t * 1e3 * lhv_fuel
    co2_wtw_t = energy_mj * co2_wtw_factor * 1e-6

    # ----- Approximate timestamp (2022–2024 range) --------------------------
    base_ts_unix = 1640995200  # 2022-01-01T00:00:00 UTC
    offset_s     = int(rng.integers(0, 2 * 365 * 24 * 3600))
    voyage_ts    = base_ts_unix + offset_s

    return {
        "voyage_id":         voyage_id,
        "vessel_class":      vessel_class,
        "fuel_type":         fuel_type,
        "route_name":        route["name"],
        "speed_kn":          round(speed_kn, 4),
        "design_speed_kn":   float(design_speed),
        "load_factor":       round(load_factor, 4),
        "weather_factor":    round(weather_raw, 4),   # raw, for physics baseline
        "distance_nm":       distance_nm,
        "vessel_age_yr":     round(age_yr, 2),
        "duration_h":        round(duration_h, 4),
        "fuel_consumed_t":   round(fuel_t, 4),
        "fuel_type_lhv":     float(lhv_fuel),
        "co2_ttw_t":         round(co2_ttw_t, 4),
        "co2_wtw_t":         round(co2_wtw_t, 4),
        "voyage_ts":         int(voyage_ts),
    }


# ---------------------------------------------------------------------------
# Main generation function (self-contained seed → reproducible)
# ---------------------------------------------------------------------------

def generate_voyages(
    n:    int = config.GENERATOR["N_VOYAGES"],
    seed: int = config.GENERATOR["RANDOM_SEED"],
) -> pd.DataFrame:
    """
    Generate `n` synthetic voyages.  Calling with the same `seed` always
    produces an identical DataFrame (GENERATED_AT metadata is not in the df).

    Vessel class, fuel type, and route are assigned round-robin so every
    combination appears throughout the dataset.
    """
    rng = np.random.default_rng(seed)

    vessel_classes = list(config.VESSEL_CLASSES.keys())
    fuel_types     = config.SIMULATION_FUELS
    routes         = config.ROUTES

    records = [
        _generate_voyage(
            voyage_id=i,
            vessel_class=vessel_classes[i % len(vessel_classes)],
            fuel_type=fuel_types[i % len(fuel_types)],
            route=routes[i % len(routes)],
            rng=rng,
        )
        for i in range(n)
    ]

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def save_to_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """
    Write DataFrame to Parquet (snappy compressed) with file-level metadata:
      DATA_IS_SYNTHETIC = "true"
      GENERATED_AT      = ISO-8601 UTC timestamp (informational; not in df)
      RANDOM_SEED       = seed used
      N_VOYAGES         = row count
      GENERATOR_VERSION = "phase1-v1"
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)

    existing_meta = table.schema.metadata or {}
    new_meta = {
        **{k.decode() if isinstance(k, bytes) else k: v
           for k, v in existing_meta.items()},
        "DATA_IS_SYNTHETIC": "true",
        "GENERATED_AT":      datetime.now(timezone.utc).isoformat(),
        "RANDOM_SEED":       str(config.GENERATOR["RANDOM_SEED"]),
        "N_VOYAGES":         str(len(df)),
        "GENERATOR_VERSION": "phase1-v1",
    }
    # pyarrow requires bytes keys/values in schema metadata
    new_meta_bytes = {
        k.encode() if isinstance(k, str) else k:
        v.encode() if isinstance(v, str) else v
        for k, v in new_meta.items()
    }
    table = table.replace_schema_metadata(new_meta_bytes)
    pq.write_table(table, path, compression="snappy")
    print(f"[generate_data] Saved {len(df):,} voyages -> {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("[generate_data] Generating synthetic voyages ...")
    df = generate_voyages()
    save_to_parquet(df, config.VOYAGES_PARQUET)

    # 20-row sample for quick human inspection
    sample_path = config.DATA_DIR / "voyages_sample.csv"
    df.head(20).to_csv(sample_path, index=False)
    print(f"[generate_data] Sample (20 rows) -> {sample_path}")

    # Quick summary stats
    print("\n--- Summary ---")
    print(df[["vessel_class", "fuel_type", "fuel_consumed_t",
               "co2_ttw_t", "co2_wtw_t"]].describe().round(2))


if __name__ == "__main__":
    main()
