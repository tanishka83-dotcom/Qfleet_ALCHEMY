"""
QFleet Phase 1 — Quantum-Inspired Evolutionary Feature Selection
================================================================
Implements a Quantum-Inspired Evolutionary Algorithm (QI-EA) that jointly
selects feature groups and tunes XGBoost hyperparameters.

Algorithm: Narayanan & Moore (1996) Quantum-Inspired Genetic Algorithms
Reference: A. Narayanan and M. Moore, "Quantum-inspired genetic algorithms,"
           Proceedings of IEEE International Conference on Evolutionary
           Computation (ICEC 1996), pp. 61–66.

REPRESENTATION
--------------
Each individual is a vector of 11 qubit angles θ ∈ [0, π/2].

Feature-group angles (indices 0–5):
  0: speed_kn           (individual feature)
  1: load_factor        (individual feature)
  2: weather_factor     (individual feature)
  3: distance_nm        (individual feature)
  4: vessel_class       (GROUP — all vc_* dummies selected/dropped together)
  5: fuel_type          (GROUP — all ft_* dummies selected/dropped together)

Selection rule: p_i = sin²(θ_i); include group i if p_i > 0.5.

Hyperparameter angles (indices 6–10):
  6: max_depth          → round(2 + 8 × sin²(θ)) → int ∈ [2, 10]
  7: learning_rate      → 0.01 + 0.29 × sin²(θ)  → float ∈ [0.01, 0.30]
  8: n_estimators       → round(50 + 250 × sin²(θ))→ int ∈ [50, 300]
  9: subsample          → 0.60 + 0.40 × sin²(θ)  → float ∈ [0.60, 1.00]
  10: colsample_bytree  → 0.60 + 0.40 × sin²(θ)  → float ∈ [0.60, 1.00]

ROTATION-GATE UPDATE (Narayanan & Moore Table 1)
-------------------------------------------------
  For each feature angle i:
    if current_bit[i] == best_bit[i]  →  no rotation (Δθ = 0)
    if current_bit[i] = 0, best_bit[i] = 1, current NOT better  →  +Δθ
    if current_bit[i] = 0, best_bit[i] = 1, current IS better   →  −Δθ
    if current_bit[i] = 1, best_bit[i] = 0, current NOT better  →  −Δθ
    if current_bit[i] = 1, best_bit[i] = 0, current IS better   →  +Δθ

  For hyperparameter angles: gradient-style nudge toward best individual.
    θ_hp ← θ_hp + 0.1 × (best_angle_hp − θ_hp)

EVALUATION BUDGET (same for all methods per seed)
--------------------------------------------------
  QI-EA budget:      population_size × n_generations × n_cv_folds XGBoost fits
                   = 20 × 20 × 3 = 1,200 per seed
  Random Search:   same 1,200 total fits (400 configs × 3-fold CV)
  Plain XGBoost:   1 fit (all features, default params)

  n_inner_estimators is capped at `inner_n_estimators` (default 100) during
  the inner CV loop to keep runtime acceptable.  The fully decoded
  n_estimators is used for the final test-set evaluation.

COMPARISON PROTOCOL
-------------------
  Each method runs across n_seeds outer seeds (default 5).
  CV MAE measured on the TRAINING split only.
  Final MAE measured on the UNTOUCHED TEST split.
  Results: mean ± std of test MAE across seeds → qi_comparison.csv.
  If QI-EA does not beat Random Search, this is reported honestly.
"""

from __future__ import annotations

import sys
import pathlib
import time
import warnings

_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import config

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI_HALF: float = np.pi / 2.0

FEATURE_GROUPS: list[str] = [
    "speed_kn", "load_factor", "weather_factor", "distance_nm",
    "vessel_class", "fuel_type",
]
N_FG: int = len(FEATURE_GROUPS)   # 6  — feature-group angles
N_HP: int = 5                      # hyperparameter angles
N_ANGLES: int = N_FG + N_HP        # 11 total


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_split(seed: int) -> tuple:
    """
    Load voyages, one-hot-encode, split 80/20.

    Returns
    -------
    X_train_full, X_test_full : pd.DataFrame (all dummy columns)
    y_train, y_test           : np.ndarray
    vc_cols, ft_cols          : list[str]  column group memberships
    """
    df = pd.read_parquet(config.VOYAGES_PARQUET)
    df = df.reset_index(drop=True)

    dummies_vc = pd.get_dummies(df["vessel_class"], prefix="vc", drop_first=False)
    dummies_ft = pd.get_dummies(df["fuel_type"],    prefix="ft", drop_first=False)
    vc_cols = list(dummies_vc.columns)
    ft_cols = list(dummies_ft.columns)

    X_full = pd.concat(
        [df[["speed_kn", "load_factor", "weather_factor", "distance_nm"]],
         dummies_vc, dummies_ft],
        axis=1,
    )
    y = df["fuel_consumed_t"].values

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(
        idx, train_size=config.TRAIN_TEST_SPLIT,
        random_state=seed, shuffle=True,
    )
    return (
        X_full.iloc[tr_idx].reset_index(drop=True),
        X_full.iloc[te_idx].reset_index(drop=True),
        y[tr_idx],
        y[te_idx],
        vc_cols,
        ft_cols,
    )


def apply_mask(
    X: pd.DataFrame,
    bits: np.ndarray,
    vc_cols: list[str],
    ft_cols: list[str],
) -> pd.DataFrame:
    """
    Select feature columns according to binary group mask.

    bits[0..3]: individual features (speed, load, weather, distance)
    bits[4]   : vessel_class group (all vc_* columns)
    bits[5]   : fuel_type group (all ft_* columns)

    If all bits are 0, falls back to speed_kn to avoid degenerate input.
    """
    cols: list[str] = []
    for i, name in enumerate(["speed_kn", "load_factor",
                               "weather_factor", "distance_nm"]):
        if bits[i]:
            cols.append(name)
    if bits[4]:
        cols.extend(vc_cols)
    if bits[5]:
        cols.extend(ft_cols)
    if not cols:
        cols = ["speed_kn"]   # fallback
    return X[cols]


# ---------------------------------------------------------------------------
# Angle decoders
# ---------------------------------------------------------------------------

def _p(angles: np.ndarray, i: int) -> float:
    """Qubit selection probability for angle i: sin²(θ_i)."""
    return float(np.sin(angles[i]) ** 2)


def angles_to_bits(angles: np.ndarray) -> np.ndarray:
    """Convert feature-group angles to binary selection vector (include if p > 0.5)."""
    return np.array([int(_p(angles, i) > 0.5) for i in range(N_FG)])


def decode_hyperparams(angles: np.ndarray, cap_estimators: int | None = None) -> dict:
    """
    Map hyperparameter angles [N_FG : N_FG+N_HP] to XGBoost param values.

    cap_estimators: if set, cap n_estimators at this value (used for inner CV).
    """
    p = np.sin(angles[N_FG:]) ** 2   # shape (5,)
    n_est = int(round(50 + 250 * float(p[2])))
    if cap_estimators is not None:
        n_est = min(n_est, cap_estimators)
    return {
        "max_depth":        int(round(2 + 8 * float(p[0]))),
        "learning_rate":    float(0.01 + 0.29 * float(p[1])),
        "n_estimators":     n_est,
        "subsample":        float(0.60 + 0.40 * float(p[3])),
        "colsample_bytree": float(0.60 + 0.40 * float(p[4])),
    }


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_cv(
    angles: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    vc_cols: list[str],
    ft_cols: list[str],
    seed: int,
) -> float:
    """
    Evaluate individual on training split only (CV MAE; higher = better).

    Uses inner_n_estimators cap for runtime.  Returns negative MAE.
    """
    n_folds   = config.QIEA_PARAMS["n_cv_folds"]
    cap_est   = config.QIEA_PARAMS["inner_n_estimators"]

    bits   = angles_to_bits(angles)
    X_sel  = apply_mask(X_train, bits, vc_cols, ft_cols)
    params = decode_hyperparams(angles, cap_estimators=cap_est)

    model = xgb.XGBRegressor(
        **params,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        model, X_sel, y_train,
        cv=kf, scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return float(np.mean(scores))   # negative MAE; higher = better


def evaluate_test(
    angles: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    vc_cols: list[str],
    ft_cols: list[str],
    seed: int,
) -> float:
    """
    Train on full training set, evaluate on untouched test set.
    Uses fully decoded hyperparameters (no n_estimators cap).
    """
    bits   = angles_to_bits(angles)
    X_tr   = apply_mask(X_train, bits, vc_cols, ft_cols)
    X_te   = apply_mask(X_test,  bits, vc_cols, ft_cols)
    params = decode_hyperparams(angles, cap_estimators=None)

    model = xgb.XGBRegressor(
        **params,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_tr, y_train)
    return float(mean_absolute_error(y_test, model.predict(X_te)))


# ---------------------------------------------------------------------------
# QI-EA
# ---------------------------------------------------------------------------

def rotation_sign(x_i: int, b_i: int, current_better: bool) -> int:
    """
    Return the sign of the rotation gate Δθ for angle i.

    Implements the lookup table from Narayanan & Moore (1996), Table 1.

    x_i            : current individual's bit at position i
    b_i            : best individual's bit at position i
    current_better : True if current individual fitness > best-so-far fitness
    """
    if x_i == b_i:
        return 0
    if x_i == 0 and b_i == 1:
        return -1 if current_better else +1
    # x_i == 1 and b_i == 0
    return +1 if current_better else -1


def qiea(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    vc_cols: list[str],
    ft_cols: list[str],
    seed:    int,
) -> dict:
    """Run one QI-EA optimisation. CV on train only; final eval on test."""
    rng      = np.random.default_rng(seed)
    pop_size = config.QIEA_PARAMS["population_size"]
    n_gen    = config.QIEA_PARAMS["n_generations"]
    delta_theta = config.QIEA_PARAMS["rotation_delta"]

    # Initialise population: θ ∈ [π/6, π/3]  (sin² ≈ 0.25–0.75; avoids edges)
    population = rng.uniform(np.pi / 6, np.pi / 3, size=(pop_size, N_ANGLES))

    best_angles  = population[0].copy()
    best_fitness = -np.inf
    n_evals      = 0
    t_start      = time.perf_counter()

    for gen in range(n_gen):
        fitnesses = np.array([
            evaluate_cv(ind, X_train, y_train, vc_cols, ft_cols, seed)
            for ind in population
        ])
        n_evals += pop_size * config.QIEA_PARAMS["n_cv_folds"]

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_angles  = population[gen_best_idx].copy()

        best_bits = angles_to_bits(best_angles)

        # Rotation-gate update
        for i_ind in range(pop_size):
            current_bits   = angles_to_bits(population[i_ind])
            current_better = (fitnesses[i_ind] > best_fitness)

            # Feature-group angles: Narayanan & Moore rotation table
            for i_fg in range(N_FG):
                sign = rotation_sign(current_bits[i_fg], best_bits[i_fg],
                                     current_better)
                population[i_ind, i_fg] = float(np.clip(
                    population[i_ind, i_fg] + sign * delta_theta,
                    0.0, PI_HALF,
                ))

            # Hyperparameter angles: gradient nudge toward best
            for i_hp in range(N_FG, N_ANGLES):
                diff = best_angles[i_hp] - population[i_ind, i_hp]
                population[i_ind, i_hp] = float(np.clip(
                    population[i_ind, i_hp] + 0.10 * diff,
                    0.0, PI_HALF,
                ))

    runtime_s  = time.perf_counter() - t_start
    test_mae   = evaluate_test(best_angles, X_train, y_train,
                                X_test, y_test, vc_cols, ft_cols, seed)
    best_bits  = angles_to_bits(best_angles)
    selected   = [FEATURE_GROUPS[i] for i in range(N_FG) if best_bits[i]]
    best_hp    = decode_hyperparams(best_angles, cap_estimators=None)

    return {
        "algorithm":         "QIEA",
        "seed":              seed,
        "test_mae":          test_mae,
        "cv_mae":            -best_fitness,
        "selected_features": ", ".join(selected) if selected else "(none→speed_kn)",
        "best_params":       str(best_hp),
        "n_evaluations":     n_evals,
        "runtime_s":         round(runtime_s, 2),
    }


# ---------------------------------------------------------------------------
# Random Search baseline
# ---------------------------------------------------------------------------

def random_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    vc_cols: list[str],
    ft_cols: list[str],
    seed:    int,
    n_eval_budget: int,
) -> dict:
    """
    Random search over the same angle space with the same evaluation budget.

    n_configs = n_eval_budget // n_cv_folds
    Each configuration is a random angle vector in [0, π/2]^N_ANGLES.
    """
    rng     = np.random.default_rng(seed + 10_000)   # different stream from QI-EA
    n_folds = config.QIEA_PARAMS["n_cv_folds"]
    n_cfg   = max(1, n_eval_budget // n_folds)        # same total fits

    best_fitness = -np.inf
    best_angles  = rng.uniform(0.0, PI_HALF, size=N_ANGLES)
    t_start      = time.perf_counter()

    for _ in range(n_cfg):
        angles = rng.uniform(0.0, PI_HALF, size=N_ANGLES)
        fit    = evaluate_cv(angles, X_train, y_train, vc_cols, ft_cols, seed)
        if fit > best_fitness:
            best_fitness = fit
            best_angles  = angles.copy()

    runtime_s  = time.perf_counter() - t_start
    test_mae   = evaluate_test(best_angles, X_train, y_train,
                                X_test, y_test, vc_cols, ft_cols, seed)
    best_bits  = angles_to_bits(best_angles)
    selected   = [FEATURE_GROUPS[i] for i in range(N_FG) if best_bits[i]]
    best_hp    = decode_hyperparams(best_angles, cap_estimators=None)

    return {
        "algorithm":         "RandomSearch",
        "seed":              seed,
        "test_mae":          test_mae,
        "cv_mae":            -best_fitness,
        "selected_features": ", ".join(selected) if selected else "(none→speed_kn)",
        "best_params":       str(best_hp),
        "n_evaluations":     n_cfg * n_folds,
        "runtime_s":         round(runtime_s, 2),
    }


# ---------------------------------------------------------------------------
# Plain XGBoost baseline
# ---------------------------------------------------------------------------

def plain_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    seed:    int,
) -> dict:
    """All features, default params from config. Single fit — minimal cost."""
    params = {**config.XGBOOST_DEFAULT_PARAMS, "random_state": seed}
    t_start = time.perf_counter()
    model   = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    test_mae   = float(mean_absolute_error(y_test, model.predict(X_test)))
    runtime_s  = time.perf_counter() - t_start
    all_feat   = ", ".join(FEATURE_GROUPS)
    return {
        "algorithm":         "PlainXGBoost",
        "seed":              seed,
        "test_mae":          test_mae,
        "cv_mae":            float("nan"),
        "selected_features": all_feat,
        "best_params":       str(params),
        "n_evaluations":     1,
        "runtime_s":         round(runtime_s, 2),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run QI-EA, Random Search, and Plain XGBoost across n_seeds seeds.

    Returns
    -------
    summary_df : aggregated mean ± std per algorithm → saved to CSV
    raw_df     : one row per (algorithm, seed)
    """
    pop_size = config.QIEA_PARAMS["population_size"]
    n_gen    = config.QIEA_PARAMS["n_generations"]
    n_folds  = config.QIEA_PARAMS["n_cv_folds"]
    n_seeds  = config.QIEA_PARAMS["n_seeds"]

    qiea_budget = pop_size * n_gen * n_folds   # total inner XGBoost fits

    print(f"[qi] QI-EA budget per seed: {qiea_budget} XGBoost CV fits")
    print(f"[qi] Running {n_seeds} seeds for each of 3 methods ...")

    all_rows: list[dict] = []

    for i in range(n_seeds):
        seed = config.MODEL_RANDOM_SEED + i
        print(f"\n[qi] --- Seed {seed} ({i+1}/{n_seeds}) ---")

        (X_train, X_test, y_train, y_test,
         vc_cols, ft_cols) = load_and_split(seed)

        print(f"[qi]   QI-EA ...")
        row = qiea(X_train, y_train, X_test, y_test, vc_cols, ft_cols, seed)
        all_rows.append(row)
        print(f"[qi]   QI-EA  test MAE={row['test_mae']:.4f}  "
              f"features=[{row['selected_features']}]")

        print(f"[qi]   Random Search ...")
        row = random_search(X_train, y_train, X_test, y_test,
                             vc_cols, ft_cols, seed, qiea_budget)
        all_rows.append(row)
        print(f"[qi]   RS     test MAE={row['test_mae']:.4f}  "
              f"features=[{row['selected_features']}]")

        print(f"[qi]   Plain XGBoost ...")
        row = plain_xgboost(X_train, y_train, X_test, y_test, seed)
        all_rows.append(row)
        print(f"[qi]   XGB    test MAE={row['test_mae']:.4f}")

    raw_df = pd.DataFrame(all_rows)

    # Aggregate mean ± std across seeds
    agg = (
        raw_df.groupby("algorithm")["test_mae"]
        .agg(mean_test_mae="mean", std_test_mae="std")
        .reset_index()
    )

    # Attach representative details from the last seed of each algorithm
    last = (
        raw_df.groupby("algorithm")
        .last()
        .reset_index()[["algorithm", "selected_features",
                         "best_params", "n_evaluations"]]
    )
    summary = agg.merge(last, on="algorithm")
    summary["run_at"] = datetime.now(timezone.utc).isoformat()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.QI_COMPARISON_CSV, index=False)
    print(f"\n[qi] Comparison table -> {config.QI_COMPARISON_CSV}")

    return summary, raw_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary, raw_df = run()

    print("\n=== QI Feature Selection -- Test MAE (mean +/- std across seeds) ===")
    for _, row in summary.iterrows():
        std = row["std_test_mae"]
        std_str = f"{std:.4f}" if not np.isnan(std) else "  N/A "
        print(f"  {row['algorithm']:20s}  {row['mean_test_mae']:.4f} +/- {std_str}")

    # Honest comparison
    algo_mae = summary.set_index("algorithm")["mean_test_mae"]
    qiea_mae = algo_mae.get("QIEA", float("inf"))
    rs_mae   = algo_mae.get("RandomSearch", float("inf"))

    print()
    if qiea_mae <= rs_mae:
        print("Result: QI-EA achieves lower or equal test MAE vs Random Search "
              f"(QIEA={qiea_mae:.4f}, RS={rs_mae:.4f}).")
    else:
        diff = qiea_mae - rs_mae
        print(f"Result: QI-EA does NOT outperform Random Search on this dataset "
              f"(QIEA={qiea_mae:.4f}, RS={rs_mae:.4f}, diff={diff:.4f} t). "
              "See qi_comparison.csv for full detail.")
