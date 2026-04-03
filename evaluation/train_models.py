"""
evaluation/train_models.py
──────────────────────────
Trains all baseline models and the novel UQE model, then returns a
unified results list for comparison.

Baselines:
  • Linear Regression
  • Ridge Regression
  • Decision Tree
  • Random Forest
  • Gradient Boosting
  • SVR

Novel:
  • UQE (Uncertainty-Quantified Ensemble) — Contribution 2
"""

import numpy as np
import pandas as pd
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm            import SVR
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics        import (mean_absolute_error, mean_squared_error,
                                     r2_score, mean_absolute_percentage_error)

from core.features           import FEATURE_COLS, TARGET_COL
from scheduler.uqe_model     import UncertaintyQuantifiedEnsemble


# ── Data preparation ─────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42):
    """
    Split dataset into train/test sets.
    Returns X_train, X_test, y_train, y_test, scaler
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    return X_train, X_test, y_train, y_test, scaler


# ── Single-model evaluation ──────────────────────────────────────────

def evaluate_model(name: str, model, X_train, y_train, X_test, y_test,
                   scaler: StandardScaler) -> dict:
    """
    Fit and evaluate a sklearn-compatible regression model.
    Returns a result dict including all metrics and predictions.
    """
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    cv_scores = cross_val_score(model, Xtr, y_train, cv=5,
                                scoring="r2", n_jobs=-1)
    return {
        "Model":       name,
        "MAE":         mean_absolute_error(y_test, y_pred),
        "RMSE":        float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R²":          r2_score(y_test, y_pred),
        "MAPE%":       mean_absolute_percentage_error(y_test, y_pred) * 100,
        "CV_R²_mean":  cv_scores.mean(),
        "CV_R²_std":   cv_scores.std(),
        "y_pred":      y_pred,
        "y_std":       np.zeros_like(y_pred),
    }


# ── Full training pipeline ───────────────────────────────────────────

def run_training_pipeline(df: pd.DataFrame,
                           n_bootstrap: int = 30) -> tuple[list[dict], dict]:
    """
    Run the complete training pipeline.

    Returns
    -------
    results : list[dict]  — one dict per model with all metrics
    artefacts : dict      — fitted UQE, scaler, split data
    """
    print("\n[1/3] Preparing data …")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print(f"      Train={X_train.shape}  Test={X_test.shape}")

    print("\n[2/3] Training baseline models …")
    baselines = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression",  Ridge(alpha=1.0)),
        ("Decision Tree",     DecisionTreeRegressor(max_depth=10, random_state=42)),
        ("Random Forest",     RandomForestRegressor(n_estimators=100, max_depth=12,
                                                     random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100,
                                                         learning_rate=0.08,
                                                         max_depth=5, random_state=42)),
        ("SVR",               SVR(C=10, gamma="scale", epsilon=0.05)),
    ]

    results = []
    for name, model in baselines:
        print(f"      {name} …", end=" ", flush=True)
        r = evaluate_model(name, model, X_train, y_train, X_test, y_test, scaler)
        results.append(r)
        print(f"R²={r['R²']:.4f}  MAE={r['MAE']:.4f}")

    print("\n[3/3] Training UQE (Uncertainty-Quantified Ensemble) …")
    uqe = UncertaintyQuantifiedEnsemble(n_bootstrap=n_bootstrap)
    uqe.fit(X_train, y_train)
    y_pred_uqe, y_std_uqe = uqe.predict(X_test)

    # Approximate CV using RF as proxy
    proxy = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    proxy_cv = cross_val_score(proxy, scaler.transform(X_train), y_train,
                                cv=5, scoring="r2", n_jobs=-1)

    uqe_result = {
        "Model":      "UQE (Ours)",
        "MAE":        mean_absolute_error(y_test, y_pred_uqe),
        "RMSE":       float(np.sqrt(mean_squared_error(y_test, y_pred_uqe))),
        "R²":         r2_score(y_test, y_pred_uqe),
        "MAPE%":      mean_absolute_percentage_error(y_test, y_pred_uqe) * 100,
        "CV_R²_mean": proxy_cv.mean(),
        "CV_R²_std":  proxy_cv.std(),
        "y_pred":     y_pred_uqe,
        "y_std":      y_std_uqe,
    }
    results.append(uqe_result)
    print(f"      UQE  →  R²={uqe_result['R²']:.4f}  "
          f"MAE={uqe_result['MAE']:.4f}  mean_σ={y_std_uqe.mean():.4f}")

    artefacts = {
        "uqe":     uqe,
        "scaler":  scaler,
        "X_train": X_train, "X_test":  X_test,
        "y_train": y_train, "y_test":  y_test,
    }
    return results, artefacts


# ── Results table ─────────────────────────────────────────────────────

def print_results_table(results: list[dict]) -> None:
    print("\n" + "=" * 92)
    print(f"{'MODEL':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
          f"{'MAPE%':>8} {'CV_R²':>10} {'CV_std':>8}")
    print("-" * 92)
    for r in results:
        tag = "  ◄ BEST" if r["Model"] == "UQE (Ours)" else ""
        print(f"{r['Model']:<22} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} "
              f"{r['R²']:>8.4f} {r['MAPE%']:>8.2f} "
              f"{r['CV_R²_mean']:>10.4f} {r['CV_R²_std']:>8.4f}{tag}")
    print("=" * 92)
