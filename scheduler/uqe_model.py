"""
scheduler/uqe_model.py
──────────────────────
Uncertainty-Quantified Ensemble  (UQE)  — Novel Contribution 2

Architecture:
  • Three base regressors: RandomForest, GradientBoosting, SVR
  • Fitted once on full training data → gives point predictions
  • Fitted n_bootstrap times on bootstrap resamples → gives per-sample
    standard deviation (predictive uncertainty)

Output of predict():
  y_pred  – ensemble mean prediction (seconds)
  y_std   – bootstrap standard deviation (confidence width)

The y_std output is consumed by the UASP scheduler to make risk-aware
node-selection decisions — this is the core novelty.
"""

import numpy as np
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm              import SVR
from sklearn.preprocessing    import StandardScaler


class UncertaintyQuantifiedEnsemble:
    """
    Bootstrap-based uncertainty-quantified ensemble.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples used to estimate uncertainty.
        More → smoother uncertainty estimates, longer training.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(self, n_bootstrap: int = 30, random_seed: int = 42):
        self.n_bootstrap  = n_bootstrap
        self.random_seed  = random_seed
        self.scaler       = StandardScaler()
        self._base_models : list = []
        self._boot_models : list = []   # list[list[estimator]]
        self._fitted      = False

    # ── Internal factory ─────────────────────────────────────────────

    @staticmethod
    def _make_base_models():
        return [
            RandomForestRegressor(
                n_estimators=100, max_depth=12,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.08,
                max_depth=5, random_state=42
            ),
            SVR(C=10, gamma="scale", epsilon=0.05),
        ]

    # ── Public interface ─────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "UncertaintyQuantifiedEnsemble":
        """
        Fit base models on the full training set, then fit bootstrap
        models for uncertainty estimation.
        """
        rng = np.random.default_rng(self.random_seed)
        Xs  = self.scaler.fit_transform(X_train)
        n   = len(X_train)

        # ── Full-data base models ──────────────────────────────────
        self._base_models = self._make_base_models()
        for m in self._base_models:
            m.fit(Xs, y_train)
        print(f"    [UQE] Base models fitted on {n} samples.")

        # ── Bootstrap models for uncertainty ──────────────────────
        self._boot_models = []
        for b in range(self.n_bootstrap):
            idx   = rng.integers(0, n, size=n)   # bootstrap resample
            Xb    = Xs[idx];  yb = y_train[idx]
            boot  = self._make_base_models()
            for m in boot:
                m.fit(Xb, yb)
            self._boot_models.append(boot)
        print(f"    [UQE] Bootstrap uncertainty fitted ({self.n_bootstrap} resamples).")

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        y_pred : ndarray of shape (n,) — point prediction (ensemble mean)
        y_std  : ndarray of shape (n,) — predictive standard deviation
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        Xs = self.scaler.transform(X)

        # Point prediction from full-data base models
        base_preds = np.column_stack([m.predict(Xs) for m in self._base_models])
        y_pred     = base_preds.mean(axis=1)

        # Uncertainty from bootstrap distribution
        boot_means = np.array([
            np.column_stack([m.predict(Xs) for m in boot_set]).mean(axis=1)
            for boot_set in self._boot_models
        ])                         # shape: (n_bootstrap, n_samples)
        y_std = boot_means.std(axis=0)

        return y_pred, y_std

    def predict_point(self, X: np.ndarray) -> np.ndarray:
        """Convenience wrapper that returns only the point prediction."""
        y_pred, _ = self.predict(X)
        return y_pred
