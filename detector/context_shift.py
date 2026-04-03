"""
detector/context_shift.py
──────────────────────────
Context-Shift Detector  (CSD)  — Novel Contribution 4

Monitors the incoming task stream for distribution drift using a
sliding-window KL-divergence approach:

  1. A reference distribution is recorded from the training data.
  2. For each new window of tasks, per-feature KL divergences are computed.
  3. If the maximum KL across features exceeds a threshold, a drift event
     is raised and the scheduler is flagged to trigger model retraining.

This closed-loop mechanism — predict → schedule → monitor → retrain —
forms the self-healing aspect of AEPUAS and is central to its patentability.
"""

import numpy as np
from core.features import FEATURE_COLS


class ContextShiftDetector:
    """
    Sliding-window KL-divergence drift detector.

    Parameters
    ----------
    window_size : int
        Number of recent samples to compare against the reference.
    threshold   : float
        KL-divergence threshold above which drift is declared.
    n_bins      : int
        Number of histogram bins for distribution estimation.
    """

    def __init__(self, window_size: int = 200, threshold: float = 0.15, n_bins: int = 20):
        self.window_size = window_size
        self.threshold   = threshold
        self.n_bins      = n_bins
        self._reference  = None

    # ── Reference management ────────────────────────────────────────

    def set_reference(self, X: np.ndarray) -> None:
        """Store the reference (training) distribution."""
        self._reference = X.copy()
        print(f"    [CSD] Reference set from {len(X)} samples.")

    # ── KL divergence (smoothed) ─────────────────────────────────────

    def _kl_div(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        p = np.asarray(p, dtype=float) + eps
        q = np.asarray(q, dtype=float) + eps
        p /= p.sum()
        q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    # ── Detection ────────────────────────────────────────────────────

    def detect(self, X_new: np.ndarray) -> dict:
        """
        Compare X_new against the reference distribution.

        Returns
        -------
        dict with keys:
          drift_detected  : bool
          max_kl          : float
          worst_feature   : str
          all_kl          : dict[feature_name → kl_value]
          retrain_needed  : bool (alias of drift_detected)
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() before detect().")

        kl_scores = {}
        n_feats   = min(X_new.shape[1], len(FEATURE_COLS))

        for col in range(n_feats):
            ref_vals = self._reference[:, col]
            new_vals = X_new[:, col]
            lo = min(ref_vals.min(), new_vals.min())
            hi = max(ref_vals.max(), new_vals.max())
            if hi == lo:
                continue
            p, _ = np.histogram(ref_vals, bins=self.n_bins, range=(lo, hi))
            q, _ = np.histogram(new_vals, bins=self.n_bins, range=(lo, hi))
            kl_scores[FEATURE_COLS[col]] = self._kl_div(p, q)

        if not kl_scores:
            return {"drift_detected": False, "max_kl": 0.0,
                    "worst_feature": "", "all_kl": {}, "retrain_needed": False}

        worst_feat = max(kl_scores, key=kl_scores.get)
        max_kl     = kl_scores[worst_feat]
        drifted    = max_kl > self.threshold

        return {
            "drift_detected": drifted,
            "max_kl":         max_kl,
            "worst_feature":  worst_feat,
            "all_kl":         kl_scores,
            "retrain_needed": drifted,
        }

    # ── Window-based streaming detection ────────────────────────────

    def detect_stream(self, X_stream: np.ndarray) -> list[dict]:
        """
        Slide a window over X_stream and yield drift reports.
        Useful for animated / live monitoring scenarios.
        """
        reports = []
        for start in range(0, len(X_stream) - self.window_size + 1, self.window_size // 2):
            window  = X_stream[start : start + self.window_size]
            report  = self.detect(window)
            report["window_start"] = start
            reports.append(report)
        return reports
