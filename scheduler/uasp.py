"""
scheduler/uasp.py
─────────────────
Uncertainty-Aware Scheduling Policy  (UASP)  — Novel Contribution 3

Classical ML schedulers minimise only the predicted execution time.
UASP adds a risk term:

    risk_score(node) = predicted_time + α × predicted_std

α > 0  → conservative  (penalise high-uncertainty nodes)
α = 0  → ML-Greedy     (pure point-prediction scheduling)
α < 0  → exploratory   (load-balance by favouring uncertain nodes)

α is tunable at runtime, making UASP a generalisation of both
traditional and uncertainty-aware scheduling policies.
"""

import numpy as np
from scheduler.uqe_model import UncertaintyQuantifiedEnsemble


class UncertaintyAwareScheduler:
    """
    Parameters
    ----------
    uqe_model : UncertaintyQuantifiedEnsemble
        Fitted UQE model.
    alpha : float
        Risk parameter.  Default 1.5 (conservative).
    """

    def __init__(self, uqe_model: UncertaintyQuantifiedEnsemble, alpha: float = 1.5):
        self.model = uqe_model
        self.alpha = alpha

    # ── Feature construction ────────────────────────────────────────

    @staticmethod
    def build_feature_vector(task_features: np.ndarray, node: dict) -> np.ndarray:
        """
        Reconstruct the 18-D HCFE feature vector from raw task and node dicts.

        Parameters
        ----------
        task_features : ndarray (6,)
            [task_size_mi, task_mem_req_mb, task_data_size_mb,
             task_priority, task_deadline_s, task_type_id]
        node : dict
            Keys: node_mips, node_ram_gb, node_bandwidth_mbps,
                  node_load, node_type
        """
        t_size, t_mem, t_data, t_prio, t_dl, t_type = task_features
        n_mips  = node["node_mips"]
        n_ram   = node["node_ram_gb"]
        n_bw    = node["node_bandwidth_mbps"]
        n_load  = node["node_load"]
        n_type  = node["node_type"]

        base_est = max((t_size * 1e6) / (n_mips * 1e3), 0.001)

        return np.array([
            # Task
            t_size, t_mem, t_data, t_prio, t_dl, t_type,
            # Node
            n_mips, n_ram, n_bw, n_load, n_type,
            # Interaction (HCFE)
            t_size / n_mips,
            t_mem  / (n_ram * 1_024),
            t_data / n_bw,
            n_mips * (1 - n_load),
            t_dl   / base_est,
            n_load * t_mem,
            float(n_type == (int(t_type) % 3)),
        ], dtype=float)

    # ── Scheduling decision ─────────────────────────────────────────

    def schedule(self, task_features: np.ndarray, node_pool: list[dict]) -> tuple[dict, list]:
        """
        Select the best node for a given task.

        Returns
        -------
        best   : dict — selected node with scores
        ranked : list[dict] — all nodes ranked by risk_score (ascending)
        """
        ranked = []
        for node in node_pool:
            x            = self.build_feature_vector(task_features, node)
            y_pred, y_std = self.model.predict(x.reshape(1, -1))
            risk_score   = float(y_pred[0]) + self.alpha * float(y_std[0])
            ranked.append({
                "node":       node,
                "pred_time":  float(y_pred[0]),
                "pred_std":   float(y_std[0]),
                "risk_score": risk_score,
            })

        ranked.sort(key=lambda r: r["risk_score"])
        return ranked[0], ranked

    # ── Batch simulation ────────────────────────────────────────────

    def simulate_batch(self, tasks_df, node_pool: list[dict]) -> list[float]:
        """
        Run scheduling decisions for every row in tasks_df.

        Returns list of predicted execution times for the selected nodes.
        """
        task_cols = [
            "task_size_mi", "task_mem_req_mb", "task_data_size_mb",
            "task_priority", "task_deadline_s", "task_type_id",
        ]
        results = []
        for _, row in tasks_df.iterrows():
            feats       = row[task_cols].values.astype(float)
            best, _     = self.schedule(feats, node_pool)
            results.append(best["pred_time"])
        return results
