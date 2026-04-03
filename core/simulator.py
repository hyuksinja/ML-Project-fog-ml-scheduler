"""
core/simulator.py
─────────────────
FogCloudEnvironmentSimulator

Generates a realistic synthetic dataset for a 3-tier IoT–Fog–Cloud
environment.  Execution time is modelled with physics-based equations
inspired by papers on GWO scheduling, hybrid workflow scheduling, and
dynamic IoT task scheduling (2026 literature).

Execution-time model:
    base_time  = (task_size × 10⁶) / (node_mips × 10³ × freq_factor)
    memory_pen = 1 + max(0, (mem_req − mem_avail) / mem_avail)
    load_pen   = 1 / (1 − min(node_load, 0.95))
    net_delay  = data_size / bandwidth × propagation_factor
    exec_time  = base_time × memory_pen × load_pen + net_delay + noise
"""

import numpy as np
import pandas as pd
from core.features import FEATURE_COLS, TARGET_COL


class FogCloudEnvironmentSimulator:
    """
    Simulates a heterogeneous fog-cloud environment with three node tiers
    and five distinct task types representative of real IoT workloads.
    """

    # Node profiles: layer → (mips_range, ram_gb_range, bw_mbps_range, type_id)
    NODE_PROFILES = {
        "fog_edge": ((500,  2_000),  (1,   4),   (10,  100),  0),
        "fog_mid":  ((2_000, 8_000), (4,   16),  (50,  500),  1),
        "cloud":    ((8_000, 32_000),(16,  128),  (200, 1_000), 2),
    }

    # Task profiles: name → (size_mi_range, mem_mb_range, data_mb_range, priority)
    TASK_TYPES = {
        "sensor_agg":    ((10,   500),   (64,   512),   (0.1,  5),    1),
        "video_stream":  ((200,  2_000), (256,  2_048), (5,    50),   2),
        "health_alert":  ((50,   800),   (128,  1_024), (0.5,  10),   3),
        "vehicular_nav": ((100,  1_500), (256,  2_048), (1,    20),   2),
        "batch_ml":      ((1_000,10_000),(1_024, 8_192),(10,   200),  1),
    }

    PROPAGATION = {"fog_edge": 0.001, "fog_mid": 0.005, "cloud": 0.02}

    def __init__(self, n_samples: int = 5_000, random_seed: int = 42):
        self.n_samples   = n_samples
        self.random_seed = random_seed

    # ── Private helpers ───────────────────────────────────────────────

    def _sample_node(self, rng, layer: str) -> dict:
        mips_r, ram_r, bw_r, type_id = self.NODE_PROFILES[layer]
        return {
            "node_mips":          rng.uniform(*mips_r),
            "node_ram_gb":        rng.uniform(*ram_r),
            "node_bandwidth_mbps":rng.uniform(*bw_r),
            "node_load":          rng.beta(2, 3),   # realistic skewed load
            "node_type":          type_id,
            "_layer":             layer,
        }

    def _sample_task(self, rng, ttype: str) -> dict:
        sz_r, mem_r, data_r, prio = self.TASK_TYPES[ttype]
        return {
            "task_size_mi":     rng.uniform(*sz_r),
            "task_mem_req_mb":  rng.uniform(*mem_r),
            "task_data_size_mb":rng.uniform(*data_r),
            "task_priority":    prio,
            "task_deadline_s":  rng.uniform(0.5, 30),
            "_ttype":           ttype,
        }

    def _exec_time(self, task: dict, node: dict, rng) -> tuple[float, float]:
        freq_factor  = 1.0 + 0.3 * np.sin(rng.uniform(0, 2 * np.pi))
        base_time    = (task["task_size_mi"] * 1e6) / (node["node_mips"] * 1e3 * freq_factor)
        mem_avail_mb = node["node_ram_gb"] * 1024 * (1 - node["node_load"] * 0.6)
        memory_pen   = 1 + max(0, (task["task_mem_req_mb"] - mem_avail_mb) / max(mem_avail_mb, 1))
        load_pen     = 1.0 / max(1 - min(node["node_load"], 0.95), 0.05)
        net_delay    = (task["task_data_size_mb"] / node["node_bandwidth_mbps"]) * \
                       self.PROPAGATION[node["_layer"]] * 1_000
        raw = base_time * memory_pen * load_pen + net_delay
        noise = rng.normal(0, 0.05 * raw)
        return max(raw + noise, 0.001), base_time

    # ── Public API ────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate and return the full synthetic dataset as a DataFrame."""
        rng          = np.random.default_rng(self.random_seed)
        layer_keys   = list(self.NODE_PROFILES.keys())
        layer_probs  = [0.40, 0.35, 0.25]
        ttype_keys   = list(self.TASK_TYPES.keys())
        ttype_map    = {k: i for i, k in enumerate(ttype_keys)}
        records      = []

        for _ in range(self.n_samples):
            layer = rng.choice(layer_keys, p=layer_probs)
            ttype = rng.choice(ttype_keys)
            node  = self._sample_node(rng, layer)
            task  = self._sample_task(rng, ttype)

            exec_time, base_time = self._exec_time(task, node, rng)

            t_size  = task["task_size_mi"]
            t_mem   = task["task_mem_req_mb"]
            t_data  = task["task_data_size_mb"]
            t_prio  = task["task_priority"]
            t_dl    = task["task_deadline_s"]
            t_tid   = float(ttype_map[ttype])

            n_mips  = node["node_mips"]
            n_ram   = node["node_ram_gb"]
            n_bw    = node["node_bandwidth_mbps"]
            n_load  = node["node_load"]
            n_type  = float(node["node_type"])

            records.append({
                # Task
                "task_size_mi":       t_size,
                "task_mem_req_mb":    t_mem,
                "task_data_size_mb":  t_data,
                "task_priority":      t_prio,
                "task_deadline_s":    t_dl,
                "task_type_id":       t_tid,
                # Node
                "node_mips":          n_mips,
                "node_ram_gb":        n_ram,
                "node_bandwidth_mbps":n_bw,
                "node_load":          n_load,
                "node_type":          n_type,
                # Interaction (HCFE)
                "cpu_demand_ratio":   t_size / n_mips,
                "mem_pressure":       t_mem  / (n_ram * 1_024),
                "net_intensity":      t_data / n_bw,
                "effective_mips":     n_mips * (1 - n_load),
                "deadline_slack":     t_dl   / max(base_time, 0.001),
                "load_mem_interact":  n_load * t_mem,
                "tier_task_match":    float(n_type == (int(t_tid) % 3)),
                # Target
                TARGET_COL:           exec_time,
            })

        df = pd.DataFrame(records)
        print(f"✅  Dataset: {df.shape[0]} rows × {df.shape[1]} cols  |  "
              f"exec_time min={df[TARGET_COL].min():.3f}s  "
              f"max={df[TARGET_COL].max():.3f}s  "
              f"mean={df[TARGET_COL].mean():.3f}s")
        return df
