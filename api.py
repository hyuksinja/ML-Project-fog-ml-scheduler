"""
api.py
──────
Minimal backend API for the dashboard:

  POST /predict   -> { predicted_time, uncertainty }
  POST /schedule  -> { best_node, risk }

Also serves the existing frontend at:
  GET /dashboard  -> frontend/dashboard.html
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from core.simulator import FogCloudEnvironmentSimulator
from scheduler.uqe_model import UncertaintyQuantifiedEnsemble
from scheduler.uasp import UncertaintyAwareScheduler
from core.features import FEATURE_COLS, TARGET_COL


@dataclass(frozen=True)
class LiveInput:
    task_size: float
    memory: float
    cpu: float
    bandwidth: float
    current_load: float


def _as_float(payload: dict[str, Any], key: str) -> float:
    try:
        v = float(payload[key])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Invalid or missing field: {key}") from e
    if not np.isfinite(v):
        raise HTTPException(status_code=422, detail=f"Non-finite field: {key}")
    return v


def _parse_input(payload: dict[str, Any]) -> LiveInput:
    return LiveInput(
        task_size=_as_float(payload, "task_size"),
        memory=_as_float(payload, "memory"),
        cpu=_as_float(payload, "cpu"),
        bandwidth=_as_float(payload, "bandwidth"),
        current_load=_as_float(payload, "current_load"),
    )


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _node_from_input(inp: LiveInput) -> dict[str, float]:
    node_mips = max(500.0, inp.cpu * 2_000.0)
    node_ram_gb = max(0.25, inp.memory / 1024.0)
    node_bandwidth_mbps = max(1.0, inp.bandwidth)
    node_load = _clip(inp.current_load, 0.0, 0.95)

    return {
        "node_mips": node_mips,
        "node_ram_gb": node_ram_gb,
        "node_bandwidth_mbps": node_bandwidth_mbps,
        "node_load": node_load,
        "node_type": 0.0,  # fog-edge by default for direct device-facing tasks
    }


def _task_from_input(inp: LiveInput, node: dict[str, float]) -> np.ndarray:
    task_size_mi = max(1.0, inp.task_size)
    task_mem_req_mb = max(1.0, inp.memory)

    # If user doesn't provide data size / deadline / type, derive conservative defaults.
    task_data_size_mb = max(0.1, min(task_mem_req_mb, max(0.1, task_size_mi * 0.02)))
    task_priority = 2.0
    base_est = max((task_size_mi * 1e6) / (node["node_mips"] * 1e3), 0.001)
    task_deadline_s = max(0.5, base_est * 3.0)
    task_type_id = 0.0

    return np.array(
        [
            task_size_mi,
            task_mem_req_mb,
            task_data_size_mb,
            task_priority,
            task_deadline_s,
            task_type_id,
        ],
        dtype=float,
    )


def _node_pool(inp: LiveInput) -> list[dict[str, float]]:
    # Small heterogeneous pool; loads are adjusted around current_load.
    base = _clip(inp.current_load, 0.0, 0.95)
    loads = [_clip(base + d, 0.05, 0.95) for d in (0.25, 0.05, -0.10, 0.15, -0.20, 0.00)]

    return [
        {"name": "fog-edge-1", "node_mips": 1200.0, "node_ram_gb": 2.0, "node_bandwidth_mbps": 50.0, "node_load": loads[0], "node_type": 0.0},
        {"name": "fog-edge-2", "node_mips": 800.0, "node_ram_gb": 1.0, "node_bandwidth_mbps": 30.0, "node_load": loads[1], "node_type": 0.0},
        {"name": "fog-mid-1", "node_mips": 5000.0, "node_ram_gb": 8.0, "node_bandwidth_mbps": 200.0, "node_load": loads[2], "node_type": 1.0},
        {"name": "fog-mid-2", "node_mips": 3500.0, "node_ram_gb": 12.0, "node_bandwidth_mbps": 400.0, "node_load": loads[3], "node_type": 1.0},
        {"name": "cloud-1", "node_mips": 20000.0, "node_ram_gb": 64.0, "node_bandwidth_mbps": 800.0, "node_load": loads[4], "node_type": 2.0},
        {"name": "cloud-2", "node_mips": 15000.0, "node_ram_gb": 32.0, "node_bandwidth_mbps": 600.0, "node_load": loads[5], "node_type": 2.0},
    ]


def _risk_from(pred_time: float, pred_std: float) -> float:
    # Map to a bounded 0–1 score (higher = riskier).
    score = pred_std / max(pred_time, 1e-6)
    return float(_clip(score, 0.0, 1.0))


app = FastAPI(title="AEPUAS Live API", version="1.0")

_uqe: UncertaintyQuantifiedEnsemble | None = None
_scheduler: UncertaintyAwareScheduler | None = None


@app.on_event("startup")
def _startup() -> None:
    global _uqe, _scheduler

    n_samples = int(os.environ.get("AEPUAS_API_TRAIN_SAMPLES", "2500"))
    n_boot = int(os.environ.get("AEPUAS_API_BOOTSTRAP", "15"))

    print(f"[startup] training live model (samples={n_samples}, bootstrap={n_boot})", flush=True)
    t0 = time.perf_counter()
    sim = FogCloudEnvironmentSimulator(n_samples=n_samples, random_seed=42)
    df = sim.generate()
    print(f"[startup] dataset ready in {(time.perf_counter() - t0):.2f}s", flush=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    t1 = time.perf_counter()
    uqe = UncertaintyQuantifiedEnsemble(n_bootstrap=n_boot, random_seed=42)
    uqe.fit(X, y)
    print(f"[startup] model trained in {(time.perf_counter() - t1):.2f}s", flush=True)

    _uqe = uqe
    _scheduler = UncertaintyAwareScheduler(uqe, alpha=1.5)
    print(f"[startup] ready (total {(time.perf_counter() - t0):.2f}s)", flush=True)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(os.path.join("frontend", "dashboard.html"))


# Serve static frontend assets (if you add any later)
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, float]:
    if _uqe is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    t0 = time.perf_counter()
    inp = _parse_input(payload)
    node = _node_from_input(inp)
    task = _task_from_input(inp, node)

    x = UncertaintyAwareScheduler.build_feature_vector(task, node).reshape(1, -1)
    y_pred, y_std = _uqe.predict(x)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(
        "[predict] "
        f"task_size={inp.task_size:g} mem={inp.memory:g} cpu={inp.cpu:g} bw={inp.bandwidth:g} load={inp.current_load:g} "
        f"-> time={float(y_pred[0]):.3f}s std={float(y_std[0]):.3f} ({dt_ms:.1f}ms)",
        flush=True,
    )

    return {
        "predicted_time": float(y_pred[0]),
        "uncertainty": float(y_std[0]),
    }


@app.post("/schedule")
def schedule(payload: dict[str, Any]) -> dict[str, Any]:
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")

    t0 = time.perf_counter()
    inp = _parse_input(payload)
    pool = _node_pool(inp)

    # Build task vector using a representative node for deadline estimation.
    rep_node = pool[0].copy()
    rep_node.pop("name", None)
    task = _task_from_input(inp, rep_node)

    node_pool = []
    for n in pool:
        node_pool.append(
            {
                "node_mips": n["node_mips"],
                "node_ram_gb": n["node_ram_gb"],
                "node_bandwidth_mbps": n["node_bandwidth_mbps"],
                "node_load": n["node_load"],
                "node_type": n["node_type"],
                "_name": n["name"],
            }
        )

    best, ranked = _scheduler.schedule(task, node_pool)
    best_name = str(best["node"].get("_name", "unknown"))
    risk = _risk_from(float(best["pred_time"]), float(best["pred_std"]))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(
        "[schedule] "
        f"task_size={inp.task_size:g} mem={inp.memory:g} cpu={inp.cpu:g} bw={inp.bandwidth:g} load={inp.current_load:g} "
        f"-> best={best_name} risk={risk:.3f} time={float(best['pred_time']):.3f}s std={float(best['pred_std']):.3f} ({dt_ms:.1f}ms)",
        flush=True,
    )

    return {
        "best_node": best_name,
        "risk": risk,
        "debug": {
            "ranked": [
                {
                    "name": str(r["node"].get("_name", "unknown")),
                    "pred_time": float(r["pred_time"]),
                    "uncertainty": float(r["pred_std"]),
                    "risk_score": float(r["risk_score"]),
                }
                for r in ranked
            ]
        },
    }

