"""
main.py
───────
AEPUAS — Entry Point

Runs the full pipeline:
  Step 1 : Generate synthetic fog-cloud dataset
  Step 2 : Train & evaluate all models (baselines + UQE)
  Step 3 : Scheduling simulation (Round-Robin vs ML-Greedy vs UASP)
  Step 4 : Context-shift detection demo
  Step 5 : Save all plots & CSVs
  Step 6 : Write JSON results summary for the frontend dashboard

Usage:
    python main.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ── Local modules ──────────────────────────────────────────────────────
from core.simulator          import FogCloudEnvironmentSimulator
from core.features           import FEATURE_COLS, TARGET_COL
from evaluation.train_models import run_training_pipeline, print_results_table
from evaluation.visualise    import (plot_results_dashboard,
                                      plot_scheduling_simulation,
                                      plot_context_shift)
from scheduler.uasp          import UncertaintyAwareScheduler
from detector.context_shift  import ContextShiftDetector

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  AEPUAS: Adaptive Ensemble Predictor with Uncertainty-Aware")
    print("  Scheduling for Task Execution Time in Fog-Cloud Environments")
    print("=" * 72)

    # ─── Step 1: Dataset ────────────────────────────────────────────
    print("\n[STEP 1/5]  Generating synthetic fog-cloud dataset …")
    sim = FogCloudEnvironmentSimulator(n_samples=5_000)
    df  = sim.generate()
    df.to_csv("data/fog_cloud_dataset.csv", index=False)
    print(f"            Saved → data/fog_cloud_dataset.csv")

    # ─── Step 2: Train & Evaluate ───────────────────────────────────
    print("\n[STEP 2/5]  Training all models …")
    results, art = run_training_pipeline(df, n_bootstrap=30)
    print_results_table(results)

    uqe     = art["uqe"]
    X_test  = art["X_test"]
    y_test  = art["y_test"]

    # ─── Step 3: Visualise ──────────────────────────────────────────
    print("\n[STEP 3/5]  Generating plots …")
    plot_results_dashboard(results, uqe, X_test, y_test, df, out_dir=OUT)

    # ─── Step 4: Scheduling Simulation ─────────────────────────────
    print("\n[STEP 4/5]  Scheduling simulation …")
    NODE_POOL = [
        {"node_mips": 1200,  "node_ram_gb": 2,  "node_bandwidth_mbps": 50,  "node_load": 0.75, "node_type": 0},
        {"node_mips": 800,   "node_ram_gb": 1,  "node_bandwidth_mbps": 30,  "node_load": 0.20, "node_type": 0},
        {"node_mips": 5000,  "node_ram_gb": 8,  "node_bandwidth_mbps": 200, "node_load": 0.60, "node_type": 1},
        {"node_mips": 3500,  "node_ram_gb": 12, "node_bandwidth_mbps": 400, "node_load": 0.30, "node_type": 1},
        {"node_mips": 20000, "node_ram_gb": 64, "node_bandwidth_mbps": 800, "node_load": 0.10, "node_type": 2},
        {"node_mips": 15000, "node_ram_gb": 32, "node_bandwidth_mbps": 600, "node_load": 0.50, "node_type": 2},
    ]
    task_sample = df.sample(500, random_state=99).reset_index(drop=True)
    task_cols   = ["task_size_mi", "task_mem_req_mb", "task_data_size_mb",
                   "task_priority", "task_deadline_s", "task_type_id"]

    sched_rr   = UncertaintyAwareScheduler(uqe, alpha=0.0)
    sched_ml   = UncertaintyAwareScheduler(uqe, alpha=0.0)
    sched_uasp = UncertaintyAwareScheduler(uqe, alpha=1.5)

    pred_rr, pred_ml, pred_uasp = [], [], []
    for i in range(len(task_sample)):
        feats = task_sample.loc[i, task_cols].values.astype(float)
        # Round-Robin: use node at index i % 6
        from scheduler.uasp import UncertaintyAwareScheduler as UAS
        node_rr = NODE_POOL[i % len(NODE_POOL)]
        x_rr    = UAS.build_feature_vector(feats, node_rr)
        p_rr, _ = uqe.predict(x_rr.reshape(1, -1))
        pred_rr.append(float(p_rr[0]))

        best_ml, _   = sched_ml.schedule(feats, NODE_POOL)
        best_uasp, _ = sched_uasp.schedule(feats, NODE_POOL)
        pred_ml.append(best_ml["pred_time"])
        pred_uasp.append(best_uasp["pred_time"])

    sched_df = pd.DataFrame({
        "Round-Robin": pred_rr,
        "ML-Greedy":   pred_ml,
        "UASP (Ours)": pred_uasp,
    })
    print(f"            RR mean={sched_df['Round-Robin'].mean():.4f}s  "
          f"ML mean={sched_df['ML-Greedy'].mean():.4f}s  "
          f"UASP mean={sched_df['UASP (Ours)'].mean():.4f}s")
    plot_scheduling_simulation(sched_df, out_dir=OUT)

    # ─── Step 5: Context-Shift Detection ───────────────────────────
    print("\n[STEP 5/5]  Context-shift detection demo …")
    csd   = ContextShiftDetector(window_size=200, threshold=0.15)
    X_ref = df[FEATURE_COLS].values[:1_000]
    csd.set_reference(X_ref)

    r_normal  = csd.detect(df[FEATURE_COLS].values[1_000:1_200])

    df_shifted              = df.copy()
    df_shifted["task_size_mi"] *= 5
    df_shifted["node_load"]    += 0.3
    df_shifted["node_load"]     = df_shifted["node_load"].clip(0, 0.95)
    r_shifted = csd.detect(df_shifted[FEATURE_COLS].values[:200])

    print(f"            Normal  → drift={r_normal['drift_detected']}  "
          f"max_KL={r_normal['max_kl']:.4f}")
    print(f"            Shifted → drift={r_shifted['drift_detected']}  "
          f"max_KL={r_shifted['max_kl']:.4f}  worst={r_shifted['worst_feature']}")
    plot_context_shift(r_normal, r_shifted, out_dir=OUT)

    # ─── Save summary JSON for frontend ────────────────────────────
    summary = {
        "models": [
            {
                "name":       r["Model"],
                "mae":        round(r["MAE"],        5),
                "rmse":       round(r["RMSE"],       5),
                "r2":         round(r["R²"],         5),
                "mape":       round(r["MAPE%"],      3),
                "cv_r2_mean": round(r["CV_R²_mean"], 5),
                "cv_r2_std":  round(r["CV_R²_std"],  5),
            }
            for r in results
        ],
        "scheduling": {
            "round_robin_mean": round(float(sched_df["Round-Robin"].mean()), 5),
            "ml_greedy_mean":   round(float(sched_df["ML-Greedy"].mean()),   5),
            "uasp_mean":        round(float(sched_df["UASP (Ours)"].mean()), 5),
        },
        "drift": {
            "normal_max_kl":  round(r_normal["max_kl"],  5),
            "shifted_max_kl": round(r_shifted["max_kl"], 5),
            "worst_feature":  r_shifted["worst_feature"],
            "threshold":      0.15,
        },
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📋 Summary JSON → {OUT}/summary.json")

    # Save model comparison CSV
    comp_df = pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ("y_pred", "y_std")}
                              for r in results])
    comp_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
    print(f"📋 Model CSV   → {OUT}/model_comparison.csv")

    print("\n✅  All done!  Open frontend/dashboard.html in your browser.")
    print("=" * 72)


if __name__ == "__main__":
    main()
