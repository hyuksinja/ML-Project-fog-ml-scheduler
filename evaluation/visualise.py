"""
evaluation/visualise.py
───────────────────────
All matplotlib visualisation functions for AEPUAS.

Plots:
  1. Metrics comparison bar charts (MAE, R², RMSE)
  2. Actual vs Predicted with 95% confidence interval
  3. Scatter: Actual vs Predicted
  4. Feature importance (Random Forest)
  5. Residual histogram
  6. 5-fold Cross-validation R²
  7. Predicted σ vs Absolute Error (uncertainty calibration)
  8. Scheduling simulation histograms & bar chart
  9. Context-Shift Detector KL-divergence chart
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble      import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from core.features import FEATURE_COLS, TARGET_COL


# ── Colour palette ────────────────────────────────────────────────────
MODEL_COLORS = {
    "Linear Regression": "#AED6F1",
    "Ridge Regression":  "#85C1E9",
    "Decision Tree":     "#F9E79F",
    "Random Forest":     "#82E0AA",
    "Gradient Boosting": "#F0B27A",
    "SVR":               "#C39BD3",
    "UQE (Ours)":        "#EC7063",
}
BG_DARK  = "#0F1117"
PANEL_BG = "#1A1B2E"
SPINE_C  = "#333344"


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C)
    ax.tick_params(colors="#AAAAAA", labelsize=8)
    ax.xaxis.label.set_color("#CCCCCC")
    ax.yaxis.label.set_color("#CCCCCC")


TITLE_KW = dict(fontsize=10, fontweight="bold", color="white", pad=6)


# ── Main results dashboard ────────────────────────────────────────────

def plot_results_dashboard(results: list[dict], uqe, X_test, y_test, df,
                            out_dir: str = "outputs") -> str:
    """
    Generate the 4×3 dashboard of model-comparison plots.
    Returns the saved file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    model_names = [r["Model"] for r in results]
    colors      = [MODEL_COLORS.get(n, "#AAAAAA") for n in model_names]

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    uqe_res = next(r for r in results if r["Model"] == "UQE (Ours)")
    y_pred_uqe = uqe_res["y_pred"]
    y_std_uqe  = uqe_res.get("y_std", np.zeros_like(y_pred_uqe))

    # ── Row 0: MAE | R² | RMSE ───────────────────────────────────────
    for col_idx, (metric, title, fmt) in enumerate([
        ("MAE",  "Mean Absolute Error (↓ better)",    ".4f"),
        ("R²",   "R² Score (↑ better)",               ".4f"),
        ("RMSE", "Root Mean Squared Error (↓ better)",".4f"),
    ]):
        ax   = fig.add_subplot(gs[0, col_idx])
        vals = [r[metric] for r in results]
        bars = ax.bar(range(len(model_names)), vals, color=colors, width=0.6, edgecolor="none")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=6)
        ax.set_title(title, **TITLE_KW)
        if metric == "R²":
            ax.set_ylim(0, 1.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v * 1.02 if metric != "R²" else v + 0.005,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=6.5, color="white")
        _style_ax(ax)

    # ── Row 1: Actual vs Predicted (CI) | Scatter ────────────────────
    ax = fig.add_subplot(gs[1, :2])
    sort_idx = np.argsort(y_test)[:500]
    xt, xp, xs = y_test[sort_idx], y_pred_uqe[sort_idx], y_std_uqe[sort_idx]
    ax.fill_between(range(len(xt)), xp - 1.96*xs, xp + 1.96*xs,
                    alpha=0.25, color="#EC7063", label="95% CI")
    ax.plot(range(len(xt)), xt, color="#5DADE2", lw=1.2, label="Actual")
    ax.plot(range(len(xt)), xp, color="#EC7063", lw=0.9, linestyle="--", label="UQE Pred")
    ax.set_title("UQE: Actual vs Predicted with 95% Confidence Interval (500 samples)", **TITLE_KW)
    ax.set_xlabel("Sample Index");  ax.set_ylabel("Execution Time (s)")
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor="none", labelcolor="white")
    _style_ax(ax)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(y_test[:800], y_pred_uqe[:800],
               c="#EC7063", alpha=0.35, s=8, edgecolors="none")
    lim = max(y_test.max(), y_pred_uqe.max()) * 1.05
    ax.plot([0, lim], [0, lim], "w--", lw=1)
    ax.set_title("Scatter: Actual vs Predicted (UQE)", **TITLE_KW)
    ax.set_xlabel("Actual");  ax.set_ylabel("Predicted")
    _style_ax(ax)

    # ── Row 2: Feature Importance | Residuals ────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    sc   = StandardScaler()
    n_tr = int(0.8 * len(df))
    Xtr  = sc.fit_transform(df[FEATURE_COLS].values[:n_tr])
    ytr  = df[TARGET_COL].values[:n_tr]
    rf   = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    imp     = rf.feature_importances_
    s_idx   = np.argsort(imp)
    ax.barh(range(len(s_idx)), imp[s_idx], color="#82E0AA", edgecolor="none")
    ax.set_yticks(range(len(s_idx)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in s_idx], fontsize=7)
    ax.set_title("Random Forest Feature Importances (HCFE)", **TITLE_KW)
    ax.set_xlabel("Importance Score")
    _style_ax(ax)

    ax = fig.add_subplot(gs[2, 2])
    residuals = y_test - y_pred_uqe
    ax.hist(residuals, bins=50, color="#EC7063", alpha=0.75, edgecolor="none")
    ax.axvline(0, color="white", lw=1.2, linestyle="--")
    ax.set_title("UQE Residual Distribution", **TITLE_KW)
    ax.set_xlabel("Residual (s)");  ax.set_ylabel("Count")
    _style_ax(ax)

    # ── Row 3: CV R² | Uncertainty Calibration ───────────────────────
    ax = fig.add_subplot(gs[3, :2])
    cv_means = [r["CV_R²_mean"] for r in results]
    cv_stds  = [r["CV_R²_std"]  for r in results]
    bars = ax.bar(range(len(model_names)), cv_means, color=colors, width=0.55,
                  yerr=cv_stds, capsize=4, ecolor="#DDDDDD", edgecolor="none")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=6)
    ax.set_title("5-Fold Cross-Validation R² (mean ± std)", **TITLE_KW)
    ax.set_ylabel("CV R²");  ax.set_ylim(0, 1.05)
    for bar, v, s in zip(bars, cv_means, cv_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color="white")
    _style_ax(ax)

    ax = fig.add_subplot(gs[3, 2])
    abs_err = np.abs(y_test - y_pred_uqe)
    ax.scatter(y_std_uqe[:1000], abs_err[:1000],
               c="#F0B27A", alpha=0.25, s=6, edgecolors="none")
    ax.set_title("UQE Uncertainty Calibration\n(σ vs |error| — good: positive corr)", **TITLE_KW)
    ax.set_xlabel("Predicted σ (s)");  ax.set_ylabel("|Error| (s)")
    _style_ax(ax)

    fig.suptitle(
        "AEPUAS — Adaptive Ensemble Predictor with Uncertainty-Aware Scheduling\n"
        "Task Execution Time Prediction in Fog-Cloud Environments",
        fontsize=14, fontweight="bold", color="white", y=0.997,
    )
    out = f"{out_dir}/results_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Dashboard → {out}")
    return out


# ── Scheduling simulation plot ────────────────────────────────────────

def plot_scheduling_simulation(sched_df, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)

    bar_colors = ["#AED6F1", "#82E0AA", "#EC7063"]
    labels     = list(sched_df.columns)

    for ax in axes:
        _style_ax(ax)

    for col, clr in zip(labels, bar_colors):
        axes[0].hist(sched_df[col], bins=40, alpha=0.6, color=clr, label=col)
    axes[0].set_title("Distribution of Scheduled Execution Times",
                      color="white", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Execution Time (s)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(facecolor=PANEL_BG, edgecolor="none", labelcolor="white")

    means = sched_df.mean()
    bars  = axes[1].bar(labels, means.values, color=bar_colors, width=0.5, edgecolor="none")
    axes[1].set_title("Mean Predicted Execution Time per Policy",
                      color="white", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mean Exec Time (s)")
    for bar, v in zip(bars, means.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v * 1.01,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9, color="white")

    fig.suptitle("UASP Scheduling Simulation — Policy Comparison",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"{out_dir}/scheduling_simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Scheduling plot → {out}")
    return out


# ── CSD plot ──────────────────────────────────────────────────────────

def plot_context_shift(r_normal: dict, r_shifted: dict, out_dir: str = "outputs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    feats     = list(r_shifted["all_kl"].keys())
    kl_norm   = [r_normal["all_kl"].get(f, 0)  for f in feats]
    kl_shift  = [r_shifted["all_kl"].get(f, 0) for f in feats]
    x         = np.arange(len(feats))

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    ax.bar(x - 0.2, kl_norm,  0.38, label="Normal",  color="#82E0AA", alpha=0.85)
    ax.bar(x + 0.2, kl_shift, 0.38, label="Shifted", color="#EC7063", alpha=0.85)
    ax.axhline(0.15, color="yellow", lw=1.5, linestyle="--", label="Threshold=0.15")
    ax.set_xticks(x)
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8, color="#CCCCCC")
    ax.set_title("Context-Shift Detector: KL Divergence per Feature",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("KL Divergence")
    ax.legend(facecolor=PANEL_BG, edgecolor="none", labelcolor="white")

    plt.tight_layout()
    out = f"{out_dir}/context_shift_detector.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 CSD plot → {out}")
    return out
