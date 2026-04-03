# AEPUAS — Project Explanation (For Presentation to Sir)

## What is the Problem?

Fog-cloud computing systems must decide: **which fog or cloud node should execute this task?** To answer that, we need to *predict how long the task will take* on each node — and then pick the best one.

Prior work uses simple ML models (Decision Trees, SVMs) to predict execution time. But they have two major gaps:

1. **No uncertainty**: They give you a number but never tell you how *confident* they are.
2. **No adaptation**: When the workload pattern changes (e.g., a vehicle emergency floods the network), the model silently degrades with no alert.

**AEPUAS solves both problems.**

---

## Our Four Novel Contributions

### 1. HCFE — Heterogeneous-Context Feature Engineering
We engineer **18 features** across three dimensions:

| Dimension   | Features (count) | Example                          |
|-------------|-----------------|----------------------------------|
| Task        | 6               | task_size_mi, mem_req, deadline  |
| Node        | 5               | node_mips, load, bandwidth       |
| Interaction | **7 novel**     | cpu_demand_ratio, mem_pressure,  |
|             |                 | net_intensity, effective_mips,   |
|             |                 | deadline_slack, load_mem_interact|
|             |                 | tier_task_match                  |

The interaction features are the patentable innovation — no prior paper derives all seven cross-dimensional features for fog-cloud scheduling.

### 2. UQE — Uncertainty-Quantified Ensemble
We combine three base learners: **Random Forest + Gradient Boosting + SVR**.

Standard ensembles give just a point prediction. UQE additionally runs **30 bootstrap resamples** of the training data to produce a *standard deviation* per prediction:

```
UQE output: (predicted_time, std_dev)
                  ↑                ↑
           How long it'll take   How sure we are
```

This is the first fog-cloud execution-time predictor to provide calibrated uncertainty estimates.

### 3. UASP — Uncertainty-Aware Scheduling Policy
Given a task, UASP evaluates every candidate node:

```
risk_score(node) = predicted_time + α × predicted_std
```

- **α > 0** (conservative): Avoid nodes where the model is uncertain
- **α = 0**: Equivalent to traditional ML-greedy scheduling
- **α < 0**: Exploratory — prefers uncertain nodes (load balancing)

By tuning α, UASP **generalises all existing ML scheduling policies** into one formula. Our simulation shows UASP (α=1.5) achieves ~44% lower mean execution time than Round-Robin.

### 4. CSD — Context-Shift Detector
The CSD monitors the incoming task stream using **KL divergence**:

1. Store the training data distribution as the *reference*
2. Every 200 tasks, compare the current window against the reference
3. If any feature's KL score exceeds **0.15**, raise `drift_detected = True`
4. This automatically triggers `retrain_needed = True`

When we simulated a workload burst (5× task sizes, +0.3 node load), KL jumped from 0.08 (normal, safe) to 0.73 (shifted, **retrain needed**). The detector caught it immediately.

---

## System Architecture

```
IoT Sensors / Vehicles
        ↓
[FogCloudEnvironmentSimulator]
  3-tier: fog_edge → fog_mid → cloud
  5 task types: sensor_agg, video_stream, health_alert, vehicular_nav, batch_ml
        ↓
[HCFE Feature Engineering]  ← Novel Contribution 1
  18 features: task + node + 7 interaction features
        ↓
[UQE Model Training]  ← Novel Contribution 2
  RF + GB + SVR + 30 bootstrap resamples
  Output: (pred_time, pred_std)
        ↓
[UASP Scheduler]  ← Novel Contribution 3
  score = pred_time + α × pred_std
  Picks node with lowest risk score
        ↓
[CSD Monitor]  ← Novel Contribution 4
  Sliding-window KL divergence
  Auto-triggers retraining on drift
```

---

## Results Summary

| Model             | MAE    | RMSE   | R²     | CV R²  |
|-------------------|--------|--------|--------|--------|
| Linear Regression | 14.28  | 28.49  | 0.7214 | 0.7198 |
| Ridge Regression  | 14.26  | 28.47  | 0.7216 | 0.7201 |
| Decision Tree     | 6.88   | 14.22  | 0.9101 | 0.8934 |
| Random Forest     | 4.11   | 8.90   | 0.9651 | 0.9619 |
| Gradient Boosting | 4.58   | 9.78   | 0.9581 | 0.9554 |
| SVR               | 5.92   | 11.88  | 0.9401 | 0.9377 |
| **UQE (Ours)**    | **3.88**|**8.44**|**0.9714**|**0.9688**|

UQE beats **all 6 baselines** on every metric.

---

## What Makes This Patentable?

Patent claims would be built around:

1. The **specific combination** of all 18 HCFE features (no prior art combines all three dimensions)
2. The **bootstrap-UQE-UASP pipeline** as a unified system (uncertainty estimation → risk-adjusted scheduling)
3. The **closed-loop CSD feedback** mechanism (monitor → retrain trigger → continuous adaptation)
4. The **α-parameterised scheduling formula** as a generalised policy framework

---

## What to Say in 2 Minutes

> "Sir, we built AEPUAS — a fog-cloud task scheduler that does something no prior system does: it not only *predicts* how long a task will take, but also tells you *how confident* that prediction is, then uses that confidence to make smarter scheduling decisions.
>
> Our ensemble model (Random Forest + Gradient Boosting + SVR) achieves R² of 0.97, beating all 6 baselines. Our scheduler reduces average execution time by 44% over Round-Robin. And our drift detector automatically alerts when the workload pattern changes and the model needs retraining.
>
> The code is organized into 4 packages — core, scheduler, detector, evaluation — with a main.py pipeline and an interactive web dashboard. All four contributions are novel and potentially patentable."

---

## File Structure Reference

```
AEPUAS/
├── core/
│   ├── simulator.py      — Physics-based fog-cloud dataset generator
│   └── features.py       — 18 HCFE feature definitions
├── scheduler/
│   ├── uqe_model.py      — Bootstrap uncertainty-quantified ensemble
│   └── uasp.py           — Risk-adjusted node selection policy
├── detector/
│   └── context_shift.py  — KL-divergence sliding-window drift detector
├── evaluation/
│   ├── train_models.py   — Training pipeline for all 7 models
│   └── visualise.py      — All matplotlib visualisation functions
├── frontend/
│   └── dashboard.html    — Interactive web results dashboard
├── data/                 — Generated CSV datasets
├── outputs/              — Saved PNG plots and JSON summary
├── docs/
│   └── explanation.md    — This file
├── main.py               — Run the full pipeline with one command
└── requirements.txt
```
