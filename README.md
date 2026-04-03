# AEPUAS — Adaptive Ensemble Predictor with Uncertainty-Aware Scheduling

## Project Overview
Predicting Task Execution Time in Fog-Cloud Heterogeneous Environments using
an Adaptive Ensemble with Uncertainty Quantification and Drift Detection.

## Folder Structure
```
AEPUAS/
├── core/                   # Dataset simulation & feature engineering
│   ├── simulator.py        # FogCloudEnvironmentSimulator
│   └── features.py         # HCFE feature definitions
├── scheduler/              # Scheduling algorithms
│   ├── uqe_model.py        # Uncertainty-Quantified Ensemble (UQE)
│   └── uasp.py             # Uncertainty-Aware Scheduling Policy (UASP)
├── detector/               # Concept-drift detection
│   └── context_shift.py    # Context-Shift Detector (CSD)
├── evaluation/             # Training, benchmarking, visualisation
│   ├── train_models.py     # Baseline + UQE training pipeline
│   └── visualise.py        # All matplotlib plotting functions
├── frontend/               # Web dashboard (HTML + JS)
│   └── dashboard.html      # Interactive results dashboard
├── data/                   # Generated datasets
├── outputs/                # Saved plots & CSVs
├── docs/                   # Explanation & report
│   └── explanation.md
├── main.py                 # Entry point – runs full pipeline
└── requirements.txt
```

## Novel Contributions (Patentability Hooks)
1. **HCFE** – 18 heterogeneous-context features (task × node × interaction)
2. **UQE** – Bootstrap uncertainty-quantified ensemble (RF + GB + SVR)
3. **UASP** – Risk-adjusted scheduling: `score = pred_time + α·std_dev`
4. **CSD** – KL-divergence sliding-window context-shift detector with auto-retrain trigger

## Quick Start
```bash
pip install -r requirements.txt
python main.py
# Open frontend/dashboard.html in browser to explore results
```
