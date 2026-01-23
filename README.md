# Cost-Aware Shadow ML Comparator

A minimal, production-inspired **shadow deployment evaluation** system that compares two fraud detection models under a **fixed review budget** and recommends whether to ship the new model based on **business cost (FP/FN)** and **behavior disagreement risk**.

---

## Overview

In real-world ML deployments, a model with a better offline metric (e.g., AUC) is not automatically safer to ship. Teams must answer:

- How much will production behavior change?
- Under the same operational constraints (review capacity), which model performs better?
- Does the new model reduce costly errors (missed fraud) or increase false alarms?

This repository implements a clean, reproducible pipeline to compare:

- **Model A** (current production baseline)
- **Model B** (candidate model in shadow mode)

Both models are evaluated under the **same review budget** (Top *K%* flagged), and the system outputs a **ship / no-ship recommendation**.

---

## Key Ideas

### 1) Shadow Deployment
Model B runs in parallel with Model A on the same traffic/data but does not affect decisions. We compare behavior and cost impact before rollout.


### 2) Same Review Budget (Top K% Flagged)
Most fraud systems have limited investigation/review capacity. This project enforces the same constraint for both models:

> Both models can flag only the **top K% highest-risk transactions**.

This ensures a fair comparison and avoids misleading threshold-based metrics.



### 3) Disagreement Rate (Rollout Risk)
The system measures how often Model B would behave differently than Model A:

> “In X% of transactions, Model B behaves differently than Model A.”


### 4) Cost-Aware Decision
We compute business cost using weighted FP/FN:
**Cost = (FN_cost * FN) + (FP_cost * FP)**

Default weights used here:
- **FN_cost = 10** (missing fraud is expensive in fraudulent scenarios)
- **FP_cost = 1** (false alarm causes review cost + user friction)

The model with lower estimated cost is recommended.

---

### Dataset Requirements

Dataset can be downloaded from 
https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

Place your dataset at:

data/raw/creditcard.csv

This project expects a binary label column:
fraud → 0 (not fraud), 1 (fraud)

note: If your dataset uses a different label name (e.g., Class), update the label column in src/data_loader.py.


### How to Run

Install the dependencies

> pip install -r requirements.txt


Run the full pipeline

> python main.py


This will:
- Split raw data into train/val/test (once)
- Train Model A and Model B
- Evaluate validation AUC for sanity check
- Compare models in shadow mode using Top K% review budget
- Compute disagreement rate + FP/FN + cost
- Save reports and plots
---

### Outputs Generated

After running, the following artifacts are created:

Reports (outputs/reports/)

comparison_table.csv
- Row-wise table containing:
  proba_A, proba_B
  pred_A, pred_B (budget-based decisions)disagree flag

shadow_report.txt
- Summary report with key metrics and final decision.

disagreement_slices.csv
- Disagreement analysis across feature slices (quantile bins).

Plots (outputs/plots/)
score_hist.png
- Predicted probability distribution for Model A vs Model B.

disagreement_slices.png
- Top slices with highest disagreement rates.


### Worked by

Shajil BP
