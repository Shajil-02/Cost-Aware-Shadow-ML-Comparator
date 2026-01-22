# Cost-Aware Shadow Comparison of ML Models

This project simulates a **shadow deployment** setup to compare two fraud detection models under a **fixed review budget**.
Instead of choosing a model based only on offline metrics like AUC, the system evaluates **business impact** using FP/FN costs and reports how much model behavior changes.

## Key Ideas
- **Shadow Mode**: Model B runs in parallel with Model A and is evaluated without affecting production decisions.
- **Same Review Budget**: Both models can flag only the **top K%** highest-risk transactions (e.g., 2%).
- **Disagreement Rate**: Measures rollout risk by showing how often Model B decisions differ from Model A.
- **Cost-Aware Decision**: Uses a simple cost function:
  `Cost = 10 * FN + 1 * FP`

## Outputs
The pipeline generates:
- `outputs/reports/comparison_table.csv` → row-wise predictions + disagreement
- `outputs/reports/shadow_report.txt` → summary metrics and decision
- `outputs/reports/disagreement_slices.csv` → slice-based disagreement analysis
- `outputs/plots/score_hist.png` → score distribution comparison
- `outputs/plots/disagreement_slices.png` → top disagreement slices plot

## How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
