import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_cost(y_true, preds, fn_cost=10, fp_cost=1):
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    # cost = 10*FN + 1*FP
    return (fn_cost * fn) + (fp_cost * fp), fp, fn

def top_k_predictions(proba, k_frac=0.02):
    # Return predictions where exactly top k_frac scores are flagged as 1.
    proba = np.asarray(proba)
    n = len(proba)
    k = int(n * k_frac)
    preds = np.zeros(n, dtype=int)
    if k <= 0:
        return preds
    top_idx = np.argsort(proba)[-k:]
    preds[top_idx] = 1
    return preds

def threshold_for_top_k(proba, k_frac=0.02):
    # Threshold for reference (not used for exact predictions).
    proba = np.asarray(proba)
    return float(np.quantile(proba, 1 - k_frac))

def compare_models_budget(X_test, y_test, proba_A, proba_B, k_frac=0.02):
    # Shadow compare under SAME REVIEW BUDGET:
    # Both models flag top k_frac transactions.
    pred_A = top_k_predictions(proba_A, k_frac=k_frac)
    pred_B = top_k_predictions(proba_B, k_frac=k_frac)
    disagreement_rate = float(np.mean(pred_A != pred_B))
    avg_score_A = float(np.mean(proba_A))
    avg_score_B = float(np.mean(proba_B))
    high_risk_A = float(np.mean(proba_A >= 0.9))
    high_risk_B = float(np.mean(proba_B >= 0.9))
    cost_A, fpA, fnA = compute_cost(y_test, pred_A)
    cost_B, fpB, fnB = compute_cost(y_test, pred_B)
    tA_ref = threshold_for_top_k(proba_A, k_frac=k_frac)
    tB_ref = threshold_for_top_k(proba_B, k_frac=k_frac)
    summary = {
        "review_budget": f"Top {k_frac*100:.0f}%",
        "flagged_count_A": int(pred_A.sum()),
        "flagged_count_B": int(pred_B.sum()),
        "budget_threshold_A_ref": f"{tA_ref:.4f}",
        "budget_threshold_B_ref": f"{tB_ref:.4f}",
        "disagreement_rate": f"{disagreement_rate*100:.2f}%",
        "avg_score_A": f"{avg_score_A:.4f}",
        "avg_score_B": f"{avg_score_B:.4f}",
        "high_risk_rate_A": f"{high_risk_A*100:.2f}%",
        "high_risk_rate_B": f"{high_risk_B*100:.2f}%",
        "cost_A": f"{cost_A}",
        "cost_B": f"{cost_B}",
        "False Positives_A": f"{int(fpA)}",
        "False Negatives_A": f"{int(fnA)}",
        "False Positives_B": f"{int(fpB)}",
        "False Negatives_B": f"{int(fnB)}",
    }

    table = X_test.copy()
    table["y_true"] = y_test.values
    table["proba_A"] = proba_A
    table["proba_B"] = proba_B
    table["pred_A"] = pred_A
    table["pred_B"] = pred_B
    table["disagree"] = (pred_A != pred_B).astype(int)
    return summary, table

def disagreement_by_feature(table, feature_name="ratio_to_median_purchase_price"):
    # Finding where models disagree most (slice analysis).
    if feature_name not in table.columns:
        return pd.DataFrame()
    table = table.copy()
    table["slice"] = pd.qcut(table[feature_name], 5, duplicates="drop")
    rows = []
    for s in table["slice"].unique():
        sub = table[table["slice"] == s]
        rows.append({
            "slice": str(s),
            "count": len(sub),
            "disagreement_rate": float(sub["disagree"].mean())
        })
    out = pd.DataFrame(rows).sort_values("disagreement_rate", ascending=False)
    return out
