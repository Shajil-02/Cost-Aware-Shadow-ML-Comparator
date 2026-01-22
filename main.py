from pathlib import Path
from src.data_loader import split_data, load_data
from src.train_model import train_model_A, train_model_B, evaluate_auc, save_model
from src.shadow_compare import compare_models_budget, disagreement_by_feature
from src.plots import plot_score_hist, plot_disagreement_by_feature

def make_dirs(processed_dir, models_dir, report_dir, plots_dir):
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

def write_report(summary, path):
    lines = []
    lines.append("SHADOW DEPLOYMENT REPORT (SAME REVIEW BUDGET)")
    lines.append("------------------------------------------")
    for k, v in summary.items():
        lines.append(f"{k} : {v}")
    path.write_text("\n".join(lines))

def main():
    BASE_DIR = Path(__file__).parent
    RAW_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    MODELS_DIR = BASE_DIR / "models"
    REPORT_DIR = BASE_DIR / "outputs" / "reports"
    PLOTS_DIR = BASE_DIR / "outputs" / "plots"
    TRAIN_PATH = PROCESSED_DIR / "train.csv"
    VAL_PATH = PROCESSED_DIR / "val.csv"
    TEST_PATH = PROCESSED_DIR / "test.csv"
    make_dirs(PROCESSED_DIR, MODELS_DIR, REPORT_DIR, PLOTS_DIR)

    # 1) Split once
    if not TRAIN_PATH.exists():
        split_data(RAW_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH)

    # 2) Load data
    X_train, y_train = load_data(TRAIN_PATH)
    X_val, y_val = load_data(VAL_PATH)
    X_test, y_test = load_data(TEST_PATH)
    print(f"Fraud rate in test: {y_test.mean()*100:.2f}%")

    # 3) Train models
    model_A = train_model_A(X_train, y_train)
    model_B = train_model_B(X_train, y_train)
    save_model(model_A, MODELS_DIR / "model_A.pkl")
    save_model(model_B, MODELS_DIR / "model_B.pkl")

    # 4) Validation AUC (just for info)
    auc_A, _ = evaluate_auc(model_A, X_val, y_val)
    auc_B, _ = evaluate_auc(model_B, X_val, y_val)
    print("\nValidation check")
    print(f"Model A AUC: {auc_A:.4f}")
    print(f"Model B AUC: {auc_B:.4f}")

    # 5) Shadow compare on test (same review budget)
    test_proba_A = model_A.predict_proba(X_test)[:, 1]
    test_proba_B = model_B.predict_proba(X_test)[:, 1]

    k_frac = 0.02  # top 2% review budget
    summary, table = compare_models_budget(X_test, y_test, test_proba_A, test_proba_B, k_frac=k_frac)

    print("\n=== SHADOW DEPLOYMENT SUMMARY (SAME REVIEW BUDGET) ===\n")
    print(f"Review Budget: {summary['review_budget']}")
    print(f"Flagged count A: {summary['flagged_count_A']} | Flagged count B: {summary['flagged_count_B']}")
    print(f"Budget Threshold A (ref): {summary['budget_threshold_A_ref']}")
    print(f"Budget Threshold B (ref): {summary['budget_threshold_B_ref']}")
    print(f"\nDisagreement rate: {summary['disagreement_rate']}")
    print(f"In {summary['disagreement_rate']} of transactions, Model B behaves differently than Model A.")
    print(f"\nFalse Positives: A={summary['False Positives_A']} | B={summary['False Positives_B']}")
    print(f"False Negatives: A={summary['False Negatives_A']} | B={summary['False Negatives_B']}")
    print("Business Cost (FN=10, FP=1):")
    print(f"A={summary['cost_A']} | B={summary['cost_B']}")
    diff = int(summary["cost_B"]) - int(summary["cost_A"])
    print(f"Cost delta (B - A): {diff}")

    if int(summary["cost_B"]) < int(summary["cost_A"]):
        print("\nDecision: ✅ Recommend Model B (lower cost)")
    else:
        print("\nDecision: ❌ Keep Model A (lower cost)")

    # 6) Save report + table
    table.to_csv(REPORT_DIR / "comparison_table.csv", index=False)
    write_report(summary, REPORT_DIR / "shadow_report.txt")

    # 7) Slice disagreement analysis
    by_feature = disagreement_by_feature(table)
    by_feature.to_csv(REPORT_DIR / "disagreement_slices.csv", index=False)
    if not by_feature.empty:
        top = by_feature.iloc[0]
        print(f"\nTop disagreement slice: {top['slice']} | disagreement={top['disagreement_rate']:.4f}")

    # 8) Plots
    plot_score_hist(test_proba_A, test_proba_B, PLOTS_DIR / "score_hist.png")
    plot_disagreement_by_feature(by_feature, PLOTS_DIR / "disagreement_slices.png")
    print("\nDone. Check outputs/reports/ and outputs/plots/")

if __name__ == "__main__":
    main()
