import matplotlib.pyplot as plt

def plot_score_hist(proba_A, proba_B, path):
    plt.hist(proba_A, bins=50, alpha=0.5, label="Model A")
    plt.hist(proba_B, bins=50, alpha=0.5, label="Model B")
    plt.legend()
    plt.title("Score Distribution")
    plt.savefig(path)
    plt.close()

def plot_disagreement_by_feature(df, path):
    if df is not None and not df.empty:
        top = df.head(5)
        plt.barh(top["slice"], top["disagreement_rate"])
        plt.title("Top Disagreement Slices")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()