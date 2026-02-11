import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_roc(csv_path: str):
    """Plot ROC curve from a CSV file with columns: fpr,tpr,thresholds."""
    df = pd.read_csv(csv_path)

    fpr = df["fpr"].values
    tpr = df["tpr"].values

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=os.path.basename(csv_path), color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", lw=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (no smoothing)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(here, "model_trial_0_roc_quick.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Default to model_trial_0_roc_data.csv in this directory
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(here, "model_trial_0_roc_data.csv")

    if not os.path.exists(default_csv):
        raise FileNotFoundError(f"Default ROC CSV not found: {default_csv}")

    plot_roc(default_csv)

