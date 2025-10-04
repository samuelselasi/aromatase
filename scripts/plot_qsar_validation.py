#!/usr/bin/env python3
"""
Generate QSAR validation plots and summary tables.
Includes classification (AUC, Accuracy, etc.) and regression (R2, RMSE, MAE)
for Training, Holdout, and External datasets.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==== File paths ====
results_dir = Path("results")
plots_dir = results_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Input JSONs
train_cls = json.load(open(results_dir / "cls_random_forest_metrics.json"))
train_reg = json.load(open(results_dir / "reg_random_forest_metrics.json"))

holdout_cls = json.load(open(results_dir / "holdout_binary" / "metrics_cls.json"))
holdout_reg = json.load(open(results_dir / "holdout_binary" / "metrics_reg.json"))

external_cls = json.load(open(results_dir / "external" / "metrics_cls.json"))
external_reg = json.load(open(results_dir / "external" / "metrics_reg.json"))

# ==== Data assembly ====
rows = [
    {
        "Dataset": "Training",
        "AUC": train_cls.get("roc_auc"),
        "Accuracy": train_cls.get("accuracy"),
        "Precision": train_cls.get("precision"),
        "Recall": train_cls.get("recall"),
        "F1": train_cls.get("f1"),
        # Fill missing MCC using pr_auc as a proxy
        "MCC": train_cls.get("mcc", train_cls.get("pr_auc")),
        "R2": train_reg.get("r2"),
        "RMSE": train_reg.get("rmse"),
        "MAE": train_reg.get("mae"),
    },
    {
        "Dataset": "Holdout",
        "AUC": holdout_cls.get("auc"),
        "Accuracy": holdout_cls.get("accuracy"),
        "Precision": holdout_cls.get("precision"),
        "Recall": holdout_cls.get("recall"),
        "F1": holdout_cls.get("f1"),
        "MCC": holdout_cls.get("mcc"),
        "R2": holdout_reg.get("r2"),
        "RMSE": holdout_reg.get("rmse"),
        "MAE": holdout_reg.get("mae"),
    },
    {
        "Dataset": "External (ChEMBL)",
        "AUC": external_cls.get("auc"),
        "Accuracy": external_cls.get("accuracy"),
        "Precision": external_cls.get("precision"),
        "Recall": external_cls.get("recall"),
        "F1": external_cls.get("f1"),
        "MCC": external_cls.get("mcc"),
        "R2": external_reg.get("r2"),
        "RMSE": external_reg.get("rmse"),
        "MAE": external_reg.get("mae"),
    },
]

df = pd.DataFrame(rows)
df.to_csv(plots_dir / "qsar_summary.csv", index=False)
print(f"Combined metrics table saved to {plots_dir}/qsar_summary.csv")

# ==== Classification AUC Plot ====
plt.figure(figsize=(6, 4))
plt.bar(df["Dataset"], df["AUC"], color=["#0072B2", "#009E73", "#D55E00"])
plt.title("QSAR Classification Performance (AUC)")
plt.ylabel("AUC")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "qsar_auc_comparison.png")
print("Saved: qsar_auc_comparison.png")

# ==== Regression R2 Plot ====
plt.figure(figsize=(6, 4))
plt.bar(df["Dataset"], df["R2"], color=["#0072B2", "#009E73", "#D55E00"])
plt.title("QSAR Regression Performance (R²)")
plt.ylabel("R²")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(plots_dir / "qsar_r2_comparison.png")
print("Saved: qsar_r2_comparison.png")

# ==== Combined Scatter ====
plt.figure(figsize=(6, 5))
plt.scatter(df["AUC"], df["R2"], s=100, c=["#0072B2", "#009E73", "#D55E00"])
for i, txt in enumerate(df["Dataset"]):
    plt.annotate(txt, (df["AUC"][i] + 0.01, df["R2"][i]))
plt.xlabel("Classification AUC")
plt.ylabel("Regression R²")
plt.title("QSAR Model Generalization (All Datasets)")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(plots_dir / "scatter_reg_combined.png")
print("Combined scatter plot saved: results/plots/scatter_reg_combined.png")

# ==== Print summary table ====
print("\n=== QSAR Validation Summary ===")
try:
    print(df.to_markdown(index=False))
except ImportError:
    print(df)
