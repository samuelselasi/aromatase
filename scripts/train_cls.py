#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

ROOT = Path(".")
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = RESULTS / "models"
TABLES = RESULTS / "tables"
for d in [RESULTS, FIGS, MODELS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

CSV = DATA / "bioactivity_data_descriptors_morgan.csv"

def load_data(csv=CSV):
    df = pd.read_csv(csv)
    # classification uses only active/inactive
    df = df[df["class"].isin(["active", "inactive"])].copy()
    y = df["class"].map({"inactive": 0, "active": 1}).astype(int).values
    fp_cols = [c for c in df.columns if c.startswith("FP_")]
    X = df[fp_cols].astype(np.uint8).values  # binary features, compact
    return df, X, y, fp_cols

def eval_and_save(name, model, X_train, X_test, y_train, y_test, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
    }
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=["true_inactive","true_active"], columns=["pred_inactive","pred_active"])\
      .to_csv(TABLES / f"{name}_confusion_matrix.csv", index=True)

    # report
    report = classification_report(y_test, y_pred, target_names=["inactive","active"], zero_division=0, output_dict=True)
    pd.DataFrame(report).to_csv(TABLES / f"{name}_classification_report.csv")

    # ROC curve
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC — {name}")
    plt.savefig(FIGS / f"{name}_roc.png", bbox_inches="tight")
    plt.close()

    # PR curve
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Precision–Recall — {name}")
    plt.savefig(FIGS / f"{name}_pr.png", bbox_inches="tight")
    plt.close()

    # save metrics json
    with open(RESULTS / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # save model
    dump(model, MODELS / f"{name}.joblib")
    print(f"[{name}] metrics:", metrics)

def main():
    df, X, y, fp_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model 1: Logistic Regression (balanced)
    logreg = LogisticRegression(
        max_iter=2000, solver="liblinear", class_weight="balanced"
    )
    logreg.fit(X_train, y_train)
    y_prob_lr = logreg.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    eval_and_save("cls_logreg", logreg, X_train, X_test, y_train, y_test, y_pred_lr, y_prob_lr)

    # Model 2: Random Forest (balanced)
    rf = RandomForestClassifier(
        n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)
    eval_and_save("cls_random_forest", rf, X_train, X_test, y_train, y_test, y_pred_rf, y_prob_rf)

    # Feature importances (RF): top 50
    importances = pd.Series(rf.feature_importances_, index=fp_cols).sort_values(ascending=False)
    importances.head(50).to_csv(TABLES / "cls_random_forest_top50_features.csv")

if __name__ == "__main__":
    main()
