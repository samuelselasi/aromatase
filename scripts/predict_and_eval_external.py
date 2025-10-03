#!/usr/bin/env python3
"""
scripts/predict_and_eval_external.py

Apply saved models to external dataset and compute metrics.
Handles NaN values in regression targets gracefully.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from joblib import load as joblib_load


def classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def regression_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", required=True)
    ap.add_argument("--cls-model", default="results/models/cls_random_forest_dedup.joblib")
    ap.add_argument("--reg-model", default="results/models/reg_random_forest.joblib")
    ap.add_argument("--out-dir", default="results/external")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.fps)
    X = df.filter(like="morgan_").values

    # -------- Classification --------
    if os.path.exists(args.cls_model):
        cls = joblib_load(args.cls_model)
        y_prob = cls.predict_proba(X)[:, 1]
        df["prob_active"] = y_prob
        if "is_active" in df.columns:
            metrics = classification_metrics(df["is_active"].values, y_prob)
            with open(os.path.join(args.out_dir, "metrics_cls.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print("Classification metrics saved.")
        df.to_csv(os.path.join(args.out_dir, "pred_external_cls.csv"), index=False)
        print("Classification predictions saved.")

    # -------- Regression --------
    if os.path.exists(args.reg_model):
        reg = joblib_load(args.reg_model)
        y_pred = reg.predict(X)
        df["pred_pIC50"] = y_pred

        if "pIC50" in df.columns:
            # Drop NaN values before scoring
            mask = ~df["pIC50"].isna()
            if mask.sum() > 0:
                metrics = regression_metrics(
                    df.loc[mask, "pIC50"].values,
                    df.loc[mask, "pred_pIC50"].values,
                )
                with open(os.path.join(args.out_dir, "metrics_reg.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                print("Regression metrics saved.")
            else:
                print("No valid pIC50 values found. Skipping regression metrics.")
        df.to_csv(os.path.join(args.out_dir, "pred_external_reg.csv"), index=False)
        print("Regression predictions saved.")


if __name__ == "__main__":
    main()
