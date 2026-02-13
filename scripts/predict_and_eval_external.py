#!/usr/bin/env python3
"""
scripts/predict_and_eval_external.py

Apply saved models to an evaluation set and compute metrics.
- Classification: binary (active vs inactive), with robust alignment to model.classes_
  and collapse of 'intermediate' -> 'inactive' (0).
- Regression: pIC50 regression.

Supports fingerprint columns named 'FP_*' or 'morgan_*'.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from joblib import load as joblib_load
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


# -------------------- helpers -------------------- #

def map_labels_to_binary(y):
    """
    Map labels to binary ints: inactive=0, active=1.
    - Accepts strings: {'inactive','intermediate','active'}
    - Accepts ints: {0,1,2} with 2 -> 0
    Returns an array of dtype float with NaN for unknowns.
    """
    y = pd.Series(y)
    # string case
    if y.dtype == object:
        mapping = {
            "inactive": 0,
            "active": 1,
            "intermediate": 0,
            "Inactive": 0,
            "Active": 1,
            "Intermediate": 0,
        }
        y2 = y.map(mapping)
    else:
        # numeric case: collapse 2 -> 0; keep 0,1
        def _map_num(v):
            if pd.isna(v):
                return np.nan
            if v == 1:
                return 1
            if v in (0, 2):
                return 0
            return np.nan
        y2 = y.apply(_map_num)

    return y2.astype(float)


def regression_metrics(y_true, y_pred):
    mask = ~pd.isna(y_true)
    if mask.sum() == 0:
        return {}
    return {
        "r2": float(r2_score(y_true[mask], y_pred[mask])),
        "rmse": float(mean_squared_error(y_true[mask], y_pred[mask], squared=False)),
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
    }


def binary_classification_metrics(y_true_bin, y_prob_pos):
    """
    y_true_bin: array of {0,1}
    y_prob_pos: probability for the positive class (1)
    """
    y_true_bin = np.asarray(y_true_bin, dtype=int)
    y_pred_bin = (y_prob_pos >= 0.5).astype(int)

    return {
        "auc": float(roc_auc_score(y_true_bin, y_prob_pos)) if len(np.unique(y_true_bin)) == 2 else None,
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
    }


# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", required=True, help="CSV with fingerprints + labels")
    ap.add_argument("--cls-model", required=True, help="Path to classifier .joblib")
    ap.add_argument("--reg-model", required=True, help="Path to regressor .joblib")
    ap.add_argument("--out-dir", required=True, help="Directory to write metrics and predictions")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.fps)

    # Fingerprints
    fp_cols = [c for c in df.columns if c.startswith("FP_") or c.startswith("morgan_")]
    if not fp_cols:
        raise RuntimeError("No fingerprint columns found (expected 'FP_*' or 'morgan_*').")
    X_all = df[fp_cols].values

    # ---------------- Classification ----------------
    cls_metrics = {}
    if os.path.exists(args.cls_model) and "class" in df.columns:
        cls = joblib_load(args.cls_model)
        model_classes = np.array(cls.classes_)  # e.g., array([0, 1])

        # Map ground-truth labels to binary 0/1
        y_true_bin_all = map_labels_to_binary(df["class"].values)

        # Keep only rows with valid binary labels AND where that label is in model_classes
        valid_mask = ~pd.isna(y_true_bin_all)
        y_true_bin = y_true_bin_all[valid_mask]
        X_cls = X_all[valid_mask]

        if X_cls.shape[0] > 0:
            # Predict probabilities and align to the model's positive class index
            y_prob = cls.predict_proba(X_cls)  # shape (n, 2) expected
            if y_prob.shape[1] != len(model_classes):
                raise RuntimeError(
                    f"Model predict_proba columns ({y_prob.shape[1]}) != model classes ({len(model_classes)})."
                )

            # Determine which column is the "active" (positive=1) class
            # If model trained on numeric 0/1, this finds index of 1. If strings, try 'active'.
            pos_label = None
            # prefer numeric 1 if present
            if any(isinstance(x, (int, np.integer)) for x in model_classes):
                if 1 in model_classes:
                    pos_label = 1
                elif "active" in model_classes:
                    pos_label = "active"
            else:
                # string classes
                if "active" in model_classes:
                    pos_label = "active"
                elif 1 in model_classes:
                    pos_label = 1

            if pos_label is None:
                # fallback: assume the larger/last class is positive
                pos_idx = int(np.argmax(model_classes))
            else:
                matches = np.where(model_classes == pos_label)[0]
                pos_idx = int(matches[0]) if len(matches) else 1  # default to col 1

            y_prob_pos = y_prob[:, pos_idx]

            # Ensure y_true_bin values are {0,1} ints
            y_true_bin = y_true_bin.astype(int)

            # Compute metrics
            cls_metrics = binary_classification_metrics(y_true_bin, y_prob_pos)

            # Save per-row predictions for those valid rows
            out_cls = df.loc[valid_mask, ["smiles", "class"]].copy()
            out_cls["prob_active"] = y_prob_pos
            out_cls["pred_class_bin"] = (y_prob_pos >= 0.5).astype(int)
            # Map binary pred back to strings for convenience
            out_cls["pred_class_str"] = out_cls["pred_class_bin"].map({0: "inactive", 1: "active"})
            out_cls.to_csv(os.path.join(args.out_dir, "pred_cls.csv"), index=False)
            print("Classification predictions saved.")

            with open(os.path.join(args.out_dir, "metrics_cls.json"), "w") as f:
                json.dump(cls_metrics, f, indent=2)
            print("Classification metrics saved.")
        else:
            print("No valid samples for classification after label mapping.")

    # ---------------- Regression ----------------
    reg_metrics = {}
    if os.path.exists(args.reg_model) and "pIC50" in df.columns:
        reg = joblib_load(args.reg_model)
        mask_reg = ~pd.isna(df["pIC50"].values)
        X_reg = X_all[mask_reg]
        y_reg = df.loc[mask_reg, "pIC50"].values

        if X_reg.shape[0] > 0:
            y_pred_reg = reg.predict(X_reg)
            reg_metrics = regression_metrics(y_reg, y_pred_reg)

            out_reg = df.loc[mask_reg, ["smiles", "pIC50"]].copy()
            out_reg["pred_pIC50"] = y_pred_reg
            out_reg.to_csv(os.path.join(args.out_dir, "pred_reg.csv"), index=False)
            print("Regression predictions saved.")

            with open(os.path.join(args.out_dir, "metrics_reg.json"), "w") as f:
                json.dump(reg_metrics, f, indent=2)
            print("Regression metrics saved.")
        else:
            print("No valid samples for regression.")

    # Print summary to console (optional)
    if cls_metrics:
        print("CLS:", cls_metrics)
    if reg_metrics:
        print("REG:", reg_metrics)


if __name__ == "__main__":
    main()
