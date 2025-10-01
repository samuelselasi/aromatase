#!/usr/bin/env python3
from pathlib import Path
import argparse, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

# optional SMOTE
def maybe_make_sampler(use_smote):
    if not use_smote: return None
    try:
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        return Pipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=42)),
            ("clf", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1))
        ])
    except Exception as e:
        raise SystemExit(f"Install imbalanced-learn to use --smote: pip install imbalanced-learn\n{e}")

def main():
    ap = argparse.ArgumentParser(description="Train RF classifier on deduplicated features.")
    ap.add_argument("--fps", default="data/processed/bioactivity_data_descriptors_morgan_dedup.csv",
                    help="Feature CSV with FP_* columns and 'class'")
    ap.add_argument("--out", default="results/models/cls_random_forest_dedup.joblib",
                    help="Output model path")
    ap.add_argument("--smote", action="store_true", help="Use SMOTE oversampling instead of class_weight")
    args = ap.parse_args()

    p = Path(args.fps)
    df = pd.read_csv(p)

    # keep only active/inactive for classification
    df = df[df["class"].isin(["active","inactive"])].copy()
    y = (df["class"] == "active").astype(int).values
    X = df[[c for c in df.columns if c.startswith("FP_")]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if args.smote:
        model = maybe_make_sampler(use_smote=True)
        model.fit(X_train, y_train)
        # extract inner RF for saving? we just save the full pipeline:
        to_save = model
        # get predicted probs for metrics:
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        clf = RandomForestClassifier(
            n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        to_save = clf
        y_prob = clf.predict_proba(X_test)[:,1]

    y_hat = (y_prob >= 0.5).astype(int)
    metrics = {
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "pos_ratio_train": float(y_train.mean()),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "balanced_acc": float(balanced_accuracy_score(y_test, y_hat)),
        "f1": float(f1_score(y_test, y_hat)),
    }
    print(metrics)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(to_save, args.out)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
