#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
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
    df = df.dropna(subset=["pIC50"]).copy()
    y = df["pIC50"].astype(float).values
    fp_cols = [c for c in df.columns if c.startswith("FP_")]
    X = df[fp_cols].astype(np.uint8).values
    return df, X, y, fp_cols

def eval_and_save(name, model, X_test, y_test, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    with open(RESULTS / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # scatter: predicted vs actual
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, s=8, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims)  # y=x
    plt.xlabel("Actual pIC50"); plt.ylabel("Predicted pIC50"); plt.title(f"Pred vs Actual — {name}")
    plt.savefig(FIGS / f"{name}_scatter.png", bbox_inches="tight")
    plt.close()

    dump(model, MODELS / f"{name}.joblib")
    print(f"[{name}] metrics:", metrics)

def main():
    df, X, y, fp_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model 1: Ridge (fast baseline)
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    eval_and_save("reg_ridge", ridge, X_test, y_test, y_pred_ridge)

    # Model 2: Random Forest Regressor
    rfr = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1
    )
    rfr.fit(X_train, y_train)
    y_pred_rfr = rfr.predict(X_test)
    eval_and_save("reg_random_forest", rfr, X_test, y_test, y_pred_rfr)

    # Save top 50 features by importance (RF)
    importances = pd.Series(rfr.feature_importances_, index=fp_cols).sort_values(ascending=False)
    importances.head(50).to_csv(TABLES / "reg_random_forest_top50_features.csv")

if __name__ == "__main__":
    main()
