#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path

PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(PROC/"bioactivty_data_preprocessed_class.csv").copy()

    # Clean values: valid IC50 must be > 0
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    before = len(df)
    df = df[df["standard_value"] > 0].copy()
    dropped = before - len(df)

    # Convert nM -> M and compute pIC50
    molar = df["standard_value"] * 1e-9
    df["pIC50"] = -np.log10(molar)

    out = PROC/"bioactivity_data_3class_pIC50.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(df)}  (dropped {dropped} invalid IC50 rows)")

if __name__ == "__main__":
    main()
