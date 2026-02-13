#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def label(v: float) -> str:
    v = float(v)
    if v >= 10000: return "inactive"
    if v <= 1000:  return "active"
    return "intermediate"

def main():
    df = pd.read_csv(RAW/"bioactivty_data.csv")
    df["class"] = df["standard_value"].apply(label)
    df.to_csv(PROC/"bioactivty_data_preprocessed_class.csv", index=False)
    df[df["class"]!="intermediate"].to_csv(PROC/"bioactivity_data_no_intermediate.csv", index=False)
    print("Wrote processed 3-class and 2-class CSVs.")

if __name__ == "__main__":
    main()
