#!/usr/bin/env python3
"""
scripts/ad_report.py

Compute a simple applicability domain report by nearest-neighbor
Tanimoto similarities between training FPS and external FPS (bit vectors).

Inputs:
  - Training FPS CSV (e.g., data/processed/bioactivity_data_descriptors_morgan.csv)
  - External FPS CSV (e.g., data/external/external_fps_morgan.csv)

Outputs:
  - Prints summary stats of NN Tanimoto similarities
  - Writes results/external/ad_report.json with summary
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

def tanimoto_bit(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are 0/1 bit arrays
    inter = np.bitwise_and(a, b).sum()
    denom = a.sum() + b.sum() - inter
    return float(inter / denom) if denom > 0 else 0.0

def nn_tanimoto(train_bits: np.ndarray, ext_bits: np.ndarray, batch=256):
    # For each external bitvector, find nearest-neighbor similarity in train_bits
    sims = []
    for i in range(0, len(ext_bits), batch):
        chunk = ext_bits[i:i+batch]
        # brute-force; fine for few thousands
        for x in chunk:
            # vectorized tanimoto over train set
            inter = np.bitwise_and(train_bits, x).sum(axis=1)
            denom = train_bits.sum(axis=1) + x.sum() - inter
            s = np.where(denom > 0, inter / denom, 0.0)
            sims.append(float(s.max()))
    return np.array(sims, dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-fps", default="data/processed/bioactivity_data_descriptors_morgan.csv")
    ap.add_argument("--ext-fps", default="data/external/external_fps_morgan.csv")
    ap.add_argument("--out", default="results/external/ad_report.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df_tr = pd.read_csv(args.train_fps)
    df_ex = pd.read_csv(args.ext_fps)

    Xtr = df_tr.filter(like="morgan_").values.astype(np.uint8)
    Xex = df_ex.filter(like="morgan_").values.astype(np.uint8)

    sims = nn_tanimoto(Xtr, Xex)
    summary = {
        "n_external": int(len(sims)),
        "nn_tanimoto_mean": float(np.mean(sims)),
        "nn_tanimoto_median": float(np.median(sims)),
        "nn_tanimoto_p10": float(np.percentile(sims, 10)),
        "nn_tanimoto_p90": float(np.percentile(sims, 90)),
    }

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("Applicability Domain (NN Tanimoto) summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

if __name__ == "__main__":
    main()
