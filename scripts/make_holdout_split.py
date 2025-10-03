#!/usr/bin/env python3
"""
scripts/make_holdout_split.py

Split deduplicated AfroDB dataset into train/holdout sets.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/processed/bioactivity_data_3class_pIC50_dedup.csv")
    ap.add_argument("--train-out", default="data/processed/train.csv")
    ap.add_argument("--holdout-out", default="data/processed/holdout.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=1337)
    args = ap.parse_args()

    if not os.path.exists(args.infile):
        raise FileNotFoundError(f"Input file not found: {args.infile}")

    df = pd.read_csv(args.infile)

    if "class" not in df.columns:
        raise RuntimeError("Expected column 'class' not found in dataset")

    train_df, holdout_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["class"]
    )

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    train_df.to_csv(args.train_out, index=False)
    holdout_df.to_csv(args.holdout_out, index=False)

    print(f"Train set: {len(train_df)} rows → {args.train_out}")
    print(f"Holdout set: {len(holdout_df)} rows → {args.holdout_out}")


if __name__ == "__main__":
    main()

