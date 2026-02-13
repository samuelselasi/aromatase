#!/usr/bin/env python3
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Input holdout CSV with class labels")
    ap.add_argument("--outfile", required=True, help="Output CSV with collapsed binary labels")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    if "class" not in df.columns:
        raise RuntimeError("No 'class' column found in dataset")

    # Collapse: active = keep as active; intermediate → inactive
    df["class"] = df["class"].replace({"intermediate": "inactive"})

    df.to_csv(args.outfile, index=False)
    print(f"Collapsed labels saved to {args.outfile}")
    print(df["class"].value_counts())

if __name__ == "__main__":
    main()
