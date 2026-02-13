#!/usr/bin/env python3
"""
scripts/prepare_external_validation.py

Prepare external dataset for validation:
 - Canonicalize SMILES
 - Convert activity values to pIC50 if numeric
 - Assign binary labels (active/inactive)
 - Remove overlap with training dataset
"""

import os
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem

def canonical_smiles(smi: str):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None

def to_pIC50(val, units="nM"):
    try:
        x = float(val)
        if x <= 0:
            return np.nan
        if units.lower() in ["nm", "nanomolar", "nM"]:
            return 9.0 - np.log10(x)  # pIC50 = 9 - log10(IC50 in nM)
        elif units.lower() in ["um", "micromolar"]:
            return 6.0 - np.log10(x)  # convert μM to nM first
        else:
            return np.nan
    except:
        return np.nan

def load_training_smiles(training_csv):
    if not os.path.exists(training_csv):
        return set()
    tr = pd.read_csv(training_csv)
    smi_col = [c for c in tr.columns if "smiles" in c.lower()]
    if not smi_col:
        return set()
    tr["can_smi"] = tr[smi_col[0]].astype(str).apply(canonical_smiles)
    return set(tr["can_smi"].dropna().unique())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="External CSV")
    ap.add_argument("--training-csv", default="data/processed/bioactivity_data_3class_pIC50.csv")
    ap.add_argument("--ic50-threshold-nm", type=float, default=1000.0, help="Active if IC50 <= threshold (default 1000 nM)")
    ap.add_argument("--out", default="data/external/external_clean.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    smi_col = [c for c in df.columns if "smiles" in c.lower()]
    if not smi_col:
        raise RuntimeError("Could not find a SMILES column")
    smi_col = smi_col[0]

    df["can_smi"] = df[smi_col].astype(str).apply(canonical_smiles)
    df = df.dropna(subset=["can_smi"]).drop_duplicates("can_smi")

    # Try to use pChEMBL if available, otherwise compute from standard_value
    if "pchembl_value" in df.columns:
        df["pIC50"] = df["pchembl_value"]
    elif "standard_value" in df.columns and "standard_units" in df.columns:
        df["pIC50"] = [
            to_pIC50(v, u) for v, u in zip(df["standard_value"], df["standard_units"])
        ]
    else:
        df["pIC50"] = np.nan

    # Binary label
    df["is_active"] = (df["pIC50"] >= 6).astype(int)  # threshold ~ 1 μM

    # Remove overlap with training data
    train_can = load_training_smiles(args.training_csv)
    before = len(df)
    df = df[~df["can_smi"].isin(train_can)]
    after = len(df)
    print(f"Removed {before - after} overlapping compounds.")

    df.to_csv(args.out, index=False)
    print(f"Saved cleaned external dataset with {len(df)} molecules → {args.out}")

if __name__ == "__main__":
    main()
