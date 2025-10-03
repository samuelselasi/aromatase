#!/usr/bin/env python3
"""
scripts/fetch_aromatase_chembl.py

Fetch up to 200 strictly filtered aromatase (CYP19A1, CHEMBL220) bioactivity
rows from ChEMBL using the official client. Filters:
 - target_chembl_id = CHEMBL220
 - assay_type = 'B' (binding)
 - target_organism = 'Homo sapiens'
 - standard_type in {'IC50','Ki'}
 - standard_units = 'nM'
 - standard_relation = '='
 - pchembl_value not null

This produces a small, clean external validation dataset far closer in label
semantics to your training labels.
"""

import os
import random
import pandas as pd
from chembl_webresource_client.new_client import new_client

OUT_DIR = "data/external/chembl"
OUT_CSV = os.path.join(OUT_DIR, "aromatase_chembl_external.csv")
N_MAX = 200  # cap external set size

KEEP_COLS = [
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "assay_type",
    "target_organism",
    "assay_chembl_id",
]

def fetch_filtered(limit_soft=3000):
    activity = new_client.activity
    res = activity.filter(
        target_chembl_id="CHEMBL220",
        assay_type="B",                     # Binding assays
        standard_units="nM",
        standard_relation="=",
        target_organism="Homo sapiens",
        pchembl_value__isnull=False
    ).only(KEEP_COLS)[:limit_soft]

    df = pd.DataFrame(res)
    if df.empty:
        raise RuntimeError("ChEMBL returned no rows for the strict filter.")

    # Keep only IC50/Ki
    df = df[df["standard_type"].isin(["IC50", "Ki"])].copy()

    # Drop missing or empty SMILES
    df = df[df["canonical_smiles"].notna() & (df["canonical_smiles"].str.len() > 0)]
    df = df.drop_duplicates(subset=["canonical_smiles"])

    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = fetch_filtered(limit_soft=3000)

    # Sample to N_MAX if larger
    if len(df) > N_MAX:
        df = df.sample(n=N_MAX, random_state=1337).reset_index(drop=True)

    print(f"Final external (filtered) size: {len(df)}")
    df = df[KEEP_COLS]  # keep stable schema
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV → {OUT_CSV}")

if __name__ == "__main__":
    main()
