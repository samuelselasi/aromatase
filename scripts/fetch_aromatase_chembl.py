#!/usr/bin/env python3
"""
scripts/fetch_aromatase_chembl.py

Fetch up to 200 aromatase (CYP19A1, CHEMBL220) bioactivity records from ChEMBL.
Uses the official ChEMBL webresource client instead of raw HTTP for reliability.
"""

import os
import pandas as pd
from chembl_webresource_client.new_client import new_client

OUT_DIR = "data/external/chembl"
OUT_CSV = os.path.join(OUT_DIR, "aromatase_chembl_external.csv")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Get activities for aromatase (CHEMBL220)
    activity = new_client.activity
    res = activity.filter(target_chembl_id="CHEMBL220").only(
        ["canonical_smiles", "standard_type", "standard_relation",
         "standard_value", "standard_units", "pchembl_value", "assay_chembl_id"]
    )[:200]  # cap at 200

    df = pd.DataFrame(res)
    if df.empty:
        raise RuntimeError("No data returned from ChEMBL. Try increasing the limit or checking CHEMBL220 availability.")

    df.to_csv(OUT_CSV, index=False)
    print(f"Fetched {len(df)} aromatase bioactivity rows → {OUT_CSV}")

if __name__ == "__main__":
    main()
