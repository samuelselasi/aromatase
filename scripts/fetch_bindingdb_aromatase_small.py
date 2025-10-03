#!/usr/bin/env python3
"""
scripts/fetch_bindingdb_aromatase_small.py

Fetch ONLY the BindingDB aromatase dataset (CYP19A1, UniProt P11511),
directly as TSV (small and clean).
"""

import os
import requests
import pandas as pd

URL = "https://www.bindingdb.org/bind/chemsearch/marvin/DownloadSubstanceServlet?target=P11511&download_type=TSV"
OUT_DIR = "data/external/bindingdb"
TSV_PATH = os.path.join(OUT_DIR, "aromatase_bindingdb.tsv")
CSV_PATH = os.path.join(OUT_DIR, "aromatase_bindingdb_small.csv")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1. Download TSV
    print(f"Downloading aromatase dataset from {URL} ...")
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download dataset (HTTP {r.status_code})")

    with open(TSV_PATH, "wb") as f:
        f.write(r.content)
    print(f"Saved TSV → {TSV_PATH}")

    # Step 2. Load into DataFrame
    df = pd.read_csv(TSV_PATH, sep="\t", low_memory=False)
    print(f"Loaded {len(df)} rows from BindingDB (aromatase).")

    # Step 3. Keep useful columns
    keep_cols = [
        "Ligand SMILES", "Ligand InChI", "PubChem CID", "ChEMBL ID of Ligand",
        "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)", "Target Name", "UniProt", "PMID"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Step 4. Save as CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved small aromatase CSV → {CSV_PATH}")

if __name__ == "__main__":
    main()
