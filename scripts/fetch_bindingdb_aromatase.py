#!/usr/bin/env python3
"""
scripts/fetch_bindingdb_aromatase.py

Download the Sept 2025 BindingDB bulk TSV (correct rwd path),
then filter for human aromatase (CYP19A1, UniProt P11511).
"""

import os
import zipfile
import requests
import pandas as pd

# Correct URL with /rwd/
BINDINGDB_URL = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202510_tsv.zip"

OUT_DIR = "data/external/bindingdb"
ZIP_PATH = os.path.join(OUT_DIR, "BindingDB_All_latest.zip")
TSV_PATH = os.path.join(OUT_DIR, "BindingDB_All.tsv")
OUT_CSV = os.path.join(OUT_DIR, "aromatase_bindingdb.csv")

def extract_zip(path, outdir):
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(outdir)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1. Download
    print(f"Downloading BindingDB from {BINDINGDB_URL} ...")
    r = requests.get(BINDINGDB_URL, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download BindingDB (HTTP {r.status_code})")

    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Saved ZIP → {ZIP_PATH}")

    # Step 2. Extract outer zip
    print("Extracting ZIP ...")
    extract_zip(ZIP_PATH, OUT_DIR)

    # Step 3. Handle cases: TSV, nested zip, or year-based TSV
    extracted_file = None

    # If BindingDB_All.tsv is already there
    if os.path.exists(os.path.join(OUT_DIR, "BindingDB_All.tsv")):
        extracted_file = os.path.join(OUT_DIR, "BindingDB_All.tsv")

    # If a nested BindingDB_All.tsv.zip exists, unzip it
    elif os.path.exists(os.path.join(OUT_DIR, "BindingDB_All.tsv.zip")):
        nested_zip = os.path.join(OUT_DIR, "BindingDB_All.tsv.zip")
        print("Found nested BindingDB_All.tsv.zip, extracting...")
        extract_zip(nested_zip, OUT_DIR)
        extracted_file = os.path.join(OUT_DIR, "BindingDB_All.tsv")

    # If a year-stamped file exists
    else:
        for name in os.listdir(OUT_DIR):
            if name.startswith("BindingDB_All_") and name.endswith(".tsv"):
                extracted_file = os.path.join(OUT_DIR, name)
                break

    if not extracted_file or not os.path.exists(extracted_file):
        print("DEBUG: Contents of OUT_DIR:", os.listdir(OUT_DIR))
        raise RuntimeError("Could not find extracted TSV file")

    if extracted_file != TSV_PATH:
        if os.path.exists(TSV_PATH):
            os.remove(TSV_PATH)
        os.rename(extracted_file, TSV_PATH)
    print(f"Standardized {extracted_file} → {TSV_PATH}")

    # Step 4. Filter aromatase
    print("Loading TSV and filtering for aromatase (P11511)...")
    df = pd.read_csv(TSV_PATH, sep="\t", low_memory=False)
    if "UniProt" not in df.columns:
        raise RuntimeError("Missing UniProt column in BindingDB TSV")

    df_aroma = df[df["UniProt"] == "P11511"].copy()
    print(f"Found {len(df_aroma)} aromatase rows.")

    keep_cols = [
        "Ligand SMILES", "Ligand InChI", "BindingDB Reactant_set_id", "PubChem CID",
        "ChEMBL ID of Ligand", "Ki (nM)", "IC50 (nM)", "Kd (nM)", "EC50 (nM)",
        "Target Name", "UniProt", "PMID"
    ]
    df_aroma = df_aroma[[c for c in keep_cols if c in df_aroma.columns]]

    df_aroma.to_csv(OUT_CSV, index=False)
    print(f"Saved aromatase subset → {OUT_CSV}")

if __name__ == "__main__":
    main()
