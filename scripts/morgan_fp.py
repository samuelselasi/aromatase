#!/usr/bin/env python3
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

def morgan(smiles, radius=2, n_bits=2048):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            rows.append(list(fp))
        else:
            rows.append([0]*n_bits)
    return pd.DataFrame(rows, columns=[f"FP_{i}" for i in range(n_bits)])

def main():
    df = pd.read_csv(PROC/"bioactivity_data_3class_pIC50.csv")
    fps = morgan(df["canonical_smiles"])
    dataset = pd.concat([df[["molecule_chembl_id","class","pIC50"]].reset_index(drop=True), fps], axis=1)
    out = PROC/"bioactivity_data_descriptors_morgan.csv"
    dataset.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(dataset)}  cols={dataset.shape[1]}")

if __name__ == "__main__":
    main()
