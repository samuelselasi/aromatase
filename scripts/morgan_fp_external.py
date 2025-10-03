#!/usr/bin/env python3
"""
scripts/morgan_fp_external.py

Generate Morgan fingerprints for external validation set.
"""

import os
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def fp_to_bits(fp, n_bits):
    arr = np.zeros((n_bits,), dtype=int)
    for idx in fp.GetOnBits():
        arr[idx] = 1
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="Cleaned external CSV")
    ap.add_argument("--out", dest="outfile", default="data/external/external_fps_morgan.csv")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--n-bits", type=int, default=2048)
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    fps = []
    idx = []
    for i, smi in enumerate(df["can_smi"].astype(str)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, args.n_bits)
        fps.append(fp_to_bits(fp, args.n_bits))
        idx.append(i)

    X = np.array(fps, dtype=int)
    Xdf = pd.DataFrame(X, columns=[f"morgan_{j}" for j in range(X.shape[1])])
    out = pd.concat([df.iloc[idx].reset_index(drop=True), Xdf], axis=1)
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    out.to_csv(args.outfile, index=False)
    print(f"Saved {args.outfile} with {len(out)} rows and {X.shape[1]} bits.")

if __name__ == "__main__":
    main()
