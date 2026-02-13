#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def morgan_bits(smiles, radius=2, n_bits=2048):
    fps, ok = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            fps.append(list(fp)); ok.append(True)
        else:
            fps.append([0]*n_bits); ok.append(False)
    X = pd.DataFrame(fps, columns=[f"FP_{i}" for i in range(n_bits)])
    return X, pd.Series(ok, name="parsed_ok")

def main():
    ap = argparse.ArgumentParser(description="Build Morgan fingerprints from a given CSV")
    ap.add_argument("--in",  dest="infile", required=True,
                    help="Input CSV (e.g., data/processed/bioactivity_data_3class_pIC50_dedup.csv)")
    ap.add_argument("--out", dest="outfile", default="data/processed/bioactivity_data_descriptors_morgan_dedup.csv",
                    help="Output CSV path (default: data/processed/bioactivity_data_descriptors_morgan_dedup.csv)")
    args = ap.parse_args()

    p = Path(args.infile)
    if not p.exists():
        print(f"Error: not found: {p}", file=sys.stderr); sys.exit(2)

    df = pd.read_csv(p)
    # normalize smiles col
    for c in ("smiles","canonical_smiles","SMILES"):
        if c in df.columns:
            if c != "smiles": df = df.rename(columns={c:"smiles"})
            break
    if "smiles" not in df.columns:
        print("Error: input needs a 'smiles' column.", file=sys.stderr); sys.exit(2)

    X, ok = morgan_bits(df["smiles"])
    out = df.copy()
    out["parsed_ok"] = ok
    out = pd.concat([out, X], axis=1)

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(out)}  cols={out.shape[1]}")

if __name__ == "__main__":
    main()
