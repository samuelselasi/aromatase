#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, subprocess, re, unicodedata
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Try Meeko; if missing, we'll use Open Babel as a fallback
try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    USE_MEEKO = True
except Exception:
    USE_MEEKO = False

DATA = Path("data/processed")
SDF_DIR = DATA / "ligands_sdf";  SDF_DIR.mkdir(parents=True, exist_ok=True)
PQT_DIR = DATA / "ligands_pdbqt"; PQT_DIR.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    df = pd.read_csv(path)
    for col in ("smiles","canonical_smiles","SMILES"):
        if col in df.columns:
            df = df.rename(columns={col:"smiles"})
            break
    if "smiles" not in df.columns:
        raise SystemExit(f"CSV must contain a 'smiles' or 'canonical_smiles' column: {path}")
    if "id" not in df.columns:
        df["id"] = df.get("molecule_chembl_id", df.index.astype(str))
    cols = ["id","smiles"] + (["class"] if "class" in df.columns else [])
    return df[cols]

def read_smi(path):
    rows=[]
    with open(path) as fh:
        for i,line in enumerate(fh):
            line=line.strip()
            if not line or line.startswith("#"): 
                continue
            toks=line.split()
            smi=toks[0]
            mid=toks[1] if len(toks)>1 else f"mol{i}"
            rows.append({"id":mid,"smiles":smi})
    return pd.DataFrame(rows, columns=["id","smiles"])

def build_3d(mol):
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3(); params.randomSeed = 42
    if AllChem.EmbedMolecule(m, params=params) != 0:
        return None
    AllChem.UFFOptimizeMolecule(m, maxIters=200)
    return m

def sanitize_id(s: str) -> str:
    # Normalize and keep only safe chars: letters, digits, _, -, .
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\.-]+", "_", s)     # replace anything not [A-Za-z0-9_.-]
    s = s.strip("._-")
    return s or "mol"

def to_pdbqt_with_meeko(mol, out_path):
    preparator = MoleculePreparation()
    preparator.prepare(mol)              # populates internal state
    writer = PDBQTWriterLegacy()
    pdbqt = writer.write_string(preparator)
    with open(out_path, "w") as f:
        f.write(pdbqt)

def to_pdbqt_with_obabel(tmp_sdf_path, out_path, ph=7.4):
    # No -h together with -p to avoid the Open Babel warning; -p adds hydrogens at the given pH.
    cmd = ["obabel", str(tmp_sdf_path), "-O", str(out_path), "-p", str(ph), "--partialcharge", "gasteiger"]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with smiles/canonical_smiles")
    ap.add_argument("--smi", help=".smi with 'SMILES [optional_id]' per line")
    ap.add_argument("--subset", choices=["active","inactive","all"], default="all")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--ph", type=float, default=7.4)
    args = ap.parse_args()

    if args.smi:
        df = read_smi(args.smi)
    elif args.csv:
        df = read_csv(args.csv)
        if args.subset != "all" and "class" in df.columns:
            df = df[df["class"]==args.subset]
    else:
        raise SystemExit("Provide --smi or --csv")

    df = df.drop_duplicates(subset=["smiles"]).head(args.limit).copy()
    if df.empty:
        raise SystemExit("No ligands selected.")

    used = set()
    ok = 0
    for _, row in df.iterrows():
        smi, raw_id = row["smiles"], str(row["id"])
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
        mol3d = build_3d(mol)
        if not mol3d:
            continue

        # Safe, unique filename stem
        stem = sanitize_id(raw_id)
        base = stem
        k = 1
        while base in used:
            base = f"{stem}_{k}"
            k += 1
        used.add(base)

        sdf_path = SDF_DIR / f"{base}.sdf"
        w = Chem.SDWriter(str(sdf_path)); w.write(mol3d); w.close()

        pq_path = PQT_DIR / f"{base}.pdbqt"
        try:
            if USE_MEEKO:
                to_pdbqt_with_meeko(mol3d, pq_path)
            else:
                to_pdbqt_with_obabel(sdf_path, pq_path, ph=args.ph)
            ok += 1
        except Exception:
            # fallback to OBabel if Meeko path fails
            to_pdbqt_with_obabel(sdf_path, pq_path, ph=args.ph)
            ok += 1

    print(f"Prepared {ok} ligands → {PQT_DIR}")

if __name__ == "__main__":
    main()
