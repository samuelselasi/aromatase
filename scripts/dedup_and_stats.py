#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
import pandas as pd

# RDKit for canonical SMILES
from rdkit import Chem

ROOT = Path(".")
OUT_TAB = ROOT / "results" / "tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize SMILES column name
    for c in ("smiles", "canonical_smiles", "SMILES"):
        if c in df.columns:
            if c != "smiles":
                df = df.rename(columns={c: "smiles"})
            break
    if "smiles" not in df.columns:
        raise SystemExit(f"Input CSV must have a 'smiles' column. Got: {list(df.columns)}")
    return df

def to_canonical(smi: str) -> str | None:
    try:
        m = Chem.MolFromSmiles(smi)
        if not m:
            return None
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None

def ensure_binary_labels(df: pd.DataFrame, thr_active=6.0, thr_inactive=5.0) -> pd.DataFrame:
    # If a 'class' column already exists, keep it.
    if "class" in df.columns:
        return df
    # Otherwise infer from pIC50 if available
    if "pIC50" in df.columns:
        def lbl(p):
            if pd.isna(p):
                return "unknown"
            if p >= thr_active:   # IC50 <= 1 µM
                return "active"
            if p <  thr_inactive: # IC50 > 10 µM
                return "inactive"
            return "intermediate"
        df = df.copy()
        df["class"] = df["pIC50"].apply(lbl)
        return df
    # No labels and no pIC50 — keep unlabeled but still deduplicate
    df = df.copy()
    df["class"] = "unknown"
    return df

def pick_rep(df_group: pd.DataFrame) -> pd.Series:
    """Choose one representative row per canonical SMILES.
       Prefer highest pIC50 (most potent); otherwise first."""
    if "pIC50" in df_group.columns:
        # idxmax skips NaNs automatically
        idx = df_group["pIC50"].astype(float).idxmax()
        return df_group.loc[idx]
    return df_group.iloc[0]

def main():
    ap = argparse.ArgumentParser(description="Deduplicate by canonical SMILES and report class balance.")
    ap.add_argument("--in", dest="infile", required=True,
                    help="Input CSV (e.g., data/processed/bioactivity_data_3class_pIC50.csv)")
    ap.add_argument("--out", dest="outfile", default="data/processed/bioactivity_data_3class_pIC50_dedup.csv",
                    help="Output CSV path for deduplicated dataset")
    ap.add_argument("--thr-active", type=float, default=6.0, help="Active if pIC50 >= thr (default 6.0 ≈ 1 µM)")
    ap.add_argument("--thr-inactive", type=float, default=5.0, help="Inactive if pIC50 < thr (default 5.0 ≈ 10 µM)")
    args = ap.parse_args()

    inp = Path(args.infile)
    if not inp.exists():
        print(f"Error: file not found: {inp}", file=sys.stderr)
        sys.exit(2)

    df = read_table(inp)
    n0 = len(df)

    # Canonicalize SMILES
    df["canonical"] = df["smiles"].apply(to_canonical)
    n_invalid = df["canonical"].isna().sum()
    df = df.dropna(subset=["canonical"]).copy()

    # Label if needed
    df = ensure_binary_labels(df, args.thr_active, args.thr_inactive)

    # Deduplicate by canonical SMILES, keeping the most potent (highest pIC50)
    grouped = df.groupby("canonical", as_index=False, group_keys=False)
    df_dedup = grouped.apply(pick_rep).reset_index(drop=True)
    n1 = len(df_dedup)
    n_dups = n0 - n_invalid - n1

    # Class counts
    counts = df_dedup["class"].value_counts(dropna=False).to_dict()
    active = counts.get("active", 0)
    inactive = counts.get("inactive", 0)
    interm = counts.get("intermediate", 0)
    unknown = counts.get("unknown", 0)

    # Save outputs
    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    # nice ordering
    cols = ["smiles", "canonical"] + [c for c in df_dedup.columns if c not in ("smiles","canonical")]
    df_dedup[cols].to_csv(out, index=False)

    stats = {
        "input_rows": n0,
        "invalid_smiles": int(n_invalid),
        "unique_molecules": int(n1),
        "duplicates_removed": int(n_dups),
        "class_counts": {
            "active": int(active),
            "inactive": int(inactive),
            "intermediate": int(interm),
            "unknown": int(unknown),
        },
        "thresholds": {
            "active_pIC50_ge": args.thr_active,
            "inactive_pIC50_lt": args.thr_inactive
        },
        "input_path": str(inp),
        "output_path": str(out),
    }
    # print to console
    print(json.dumps(stats, indent=2))
    # also save a copy
    (OUT_TAB / "dedup_stats.json").write_text(json.dumps(stats, indent=2))
    # and a tiny CSV for quick glances
    pd.DataFrame([{
        "input_rows": n0,
        "invalid_smiles": n_invalid,
        "unique_molecules": n1,
        "duplicates_removed": n_dups,
        "active": active, "inactive": inactive,
        "intermediate": interm, "unknown": unknown
    }]).to_csv(OUT_TAB / "dedup_stats.csv", index=False)

if __name__ == "__main__":
    main()
