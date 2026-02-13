#!/usr/bin/env python3
from pathlib import Path
import argparse, math
import pandas as pd

ROOT = Path(".")
TAB = ROOT/"results"/"tables"; TAB.mkdir(parents=True, exist_ok=True)
DOCK = ROOT/"results"/"docking"/"scores.csv"
CLS  = ROOT/"results"/"predictions"/"afrodb_cls.csv"
REG  = ROOT/"results"/"predictions"/"afrodb_reg.csv"

def safe_read(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dock", default=str(DOCK))
    ap.add_argument("--cls", default=str(CLS))
    ap.add_argument("--reg", default=str(REG))
    ap.add_argument("--out", default=str(TAB/"hitlist.csv"))
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.dock)  # ligand, affinity_kcal_per_mol
    # lower (more negative) is better; NA to bottom
    df["dock_rank"] = df["affinity_kcal_per_mol"].rank(method="min", ascending=True, na_option="bottom")

    # join predictions if present
    df = df.rename(columns={"ligand":"id"})
    cls = safe_read(args.cls)
    if cls is not None and "prob_active" in cls.columns:
        cls = cls.rename(columns={"id":"id","prob_active":"prob_active"})
        df = df.merge(cls[["id","prob_active"]], on="id", how="left")
        df["cls_rank"] = (-df["prob_active"]).rank(method="min", ascending=True)

    reg = safe_read(args.reg)
    if reg is not None and "pred_pIC50" in reg.columns:
        reg = reg.rename(columns={"id":"id","pred_pIC50":"pred_pIC50"})
        df = df.merge(reg[["id","pred_pIC50"]], on="id", how="left")
        df["reg_rank"] = (-df["pred_pIC50"]).rank(method="min", ascending=True)

    # consensus rank (weights: 0.6 docking, 0.2 cls, 0.2 reg when available)
    w_d, w_c, w_r = 0.6, 0.2, 0.2
    parts, weights = [df["dock_rank"]], [w_d]
    if "cls_rank" in df: parts.append(df["cls_rank"]); weights.append(w_c)
    if "reg_rank" in df: parts.append(df["reg_rank"]); weights.append(w_r)
    wsum = sum(weights)
    cons = None
    for p, w in zip(parts, weights):
        cons = (p*w if cons is None else cons + p*w)
    df["consensus_score"] = cons / wsum

    df = df.sort_values(["consensus_score", "dock_rank"], ascending=[True, True])
    df = df.rename(columns={"id":"ligand"})
    df.to_csv(args.out, index=False)

    top_path = Path(args.out).with_name(f"hitlist_top{args.top}.csv")
    df.head(args.top).to_csv(top_path, index=False)
    print(f"Wrote {args.out} (n={len(df)})")
    print(f"Wrote {top_path}")

if __name__ == "__main__":
    main()

