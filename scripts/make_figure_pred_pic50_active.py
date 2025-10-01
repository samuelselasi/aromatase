#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

def read_hitlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # try to normalize column names
    cols = {c.lower(): c for c in df.columns}
    # accept 'ligand' or 'id'
    if "id" not in cols and "ligand" in cols:
        df = df.rename(columns={cols["ligand"]: "id"})
    # ensure prob_active & pred_pIC50 exist
    for want, aliases in {
        "prob_active": ["prob_active", "prob", "probability", "p_active"],
        "pred_pIC50": ["pred_pIC50", "pIC50_pred", "pic50_pred", "pred_pic50"],
    }.items():
        if not any(a in df.columns for a in aliases):
            # leave missing; we'll handle later
            continue
        for a in aliases:
            if a in df.columns:
                df = df.rename(columns={a: want})
                break
    return df

def merge_cls_reg(cls_csv: Path, reg_csv: Path) -> pd.DataFrame:
    c = pd.read_csv(cls_csv)
    r = pd.read_csv(reg_csv)
    # normalize key
    if "id" not in c.columns:
        raise SystemExit(f"Missing 'id' in {cls_csv}")
    if "id" not in r.columns:
        raise SystemExit(f"Missing 'id' in {reg_csv}")
    m = c.merge(r[["id","pred_pIC50"]], on="id", how="left")
    return m

def nm_from_pic50(p):
    # IC50[nM] = 10**(9 - pIC50)
    return float(10.0**(9.0 - float(p)))

def main():
    ap = argparse.ArgumentParser(description="Figure 2: Histogram of predicted pIC50 for AfroDB compounds classified as active.")
    ap.add_argument("--hitlist", help="Optional: results/tables/hitlist.csv (must contain prob_active and pred_pIC50)")
    ap.add_argument("--cls", help="Classification predictions CSV (e.g., results/predictions/afrodb_cls_dedup.csv)")
    ap.add_argument("--reg", help="Regression predictions CSV (e.g., results/predictions/afrodb_reg.csv)")
    ap.add_argument("--thr", type=float, default=0.5, help="prob_active threshold if pred_class is absent (default 0.5)")
    ap.add_argument("--out-png", default=str(FIG_DIR / "figure2_pred_pIC50_active.png"))
    ap.add_argument("--out-json", default=str(TAB_DIR / "figure2_pred_pIC50_stats.json"))
    args = ap.parse_args()

    if args.hitlist:
        df = read_hitlist(Path(args.hitlist))
    else:
        if not (args.cls and args.reg):
            raise SystemExit("Provide --hitlist OR both --cls and --reg.")
        df = merge_cls_reg(Path(args.cls), Path(args.reg))

    # filter 'active' set
    if "pred_class" in df.columns:
        mask = (df["pred_class"].astype(str).str.lower() == "active")
    elif "prob_active" in df.columns:
        mask = (df["prob_active"] >= args.thr)
    else:
        raise SystemExit("No 'pred_class' or 'prob_active' column found to filter actives.")
    act = df.loc[mask].copy()

    if "pred_pIC50" not in act.columns:
        raise SystemExit("Missing 'pred_pIC50' column. If using --hitlist, ensure it includes regression preds; otherwise pass --cls and --reg.")

    # drop NaNs
    act = act.dropna(subset=["pred_pIC50"])
    if act.empty:
        raise SystemExit("No active compounds with predicted pIC50 available.")

    vals = act["pred_pIC50"].astype(float).values
    stats = {
        "n_active": int(len(vals)),
        "min_pIC50": float(np.min(vals)),
        "max_pIC50": float(np.max(vals)),
        "mean_pIC50": float(np.mean(vals)),
        "median_pIC50": float(np.median(vals)),
        "min_IC50_nM": nm_from_pic50(np.max(vals)),  # careful: high pIC50 = low IC50
        "max_IC50_nM": nm_from_pic50(np.min(vals)),
        "note": "IC50[nM] = 10**(9 - pIC50); higher pIC50 => lower IC50."
    }

    # plot
    plt.figure(figsize=(7,4.5))
    plt.hist(vals, bins=30)
    plt.xlabel("Predicted pIC50")
    plt.ylabel("Count")
    plt.title("Distribution of predicted pIC50 for AfroDB compounds classified as active")
    plt.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"Wrote {out_png}")

    # save stats
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
