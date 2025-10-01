#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

def read_hitlist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize key columns if needed
    cols = {c.lower(): c for c in df.columns}
    # docking score
    dock_col = None
    for cand in ["affinity_kcal_per_mol", "affinity", "docking_score"]:
        if cand in df.columns:
            dock_col = cand; break
    if dock_col is None:
        raise SystemExit(f"No docking score column found in {path}. "
                         "Expected one of: affinity_kcal_per_mol / affinity / docking_score.")
    # predicted pIC50
    pic_col = None
    for cand in ["pred_pIC50", "pIC50_pred", "pred_pic50", "pic50_pred"]:
        if cand in df.columns:
            pic_col = cand; break
    if pic_col is None:
        raise SystemExit(f"No predicted pIC50 column found in {path}. "
                         "Expected one of: pred_pIC50 / pIC50_pred / pred_pic50 / pic50_pred.")
    return df.rename(columns={dock_col: "affinity_kcal_per_mol", pic_col: "pred_pIC50"})

def main():
    ap = argparse.ArgumentParser(description="Figure 3: Docking score vs predicted pIC50 for top-N docked compounds.")
    ap.add_argument("--hitlist", default="results/tables/hitlist.csv",
                    help="CSV containing at least affinity_kcal_per_mol and pred_pIC50")
    ap.add_argument("--top", type=int, default=50, help="Top-N by docking score (more negative is better)")
    ap.add_argument("--out-png", default="results/figures/figure3_dock_vs_pred_pIC50_top50.png")
    ap.add_argument("--out-json", default="results/tables/figure3_corr_top50.json")
    args = ap.parse_args()

    df = read_hitlist(Path(args.hitlist))
    df = df.dropna(subset=["affinity_kcal_per_mol", "pred_pIC50"]).copy()

    # pick top-N by best (most negative) docking score
    df_sorted = df.sort_values("affinity_kcal_per_mol", ascending=True).head(args.top).copy()

    x = df_sorted["affinity_kcal_per_mol"].astype(float).values
    y = df_sorted["pred_pIC50"].astype(float).values

    # correlations
    pearson_r = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
    spearman_rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))

    # best-fit line (y = ax + b)
    a, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = a*xs + b

    # plot
    plt.figure(figsize=(6,4.5))
    plt.scatter(x, y, s=18, alpha=0.8)
    plt.plot(xs, ys)  # regression line
    plt.xlabel("Docking score (kcal/mol)  [more negative = better]")
    plt.ylabel("Predicted pIC50")
    plt.title(f"Docking vs Predicted pIC50 (Top {args.top})\n"
              f"Pearson r={pearson_r:.2f}, Spearman ρ={spearman_rho:.2f}")
    plt.tight_layout()
    Path(args.out-png if hasattr(args, "out-png") else args.out_png)  # no-op (keep linter happy)
    out_png = Path(args.out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print(f"Wrote {out_png}")

    # save stats
    out = {
        "top_n": int(args.top),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "slope": float(a),
        "intercept": float(b),
        "n_points": int(len(x))
    }
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
