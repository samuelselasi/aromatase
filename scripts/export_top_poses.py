#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, pandas as pd

ROOT = Path(".")
OUTD = ROOT/"results"/"docking"
TABD = ROOT/"results"/"tables"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hitlist", default=str(TABD/"hitlist.csv"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-sdf", default=str(OUTD/"top_poses.sdf"))
    args = ap.parse_args()

    df = pd.read_csv(args.hitlist).head(args.top)
    sdf_list = []
    for lig in df["ligand"]:
        pose = OUTD / f"{lig}_out.pdbqt"
        sdf  = OUTD / f"{lig}.sdf"
        if not pose.exists(): 
            continue
        subprocess.run(["obabel", str(pose), "-O", str(sdf)], check=True)
        sdf_list.append(str(sdf))
    # merge into one SDF
    if sdf_list:
        subprocess.run(["obabel", "-j", "-O", args.out_sdf] + sdf_list, check=True)
        print(f"Wrote {args.out_sdf} with {len(sdf_list)} poses")
    else:
        print("No poses converted — check docking outputs.")

if __name__ == "__main__":
    main()
