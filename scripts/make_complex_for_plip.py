#!/usr/bin/env python3
import argparse
from pathlib import Path

def read_lines(p):
    return [ln.rstrip("\n") for ln in open(p, "r", errors="ignore").read().splitlines()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor", required=True)  # should contain HEM
    ap.add_argument("--ligand",   required=True)  # pose PDB from vina -> obabel, will rewrite to HETATM
    ap.add_argument("--out",      required=True)
    ap.add_argument("--ligres",   default="LIG")  # ligand residue name
    ap.add_argument("--chain",    default="Z")    # ligand chain id not used by receptor
    ap.add_argument("--resid",    type=int, default=999)
    args = ap.parse_args()

    rec = read_lines(args.receptor)
    lig = read_lines(args.ligand)

    # keep only ATOM/HETATM/TER from receptor, drop END etc.
    rec_keep = [ln for ln in rec if (ln.startswith("ATOM") or ln.startswith("HETATM") or ln.startswith("TER"))]

    # convert ligand lines to HETATM with clean PDB columns
    lig_conv = []
    for ln in lig:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        # Ensure at least 80 chars
        ln = ln + " " * (80 - len(ln)) if len(ln) < 80 else ln
        ln = "HETATM" + ln[6:]                      # record name
        ln = ln[:17] + f"{args.ligres:>3}" + ln[20:]  # resName (17-19)
        ln = ln[:21] + f"{args.chain:1}" + ln[22:]    # chainID (21)
        ln = ln[:22] + f"{args.resid:4d}" + ln[26:]   # resSeq (22-25)
        lig_conv.append(ln[:80])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fo:
        for ln in rec_keep:
            fo.write(ln + "\n")
        fo.write("TER\n")
        for ln in lig_conv:
            fo.write(ln + "\n")
        fo.write("TER\nEND\n")

    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
