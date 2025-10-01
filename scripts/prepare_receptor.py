#!/usr/bin/env python3
import argparse, sys, subprocess, gzip, shutil
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = Path(".")
RAW  = ROOT / "data" / "raw" / "pdb"
OUT  = ROOT / "docking"
OUT.mkdir(parents=True, exist_ok=True)

SKIP = {"HOH", "WAT"}  # remove waters

def _maybe_decompress(p: Path) -> Path:
    if p.suffix == ".gz":
        out = p.with_suffix("")  # drop .gz
        with gzip.open(p, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return out
    return p

def find_pdb_path(pdb_id: str) -> Path:
    cand = [
        RAW / f"{pdb_id.lower()}.pdb",
        RAW / f"pdb{pdb_id.lower()}.ent",
        RAW / f"{pdb_id.lower()}",
        RAW / f"{pdb_id.lower()}.pdb.gz",
        RAW / f"pdb{pdb_id.lower()}.ent.gz",
    ]
    for p in cand:
        if p.exists():
            return _maybe_decompress(p)
    for p in list(RAW.glob("*.pdb")) + list(RAW.glob("*.ent")) + list(RAW.glob("*.gz")):
        return _maybe_decompress(p)
    print(f"Could not find PDB file for {pdb_id} under {RAW}", file=sys.stderr)
    sys.exit(2)

def centroid(coords):
    A = np.array(coords, dtype=float)
    return A.mean(axis=0)

def guess_center(pdb_file: Path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("rec", str(pdb_file))
    # find largest hetero ligand (non-protein), prefer non-HEM
    ligs = []
    for res in s.get_residues():
        rn = res.get_resname().strip()
        hetflag = res.id[0]  # ' ' means standard residue
        if rn in SKIP or hetflag == ' ':
            continue
        size = sum(1 for _ in res.get_atoms())
        ligs.append((rn, size, res))
    ligs_sorted = sorted(ligs, key=lambda t: (t[0] == "HEM", -t[1]))
    if ligs_sorted:
        coords = [a.coord for a in ligs_sorted[0][2].get_atoms()]
        return centroid(coords)
    # fallback: protein heavy-atom centroid
    atoms = [a.coord for a in s.get_atoms() if getattr(a, "element", "") != "H"]
    return centroid(atoms)

class ProteinOnly(Select):
    def accept_residue(self, residue):
        rn = residue.get_resname().strip()
        hetflag = residue.id[0]
        if rn in SKIP:
            return 0
        return 1 if hetflag == ' ' else 0

def write_protein_only(src: Path, dst: Path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("rec", str(src))
    io = PDBIO()
    io.set_structure(s)
    io.save(str(dst), select=ProteinOnly())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-id", default="3S79")
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--size", type=float, nargs=3, default=[22,22,22])
    ap.add_argument("--obabel", default="obabel")
    args = ap.parse_args()

    src = find_pdb_path(args.pdb_id)

    # 1) protein-only PDB
    rec_pdb = OUT / "receptor.pdb"
    write_protein_only(src, rec_pdb)
    print(f"Wrote {rec_pdb}")

    # 2) convert to PDBQT with OpenBabel (adds H; rigid receptor)
    rec_pdbqt = OUT / "receptor.pdbqt"
    cmd = [args.obabel, str(rec_pdb), "-xr", "-p", str(args.ph), "-h", "-O", str(rec_pdbqt)]
    subprocess.run(cmd, check=True)
    print(f"Wrote {rec_pdbqt}")

    # 3) grid box config (center from ligands/protein)
    cx, cy, cz = guess_center(src)
    sx, sy, sz = args.size
    cfg = OUT / "config.txt"
    cfg.write_text(
        f"receptor = {rec_pdbqt}\n"
        f"center_x = {cx:.3f}\ncenter_y = {cy:.3f}\ncenter_z = {cz:.3f}\n"
        f"size_x = {sx}\nsize_y = {sy}\nsize_z = {sz}\n"
        f"exhaustiveness = 8\nnum_modes = 9\nenergy_range = 3\n"
    )
    print(f"Wrote {cfg}")

if __name__ == "__main__":
    main()
