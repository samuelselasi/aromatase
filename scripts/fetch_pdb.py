#!/usr/bin/env python3
from Bio.PDB import PDBList
from pathlib import Path
import sys

def main(pdb_id="3S79"):
    outdir = Path("data/raw/pdb")
    outdir.mkdir(parents=True, exist_ok=True)
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=str(outdir), file_format="pdb")
    print(f"Downloaded PDB {pdb_id} to {outdir}")

if __name__ == "__main__":
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "3S79"
    main(pdb_id)
