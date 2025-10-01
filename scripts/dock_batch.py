#!/usr/bin/env python3
import argparse, csv, subprocess
from pathlib import Path

ROOT = Path(".")
CFG_DEFAULT = ROOT / "docking" / "config.txt"
REC_DEFAULT = ROOT / "docking" / "receptor.pdbqt"
LIG_DIR_DEFAULT = ROOT / "data" / "processed" / "ligands_pdbqt"
OUT_DIR_DEFAULT = ROOT / "results" / "docking"

def parse_score_from_pdbqt(pdbqt_path: Path):
    if not pdbqt_path.exists():
        return None
    try:
        with open(pdbqt_path, "r") as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok.replace("RESULT:", "RESULT:") == "RESULT:" and i + 1 < len(parts):
                            return float(parts[i + 1])
    except Exception:
        pass
    return None

def run_vina(vina_bin, ligand, config, out_pdbqt, verbosity=1):
    cmd = [vina_bin, "--config", str(config),
           "--ligand", str(ligand),
           "--out", str(out_pdbqt),
           "--verbosity", str(verbosity)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

def main():
    ap = argparse.ArgumentParser(description="Batch dock PDBQT ligands with AutoDock Vina.")
    ap.add_argument("--vina", default="vina", help="Path to Vina (default: vina in PATH)")
    ap.add_argument("--config", default=str(CFG_DEFAULT), help="Path to Vina config.txt")
    ap.add_argument("--receptor", default=str(REC_DEFAULT), help="(not strictly needed if in config) receptor PDBQT")
    ap.add_argument("--ligand-dir", default=str(LIG_DIR_DEFAULT), help="Directory with ligand .pdbqt files")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT), help="Output dir for docked poses and logs")
    ap.add_argument("--limit", type=int, default=200, help="Max ligands to dock")
    ap.add_argument("--verbosity", type=int, default=1, help="Vina verbosity (0,1,2)")
    args = ap.parse_args()

    config = Path(args.config)
    receptor = Path(args.receptor)  # not used directly but nice to sanity check
    lig_dir = Path(args.ligand_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if not config.exists():
        raise SystemExit(f"Missing config: {config}. Run docking-prepare first.")
    if not receptor.exists():
        print(f"[WARN] Receptor file not found at {receptor}. Ensure config points to the right receptor.", flush=True)
    if not lig_dir.exists():
        raise SystemExit(f"Missing ligands dir: {lig_dir}.")

    ligs = sorted(lig_dir.glob("*.pdbqt"))[: args.limit]
    if not ligs:
        raise SystemExit(f"No .pdbqt ligands in {lig_dir}.")

    rows = []
    for i, lig in enumerate(ligs, 1):
        name = lig.stem
        out_pdbqt = out_dir / f"{name}_out.pdbqt"
        log_path  = out_dir / f"{name}.log"

        code, stdout, stderr = run_vina(args.vina, lig, config, out_pdbqt, verbosity=args.verbosity)
        # write combined stdout/stderr since Vina 1.2.3 has no --log
        with open(log_path, "w") as lf:
            lf.write(stdout)
            if stderr:
                lf.write("\n--- STDERR ---\n")
                lf.write(stderr)

        score = parse_score_from_pdbqt(out_pdbqt)

        if code != 0 and score is None:
            print(f"[WARN] Vina exit code {code} for {name} (see {log_path.name})")

        print(f"{i:04d}/{len(ligs)}  {name:40s}  {score if score is not None else 'NA'} kcal/mol")
        rows.append({"ligand": name, "affinity_kcal_per_mol": score})

    # sort by best (most negative first), keep NA at bottom
    def sort_key(r):
        s = r["affinity_kcal_per_mol"]
        return (s is None, s if s is not None else 1e9)
    rows_sorted = sorted(rows, key=sort_key)

    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ligand", "affinity_kcal_per_mol"])
        w.writeheader(); w.writerows(rows_sorted)
    print(f"Wrote {csv_path} (n={len(rows_sorted)})")

if __name__ == "__main__":
    main()
