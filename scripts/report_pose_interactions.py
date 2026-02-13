#!/usr/bin/env python3
import math, csv, sys
from pathlib import Path

def parse_atoms(pdb, selector):
    out=[]
    with open(pdb, "r", errors="ignore") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")): 
                continue
            rec = ln[:6].strip()
            name=ln[12:16].strip()
            resn=ln[17:20].strip()
            chain=ln[21]
            resi=ln[22:26].strip()
            try:
                x=float(ln[30:38]); y=float(ln[38:46]); z=float(ln[46:54])
            except ValueError:
                continue
            elem=ln[76:78].strip() or name[0]  # fallback
            a={"rec":rec,"name":name,"resn":resn,"chain":chain,"resi":resi,"elem":elem,"xyz":(x,y,z)}
            if selector(a): out.append(a)
    return out

def dist(a,b):
    ax,ay,az=a["xyz"]; bx,by,bz=b["xyz"]
    return math.dist((ax,ay,az),(bx,by,bz))

def main(in_pdb, out_csv):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # ligand defined as LIG Z 999 (what we wrote with make_complex_for_plip.py)
    lig = parse_atoms(in_pdb, lambda a: a["rec"]=="HETATM" and a["resn"]=="LIG" and a["chain"]=="Z" and a["resi"]=="999")
    prot= parse_atoms(in_pdb, lambda a: a["rec"]=="ATOM")
    heme= parse_atoms(in_pdb, lambda a: a["rec"]=="HETATM" and a["resn"] in ("HEM","HEC"))
    fe  = [a for a in heme if a["name"].strip()=="FE"]

    rows=[]
    # Metal proximity (Fe···N/O)
    for metal in fe:
        for la in lig:
            if la["elem"] in ("N","O"):
                rows.append(["metal_contact","FE",metal["chain"],metal["resi"],la["name"],f"{dist(metal,la):.2f}"])

    # H-bond candidates (simple distance screen, 3.5 Å, N/O only)
    for pa in prot:
        if pa["elem"] not in ("N","O"): 
            continue
        for la in lig:
            if la["elem"] not in ("N","O"): 
                continue
            d = dist(pa,la)
            if d <= 3.5:
                rows.append(["hbond_candidate", f"{pa['resn']}{pa['resi']}", pa["chain"], pa["resi"], la["name"], f"{d:.2f}"])

    # Hydrophobics (protein carbon within 4.5 Å of any ligand heavy atom)
    lig_heavy = [la for la in lig if la["elem"]!="H"]
    for pa in prot:
        if pa["elem"]!="C": 
            continue
        mind = min(dist(pa,la) for la in lig_heavy) if lig_heavy else 999.0
        if mind <= 4.5:
            rows.append(["hydrophobic", f"{pa['resn']}{pa['resi']}", pa["chain"], pa["resi"], "", f"{mind:.2f}"])

    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["type","partner","chain","resi","lig_atom","distance_A"])
        w.writerows(rows)

    # quick summary to stdout
    n_met = sum(1 for r in rows if r[0]=="metal_contact")
    n_hbd = sum(1 for r in rows if r[0]=="hbond_candidate")
    n_hph = sum(1 for r in rows if r[0]=="hydrophobic")
    print(f"Wrote {out_csv}  rows={len(rows)}  metal={n_met}  hbond≈={n_hbd}  hydrophobics={n_hph}")

if __name__=="__main__":
    if len(sys.argv)<3:
        print("Usage: report_pose_interactions.py complex.pdb out.csv"); sys.exit(2)
    main(sys.argv[1], sys.argv[2])
