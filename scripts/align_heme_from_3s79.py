#!/usr/bin/env python3
import sys, math
from pathlib import Path

def read_atoms(pdb_path, sel):
    ats=[]
    with open(pdb_path, "r", errors="ignore") as f:
        for ln in f:
            if not (ln.startswith("ATOM") or ln.startswith("HETATM")): continue
            rec=ln[:6].strip(); name=ln[12:16].strip(); resn=ln[17:20].strip()
            chain=ln[21]; resi=ln[22:26].strip()
            try:
                x=float(ln[30:38]); y=float(ln[38:46]); z=float(ln[46:54])
            except ValueError:
                continue
            if sel(rec, name, resn, chain, resi):
                ats.append([ln, (x,y,z)])
    return ats

def ca_selector(chain):
    def sel(rec, name, resn, ch, resi):
        return rec=="ATOM" and name=="CA" and ch==chain
    return sel

def heme_selector():
    def sel(rec, name, resn, ch, resi):
        return rec=="HETATM" and resn in ("HEM","HEC")
    return sel

def to_mat(vs):
    return [[vs[i][j] for j in range(3)] for i in range(len(vs))]

def centroid(pts):
    n=len(pts)
    return [sum(p[i] for p in pts)/n for i in range(3)]

def sub(a,b): return [a[i]-b[i] for i in range(3)]
def add(a,b): return [a[i]+b[i] for i in range(3)]

def matmul(A,B):  # 3x3 * 3x1 or 3x3
    if isinstance(B[0], list):
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    else:
        return [sum(A[i][k]*B[k] for k in range(3)) for i in range(3)]

def transpose(A): return list(map(list, zip(*A)))

def kabsch(P,Q):
    # P, Q: lists of 3D points (ref->mov). Returns R,t so that R*Q + t ≈ P
    Pc = centroid(P); Qc = centroid(Q)
    P0 = [sub(p,Pc) for p in P]; Q0 = [sub(q,Qc) for q in Q]
    H = [[0,0,0],[0,0,0],[0,0,0]]
    for p,q in zip(P0,Q0):
        for i in range(3):
            for j in range(3):
                H[i][j] += q[i]*p[j]
    # SVD of H
    import numpy as np
    U,S,Vt = np.linalg.svd(np.array(H))
    R = (U @ Vt)
    if np.linalg.det(R) < 0:
        U[:,-1] *= -1
        R = U @ Vt
    R = R.tolist()
    t = [Pc[i] - sum(R[i][k]*Qc[k] for k in range(3)) for i in range(3)]
    return R,t

def apply(R,t,xyz):
    return [sum(R[i][k]*xyz[k] for k in range(3)) + t[i] for i in range(3)]

def write_heme(out_path, heme_atoms, R, t):
    with open(out_path,"w") as w:
        for ln,(x,y,z) in heme_atoms:
            xx,yy,zz = apply(R,t,(x,y,z))
            w.write(f"{ln[:30]}{xx:8.3f}{yy:8.3f}{zz:8.3f}{ln[54:]}")
    print(f"Wrote {out_path}  atoms={len(heme_atoms)}")

def main():
    if len(sys.argv)<4:
        print("Usage: align_heme_from_3s79.py docking/receptor.pdb data/raw/pdb/pdb3s79.ent docking/heme_aligned.pdb")
        sys.exit(2)
    rec_pdb, ref_pdb, out_pdb = sys.argv[1:4]
    rec_CA = read_atoms(rec_pdb, ca_selector('A'))
    ref_CA = read_atoms(ref_pdb, ca_selector('A'))
    if not rec_CA or not ref_CA:
        print("ERROR: missing CA atoms on chain A", file=sys.stderr); sys.exit(1)
    # match by residue index where possible
    rec_map = {(ln[22:26].strip(), ln[21]): (xyz) for ln,xyz in rec_CA}
    P=[]; Q=[]
    for ln,xyz in ref_CA:
        key=(ln[22:26].strip(), ln[21])
        if key in rec_map:
            P.append(rec_map[key]); Q.append(xyz)
    if len(P)<10:
        # fallback: use first N pairs
        n=min(len(rec_CA), len(ref_CA))
        P=[rec_CA[i][1] for i in range(n)]
        Q=[ref_CA[i][1] for i in range(n)]
    import numpy as np
    R,t = kabsch(P,Q)
    heme = read_atoms(ref_pdb, heme_selector())
    if not heme:
        print("ERROR: no HEM/HEC in reference PDB", file=sys.stderr); sys.exit(1)
    write_heme(out_pdb, heme, R, t)

if __name__=="__main__":
    main()
