#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

# Ligands to render
LIGANDS = [
    "uvarinol",
    "isouvarinol",
    "dichamanetin",
    "isochamanetin",
]

DOCK_DIR = "results/docking"
OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def safe_similarity_map(mol, weights, cmap="coolwarm"):
    """Compatibility-safe 2D similarity map renderer."""
    fig = plt.figure(figsize=(4, 4))
    try:
        # older RDKit versions use this function signature
        SimilarityMaps.GetSimilarityMapFromWeights(mol, weights, colorMap=cmap)
    except Exception:
        # fallback simple drawing
        from rdkit.Chem import Draw
        d = Draw.MolToImage(mol, size=(300, 300))
        plt.imshow(d)
    plt.axis("off")
    return fig

for lig in LIGANDS:
    sdf_path = os.path.join(DOCK_DIR, f"{lig}.sdf")
    if not os.path.exists(sdf_path):
        print(f"[WARN] {sdf_path} not found, skipping")
        continue

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        print(f"[ERROR] Failed to load {sdf_path}")
        continue

    try:
        fp, weights = SimilarityMaps.GetMorganFingerprint(mol, radius=2)
    except Exception:
        weights = [1.0] * mol.GetNumAtoms()

    try:
        fig = safe_similarity_map(mol, weights)
        out_path = os.path.join(OUT_DIR, f"{lig}_2d_interactions.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved → {out_path}")
    except Exception as e:
        print(f"[ERROR] Could not render {lig}: {e}")
