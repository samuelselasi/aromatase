#!/usr/bin/env bash
# =====================================================
# Batch-render 3D binding-mode figures for top ligands
# Author: Samuel Selasi (Aromatase QSAR project)
# =====================================================

# Receptor path (update here if needed)
RECEPTOR="data/raw/pdb/pdb3s79.ent"
OUT_DIR="results/figures"
mkdir -p "$OUT_DIR"

# Ligands to render (all in results/docking/)
LIGANDS=("uvarinol" "isouvarinol" "dichamanetin" "isochamanetin")

# Loop over ligands and render using PyMOL
for LIG in "${LIGANDS[@]}"; do
cat << EOF > tmp_${LIG}.pml
# Load receptor and ligand
load ${RECEPTOR}, receptor
load results/docking/${LIG}_out.pdbqt, ${LIG}

# Clean and style the view
hide everything
show cartoon, receptor
color gray70, receptor
show sticks, ${LIG}
color cyan, ${LIG}

# Highlight heme cofactor (if present)
select heme, resn HEM
show sticks, heme
color red, heme

# Camera and presentation
zoom ${LIG}, 10
set cartoon_transparency, 0.4
bg_color white
set ray_shadows, off

# Output figure
ray 1600,1200
png ${OUT_DIR}/Fig_${LIG}_binding_mode.png, dpi=300
quit
EOF

echo ">>> Rendering ${LIG}..."
pymol -qc tmp_${LIG}.pml
rm -f tmp_${LIG}.pml
done

echo "All binding-mode figures saved in ${OUT_DIR}"
