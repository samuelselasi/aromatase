# Aromatase QSAR + Docking Pipeline

> A compact, reproducible pipeline that (1) *learns QSAR models for aromatase (CYP19A1) inhibition*, (2) *screens an external library (AfroDB/EANPDB)*, and (3) *docks those compounds into an aromatase crystal structure to prioritize putative hits*. Everything is wired through a `Makefile` for one-liners and easy reruns.

## What this pipeline does

1. **Fetch & clean data** from ChEMBL for aromatase activity.
2. **Label** (active/inactive) and compute **pIC50**.
3. **Featurize** molecules with RDKit **Morgan fingerprints**.
4. **Train** baseline ML models (classifier & regressor).
5. **Predict** activity and pIC50 for **AfroDB/EANPDB** molecules.
6. **Prepare receptor** (PDB **5JL9**) and **ligands** for docking.
7. **Dock** ligands with AutoDock **Vina 1.2.3** and collect scores.
8. (*Optional*) **Rank** hits by docking and (optionally) fuse with ML predictions.

---

## Repository Layout

```
aromatase/
├─ data/
│  ├─ external/afrodb/           # AfroDB/EANPDB .smi input
│  ├─ raw/                       # downloaded sources (ChEMBL, PDB)
│  └─ processed/                 # cleaned tables, fingerprints, 3D SDF/PDBQT ligands
├─ docking/                      # receptor + Vina grid (receptor.pdbqt, config.txt)
├─ results/
│  ├─ predictions/               # model outputs (AfroDB cls/reg)
│  └─ docking/                   # Vina poses + scores.csv
├─ scripts/                      # data/ML/prediction/prep/docking utilities
├─ env/requirements.txt          # Python deps (pin numpy<2 if using older RDKit builds)
└─ Makefile                      # one-liners for the end-to-end workflow
```

---

## Quickstart

```

# 0) environment
python3 -m venv .venv
. .venv/bin/activate
pip install -r env/requirements.txt

# 1) data → labels → pIC50 → fingerprints (original pipeline)
make data-all

# 2) deduplicate + stats (canonical SMILES; keep most potent per unique)
make dedup-stats
# writes:
#   data/processed/bioactivity_data_3class_pIC50_dedup.csv
#   results/tables/dedup_stats.{json,csv}

# 3) build fingerprints for the deduplicated set
make fps-dedup
# writes:
#   data/processed/bioactivity_data_descriptors_morgan_dedup.csv

# 4) train a *balanced* classifier on the dedup set
make train-cls-dedup                # uses class_weight="balanced"
# (optional) compare with SMOTE (requires imbalanced-learn)
make train-cls-dedup SMOTE=1

# 5) batch predictions on AfroDB (put your .smi at data/external/afrodb/smiles_unique_EANPDB.smi)
make predict-afrodb-cls-dedup       # uses the dedup-trained classifier
make predict-afrodb-reg             # pIC50 regression (baseline model)

# 6) docking
#    (change the PDB ID if you want, e.g. PDB=3S79)
make docking-prepare PDB=5JL9       # builds receptor.pdbqt + docking/config.txt
make docking-ligands-afrodb         # 3D SDF + PDBQT for AfroDB ligands
make docking-run                    # runs AutoDock Vina 1.2.3 over prepared ligands

# 7) inspect results
head results/predictions/afrodb_cls_dedup.csv
head results/predictions/afrodb_reg.csv
head results/docking/scores.csv

# (optional) consensus ranking + pose export
make rank-hits TOP=50
make export-top-poses TOP=20

```

### Key Outputs
* **Predictions (AfroDB)**
	* `results/predictions/afrodb_cls.csv` — `prob_active, pred_class`
	* `results/predictions/afrodb_reg.csv` — `pred_pIC50 (and pred_IC50_nM if requested)`
* **Docking**
	* `results/docking/scores.csv` — affinity (kcal/mol), **more negative = better**
	* `results/docking/<ligand>_out.pdbqt` — docked pose for each ligand

* **Intermediate data**
	* `data/processed/*pIC50.csv` — cleaned/labelled tables
	* `data/processed/bioactivity_data_descriptors_morgan.csv` — fingerprint matrix

### Mkefile Cheatsheet
```
make data-all                 # ChEMBL → labels → pIC50 → fingerprints
make train-cls / train-reg    # train models (saved in results/models/)
make predict-afrodb-cls       # classify AfroDB .smi
make predict-afrodb-reg       # regress pIC50 for AfroDB .smi
make docking-prepare          # receptor.pdbqt + config.txt (from PDB=5JL9)
make docking-ligands-afrodb   # build 3D SDF + PDBQT for AfroDB ligands
make docking-run              # batch dock with Vina 1.2.3 → scores.csv
```

---

## Reproducibility

* Python virtual environment is recommended.
* `env/requirements.txt` pins NumPy 1.26.4 to match RDKit wheels.
* Makefile uses file targets so only stale steps rebuild.

---

# Data Sources & License

* ChEMBL bioactivity data is fetched at build time. Check ChEMBL terms if you plan to redistribute.
* Optional natural products SDF can be placed in data/external/. See data/external/README.md.

---

## Makefile Summary

Run `make help` for a quick list of targets.

# env

Environment and dependency management.

## Install

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt
```

`requirements.txt` pins:
* `numpy==1.26.4` to match `rdkit-pypi==2022.9.5`
* `scikit-learn==1.3.2` and `scipy==1.10.1` for compatibility

---

## Alternative (Conda)

```
conda create -n aromatase -c conda-forge python=3.10 rdkit pandas numpy scikit-learn chembl_webresource_client biopython
conda activate aromatase
```
