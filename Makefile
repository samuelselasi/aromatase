PY := python
VENV := .venv
PIP := $(VENV)/bin/pip

# -------- data artifacts (original pipeline) --------
RAW := data/raw/bioactivty_data.csv
P3  := data/processed/bioactivty_data_preprocessed_class.csv
PPI := data/processed/bioactivity_data_3class_pIC50.csv
FPS := data/processed/bioactivity_data_descriptors_morgan.csv

# -------- deduplicated artifacts (new) --------------
PPI_DEDUP := data/processed/bioactivity_data_3class_pIC50_dedup.csv
FPS_DEDUP := data/processed/bioactivity_data_descriptors_morgan_dedup.csv
MODEL_CLS_DEDUP := results/models/cls_random_forest_dedup.joblib

# -------- external inputs (AfroDB) --------
AFRO_SMI := data/external/afrodb/smiles_unique_EANPDB.smi

# -------- external validation via ChEMBL --------
EXT_DIR := data/external/chembl
EXT_CSV := $(EXT_DIR)/aromatase_chembl_external.csv
EXT_CLEAN := data/external/external_clean.csv
EXT_FPS := data/external/external_fps_morgan.csv

# default PDB ID
PDB ?= 3S79

.PHONY: help install clean \
        data-all data-raw data-processed fps \
        dedup-stats fps-dedup train-cls-dedup \
        pdb train-cls train-reg predict-cls predict-reg \
        predict-afrodb-cls predict-afrodb-reg predict-afrodb-cls-dedup \
        docking-prepare docking-ligands docking-ligands-afrodb docking-run docking-all \
        rank-hits export-top-poses \
        external-all external-fetch external-prepare external-fps external-predict external-evaluate

help:
	@echo "======== Setup ========"
	@echo "make install                           # create venv and install deps"
	@echo ""
	@echo "======== Data (original) ========"
	@echo "make data-all                          # fetch chembl -> labels -> pIC50 -> morgan fps"
	@echo "make data-raw                          # build just the raw ChEMBL CSV"
	@echo "make data-processed                    # build labeled + pIC50 CSVs"
	@echo "make fps                               # build Morgan fingerprints only"
	@echo ""
	@echo "======== Dedup & Balanced CLS (new) ========"
	@echo "make dedup-stats                       # canonicalize, drop duplicates, report class counts"
	@echo "make fps-dedup                         # fingerprints for deduplicated CSV"
	@echo "make train-cls-dedup [SMOTE=1]         # train balanced RF (class_weight) or with SMOTE"
	@echo ""
	@echo "======== Modeling / Inference ========"
	@echo "make train-cls                         # train classifier (original FPS)"
	@echo "make train-reg                         # train regressor (pIC50)"
	@echo "make predict-cls SMILES='CCO' MODEL=cls_random_forest"
	@echo "make predict-reg SMILES='CCO' MODEL=reg_random_forest"
	@echo "make predict-afrodb-cls                # classify AfroDB .smi with baseline model"
	@echo "make predict-afrodb-reg                # regress pIC50 for AfroDB .smi"
	@echo "make predict-afrodb-cls-dedup          # classify AfroDB .smi with *dedup-trained* model"
	@echo ""
	@echo "======== Docking ========"
	@echo "make docking-prepare                   # write receptor.pdbqt + grid config (PDB=$(PDB))"
	@echo "make docking-ligands                   # build ligands from processed CSV (actives)"
	@echo "make docking-ligands-afrodb            # build ligands from $(AFRO_SMI)"
	@echo "make docking-run                       # run vina over prepared ligands"
	@echo "make docking-all                       # prepare + ligands + run"
	@echo ""
	@echo "======== Post-processing ========"
	@echo "make rank-hits [TOP=50]                # merge dock scores + ML preds into ranked hitlist"
	@echo "make export-top-poses [TOP=20]         # export top poses to one SDF"
	@echo ""
	@echo "======== External Validation (ChEMBL, 200 mols) ========"
	@echo "make external-all                      # ChEMBL aromatase dataset (200) -> clean -> fps -> predict"

# -------- setup --------
install:
	python3 -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install -r env/requirements.txt

# -------- original data pipeline --------
$(RAW): scripts/fetch_chembl.py
	$(VENV)/bin/python scripts/fetch_chembl.py

$(P3): $(RAW) scripts/build_labels.py
	$(VENV)/bin/python scripts/build_labels.py

$(PPI): $(P3) scripts/make_pic50.py
	$(VENV)/bin/python scripts/make_pic50.py

$(FPS): $(PPI) scripts/morgan_fp.py
	$(VENV)/bin/python scripts/morgan_fp.py

data-raw: $(RAW)
data-processed: $(PPI)
fps: $(FPS)
data-all: $(FPS)

# -------- dedup pipeline --------
dedup-stats: scripts/dedup_and_stats.py $(PPI)
	$(VENV)/bin/python scripts/dedup_and_stats.py --in $(PPI) --out $(PPI_DEDUP)

fps-dedup: scripts/morgan_fp_from_csv.py $(PPI_DEDUP)
	$(VENV)/bin/python scripts/morgan_fp_from_csv.py --in $(PPI_DEDUP) --out $(FPS_DEDUP)

train-cls-dedup: scripts/train_cls_balanced.py fps-dedup
	$(VENV)/bin/python scripts/train_cls_balanced.py \
	  $$( [ "$${SMOTE:-0}" = "1" ] && echo --smote ) \
	  --fps $(FPS_DEDUP) \
	  --out $(MODEL_CLS_DEDUP)

# -------- training --------
train-cls: $(FPS) scripts/train_cls.py
	$(VENV)/bin/python scripts/train_cls.py

train-reg: $(FPS) scripts/train_reg.py
	$(VENV)/bin/python scripts/train_reg.py

# -------- inference --------
predict-cls: $(FPS) scripts/predict.py
	@if [ -z "$(SMILES)" ]; then echo 'Usage: make predict-cls SMILES="CCO" MODEL=cls_random_forest'; exit 1; fi
	$(VENV)/bin/python scripts/predict.py --task cls --smiles "$(SMILES)" --model-name $${MODEL:-cls_random_forest}

predict-reg: $(FPS) scripts/predict.py
	@if [ -z "$(SMILES)" ]; then echo 'Usage: make predict-reg SMILES="CCO" MODEL=reg_random_forest'; exit 1; fi
	$(VENV)/bin/python scripts/predict.py --task reg --smiles "$(SMILES)" --model-name $${MODEL:-reg_random_forest}

predict-afrodb-cls: scripts/predict.py
	@test -f "$(AFRO_SMI)" || (echo "Missing $(AFRO_SMI). Put your .smi there."; exit 2)
	$(VENV)/bin/python scripts/predict.py --task cls --in $(AFRO_SMI) --model-name cls_random_forest --out results/predictions/afrodb_cls.csv

predict-afrodb-reg: scripts/predict.py
	@test -f "$(AFRO_SMI)" || (echo "Missing $(AFRO_SMI). Put your .smi there."; exit 2)
	$(VENV)/bin/python scripts/predict.py --task reg --in $(AFRO_SMI) --model-name reg_random_forest --as-ic50-nm --out results/predictions/afrodb_reg.csv

predict-afrodb-cls-dedup: scripts/predict.py $(MODEL_CLS_DEDUP)
	@test -f "$(AFRO_SMI)" || (echo "Missing $(AFRO_SMI). Put your .smi there."; exit 2)
	$(VENV)/bin/python scripts/predict.py --task cls --in $(AFRO_SMI) --model-path $(MODEL_CLS_DEDUP) --out results/predictions/afrodb_cls_dedup.csv

# -------- docking --------
docking-prepare: pdb scripts/prepare_receptor.py
	$(VENV)/bin/python scripts/prepare_receptor.py --pdb-id $(PDB)

docking-ligands: $(PPI) scripts/prepare_ligands.py
	$(VENV)/bin/python scripts/prepare_ligands.py --csv $(PPI) --subset active --limit 200

docking-ligands-afrodb: scripts/prepare_ligands.py
	@test -f "$(AFRO_SMI)" || (echo "Missing $(AFRO_SMI). Put your .smi there."; exit 2)
	$(VENV)/bin/python scripts/prepare_ligands.py --smi $(AFRO_SMI) --limit 200

docking-run: scripts/dock_batch.py
	$(VENV)/bin/python scripts/dock_batch.py

docking-all: docking-prepare docking-ligands docking-run

# -------- post-processing --------
rank-hits: results/docking/scores.csv scripts/rank_hits.py
	$(VENV)/bin/python scripts/rank_hits.py --top $${TOP:-50}

export-top-poses: scripts/export_top_poses.py
	$(VENV)/bin/python scripts/export_top_poses.py --top $${TOP:-20}

# -------- external validation (ChEMBL, 200 mols) --------
external-all: external-fetch external-prepare external-fps external-predict external-evaluate
	@echo "External validation pipeline complete."

external-fetch: scripts/fetch_aromatase_chembl.py
	$(VENV)/bin/python scripts/fetch_aromatase_chembl.py

external-prepare: scripts/prepare_external_validation.py $(EXT_CSV) $(PPI)
	$(VENV)/bin/python scripts/prepare_external_validation.py \
	  --infile $(EXT_CSV) \
	  --training-csv $(PPI) \
	  --ic50-threshold-nm 1000 \
	  --out $(EXT_CLEAN)

external-fps: scripts/morgan_fp_external.py $(EXT_CLEAN)
	$(VENV)/bin/python scripts/morgan_fp_external.py --in $(EXT_CLEAN) --out $(EXT_FPS)

external-predict: scripts/predict_and_eval_external.py $(EXT_FPS)
	$(VENV)/bin/python scripts/predict_and_eval_external.py \
	  --fps $(EXT_FPS) \
	  --cls-model $(MODEL_CLS_DEDUP) \
	  --reg-model results/models/reg_random_forest.joblib \
	  --out-dir results/external

external-evaluate: external-predict
	@echo "External predictions ready in results/external/"
	@ls -1 results/external | sed 's/^/  - /'

# -------- clean --------
clean:
	rm -rf data/processed/*.csv data/raw/*.csv data/raw/pdb docking/*.pdb* docking/config.txt results/* data/external/chembl/*.csv
