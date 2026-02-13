
# Docking Results

Docking outputs.

- `<ligand>_out.pdbqt` — docked pose.
- `<ligand>.log`       — stdout/stderr captured by `dock_batch.py`.
- `scores.csv`         — affinity summary (kcal/mol), sorted best→worst by the script.

## Run / Inspect
```
make docking-run
# top 15 hits (most negative):
tail -n +2 results/docking/scores.csv | sort -t, -k2,2g | head -15
```
