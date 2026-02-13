# Docking

Holds receptor and search box for AutoDock Vina.

- `receptor.pdb`   — protein only (waters/ligands removed).
- `receptor.pdbqt` — AD4-typed, protonated at set pH (Open Babel `-p`).
- `config.txt`     — Vina config with center/size.

## Regenerate
```
make docking-prepare          # uses PDB=$(PDB) from the Makefile (default 5JL9)
# or adjust box size:
.venv/bin/python scripts/prepare_receptor.py --pdb-id 5JL9 --size 32 32 32
```
