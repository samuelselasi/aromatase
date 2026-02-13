
# ligands_sdf

3D ligand conformers (SDF), H-added.

- Built by `scripts/prepare_ligands.py`.
- Atom IDs are sanitized from the source IDs (filesystem-safe).
- Count is limited by `--limit` (default 200).

## Rebuild:
```
.venv/bin/python scripts/prepare_ligands.py --smi data/external/afrodb/smiles_unique_EANPDB.smi --limit 200
```
