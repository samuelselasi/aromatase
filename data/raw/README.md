# Raw Data

Raw aromatase IC50 data fetched from ChEMBL.

- `bioactivty_data.csv`
  - `molecule_chembl_id`  ChEMBL molecule ID
  - `canonical_smiles`    SMILES string
  - `standard_value`      IC50 in nM

Rebuild with:
```
make data-raw
```

Downloaded PDB coordinates (via `scripts/fetch_pdb.py`).

- Filenames may be `.pdb`, `.ent`, or `.gz`.
- Receptor prep will auto-detect and (if needed) decompress.

## Fetch
bash
make pdb PDB=5JL9
```
