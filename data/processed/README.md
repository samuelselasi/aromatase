# Processed Data

Derived datasets.

- `bioactivty_data_preprocessed_class.csv`
  - adds `class` based on IC50 thresholds: active ≤ 1000 nM, inactive ≥ 10000 nM, else intermediate
- `bioactivity_data_no_intermediate.csv`
  - removes intermediate rows for 2-class classification
- `bioactivity_data_3class_pIC50.csv`
  - adds `pIC50 = -log10(IC50 in M)` and drops invalid IC50 (≤ 0)
- `bioactivity_data_descriptors_morgan.csv`
  - columns: `molecule_chembl_id`, `class`, `pIC50`, and `FP_0..FP_2047`

Rebuild with:
```
make data-processed
make fps
```

## Recreate
```
make data-all
make docking-ligands          # from processed CSV actives
make docking-ligands-afrodb   # from AfroDB .smi
```
