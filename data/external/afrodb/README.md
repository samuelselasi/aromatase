
# External Data

External input molecules (AfroDB/EANPDB) in `.smi` format.

- `smiles_unique_EANPDB.smi` — one per line: `SMILES [optional_id]`

Example:

CCO ethanol
c1ccccc1 benzene


## Use
- Predict:
  ```
make predict-afrodb-cls
make predict-afrodb-reg
```

* Prepare ligands for docking:

```
make docking-ligands-afrodb
```
