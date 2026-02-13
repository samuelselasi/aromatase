
# ligands_pdbqt/README.md

Ligands in PDBQT for Vina.

- Default path uses **Meeko 0.5.x** (recommended).
- If Meeko isn’t available, falls back to Open Babel (Gasteiger charges).
- Filenames are sanitized (no spaces/slashes).

Regenerate together with SDFs:
```
make docking-ligands-afrodb
# or from processed CSV actives:
make docking-ligands
```
