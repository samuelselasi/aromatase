# Models

Saved models:
- `cls_logreg.joblib`
- `cls_random_forest.joblib`
- `reg_ridge.joblib`
- `reg_random_forest.joblib`

Use them with `scripts/predict.py`:
```
.venv/bin/python scripts/predict.py --task cls --smiles "CCO" --model-name cls_random_forest
```
