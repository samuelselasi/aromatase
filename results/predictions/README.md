
# Predictions

Model predictions exported by `scripts/predict.py`.

Typical files:
- `afrodb_cls.csv` — classification (prob_active, pred_class) for AfroDB .smi.
- `afrodb_reg.csv` — regression (pred_pIC50 [+ IC50 nM]) for AfroDB .smi.
- `pred_cls_*.csv` / `pred_reg_*.csv` — generic outputs when using different models.

## Generate:
```
make predict-afrodb-cls
make predict-afrodb-reg
```
