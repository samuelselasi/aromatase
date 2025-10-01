#!/usr/bin/env python3
import pandas as pd
from chembl_webresource_client.new_client import new_client
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def main():
    target = pd.DataFrame.from_dict(new_client.target.search("aromatase"))
    selected = target.target_chembl_id.iloc[0]
    res = new_client.activity.filter(target_chembl_id=selected).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res).dropna(subset=["standard_value","canonical_smiles"]).copy()
    df["canonical_smiles"] = df["canonical_smiles"].str.strip()
    out = df[["molecule_chembl_id","canonical_smiles","standard_value"]]
    out.to_csv(RAW/"bioactivty_data.csv", index=False)  # (spelling matches notebook)
    print(f"Wrote {RAW/'bioactivty_data.csv'}  rows={len(out)}")

if __name__ == "__main__":
    main()
