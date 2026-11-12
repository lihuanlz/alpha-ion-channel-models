# -*- coding: utf-8 -*-
"""Batch4 装入统一总表"""
import pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parent
b = pd.read_csv(OUT / "beattie4_unified.csv")
m = pd.read_csv(OUT / "master_unified_v1.csv")
b = b.reindex(columns=m.columns)
out = pd.concat([m, b], ignore_index=True)
out.to_csv(OUT / "master_unified_v1.csv", index=False, encoding="utf-8-sig")
print("统一总表:", len(out), "行")
print(out.groupby("source_big").agg(rows=("value", "size"), cells=("cell_id", "nunique")))
