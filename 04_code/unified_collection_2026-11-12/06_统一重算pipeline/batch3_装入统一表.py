# -*- coding: utf-8 -*-
"""Batch3 装入统一总表"""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
lei = pd.read_csv(OUT / "lei3_unified.csv")
m = pd.read_csv(OUT / "master_unified_v1.csv")
lei = lei.reindex(columns=m.columns)
out = pd.concat([m, lei], ignore_index=True)
out.to_csv(OUT / "master_unified_v1.csv", index=False, encoding="utf-8-sig")
print("统一总表:", len(out), "行")
print(out.groupby("source_big")["cell_id"].nunique())
print(out.groupby("source_big").size())
