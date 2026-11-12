# -*- coding: utf-8 -*-
"""Batch 2 装入：cipa2_sweeps.csv → 统一总表追加
行粒度 = 每细胞×每sweep 的 τ_deact（剖面似然 CI）
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
sw = pd.read_csv(OUT / "cipa2_sweeps.csv")
print("sweeps:", len(sw))

uni = pd.DataFrame({
    "source_big": "cipa_5lab",
    "dataset": sw["lab"] + ("_blind" if False else ""),
    "cell_id": sw["cell"],
    "channel": "hERG",
    "family": "Kv",
    "species": "",
    "temp_C": np.nan,
    "parameter": "tau_deact",
    "process": "drug_block",
    "voltage_mV": np.nan,
    "value": sw["tau_tail_ms"],
    "se_pcov": np.nan,
    "prof_lo": sw["tau_prof_lo"],
    "prof_hi": sw["tau_prof_hi"],
    "model": "exp1_tail",
    "p_model": np.nan,
    "r2": sw["tail_r2"],
    "qc_flag": np.where(sw["qc_pass"], "ok", "qc_fail") + "|seg_" + sw["seg"].astype(str),
    "method": "profile_v1",
    # 药物上下文凭证（统一表附加列，Channelpedia 行为空）
    "drug": sw["drug"],
    "blind": sw["blind"],
    "A_peak_pA": sw["A_peak"],
})
uni["dataset"] = sw["lab"] + np.where(sw["blind"], "_blind", "")

m = OUT / "master_unified_v1.csv"
base = pd.read_csv(m)
for c in ["drug", "blind", "A_peak_pA"]:
    if c not in base.columns:
        base[c] = np.nan if c != "blind" else False
uni = uni[base.columns]
out = pd.concat([base, uni], ignore_index=True)
out.to_csv(m, index=False, encoding="utf-8-sig")
print("统一总表 ->", m, len(out), "行")
print(out.groupby("source_big")["cell_id"].nunique())
