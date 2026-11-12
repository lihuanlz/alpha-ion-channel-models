# -*- coding: utf-8 -*-
"""
统一重算 Batch 6：文献/组统计锚入统一 schema
三个锚层（非原始波形重提，全部标注 method）：
  A. newCiPA 官方锚：29 药 × 7 通道 IC50(nM)+Hill（CiPA 官方公布值，literature）
  B. Channelpedia 跨物种组统计：38 对 tau_fold / Vhalf_diff（derived_group）
  C. ICGenealogy 模型库层判决：10 层 dispersion_ratio / tau_obs / mid_obs（literature_model）
输出 literature6_anchors.csv（22 列统一 schema）
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
COLUMNS = ["source_big", "dataset", "cell_id", "channel", "family", "species",
           "temp_C", "parameter", "process", "voltage_mV", "value",
           "se_pcov", "prof_lo", "prof_hi", "model", "p_model", "r2",
           "qc_flag", "method", "drug", "blind", "A_peak_pA"]

rows = []

def emit(**kw):
    r = {c: np.nan for c in COLUMNS}
    r.update(species="", voltage_mV=np.nan, se_pcov=np.nan, prof_lo=np.nan,
             prof_hi=np.nan, model="", p_model=np.nan, r2=np.nan,
             qc_flag="ok", drug="", blind=False, A_peak_pA=np.nan)
    r.update(kw)
    rows.append(r)

# ---- A. newCiPA 官方锚 ----
CH_MAP = {"CaL": ("CaV1.2", "CaV"), "hERG": ("hERG", "Kv"), "Ito": ("Kv4.3_Ito", "Kv"),
          "IK1": ("Kir2.1_IK1", "Kir"), "IKs": ("KCNQ1+KCNE1_IKs", "Kv"),
          "Late_sodium": ("Nav1.5_late", "Nav"), "Peak_sodium": ("Nav1.5_peak", "Nav")}
nc = pd.read_csv(LAW / "数据总库" / "06_CiPA官方锚_7通道IC50_29药" / "newCiPA.csv")
n_a = 0
for _, r in nc.iterrows():
    drug = str(r["drug"])
    for k, (ch, fam) in CH_MAP.items():
        ic = r.get(f"{k}IC50", np.nan)
        h = r.get(f"{k}h", np.nan)
        if pd.notna(ic) and ic > 0:
            emit(source_big="cipa_official", dataset="newCiPA_29drug", cell_id=drug,
                 channel=ch, family=fam, parameter="IC50_nM", process="drug_block",
                 value=float(ic), model="hill_lit", method="literature", drug=drug)
            n_a += 1
        if pd.notna(h) and h > 0:
            emit(source_big="cipa_official", dataset="newCiPA_29drug", cell_id=drug,
                 channel=ch, family=fam, parameter="hill", process="drug_block",
                 value=float(h), model="hill_lit", method="literature", drug=drug)
            n_a += 1
print(f"A. newCiPA: {n_a} 行")

# ---- B. 跨物种组统计 ----
cs = pd.read_csv(LAW / "features" / "cross_species_v3.csv")
for _, r in cs.iterrows():
    cid = f"{r.channel}_{r.sp_a}_vs_{r.sp_b}_{r.temp}C"
    emit(source_big="channelpedia_groupstats", dataset="cross_species_v3",
         cell_id=cid, channel=r.channel, family="", species=f"{r.sp_a}|{r.sp_b}",
         temp_C=float(r.temp), parameter="tau_fold", process="cross_species",
         value=float(r.tau_fold), method="derived_group")
    emit(source_big="channelpedia_groupstats", dataset="cross_species_v3",
         cell_id=cid, channel=r.channel, family="", species=f"{r.sp_a}|{r.sp_b}",
         temp_C=float(r.temp), parameter="Vhalf_diff_mV", process="cross_species",
         value=float(r.Vhalf_diff), method="derived_group")
print(f"B. 跨物种: {2*len(cs)} 行")

# ---- C. ICGenealogy 模型库层判决 ----
v = pd.read_csv(LAW / "icg_out" / "verdict_icg_ALL.csv")
FAM_MAP = {"K": "Kv", "Na": "Nav", "KCa": "KCa", "IH": "HCN", "Ca": "CaV"}
for _, r in v.iterrows():
    cid = f"{r.family}_{r.gate}_{r.cls}"
    proc = "activation" if r.gate == "act" else "inactivation"
    for pname, val in [("dispersion_ratio", r.ratio),
                       ("tau_obs_dex", r.tau_obs), ("mid_obs_dex", r.mid_obs)]:
        emit(source_big="icgenealogy", dataset="ICG_933models", cell_id=cid,
             channel=str(r.cls), family=FAM_MAP.get(r.family, r.family),
             parameter=pname, process=proc, value=float(val),
             model="bolt_zmann_selfrefit", method="literature_model")
print(f"C. ICG: {3*len(v)} 行")

df = pd.DataFrame(rows, columns=COLUMNS)
fp = HERE / "literature6_anchors.csv"
df.to_csv(fp, index=False, encoding="utf-8-sig")
print(f"落盘 {fp}: {len(df)} 行")
print(df.groupby(["source_big", "parameter"]).size().to_string())
