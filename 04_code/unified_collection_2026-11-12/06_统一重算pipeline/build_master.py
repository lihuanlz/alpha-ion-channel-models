# -*- coding: utf-8 -*-
"""幂等重建统一总表：master = batch1(Channelpedia) + cipa2_sweeps + lei3 + beattie4
每次全量重建，杜绝重复 concat。运行前提：各 batch 中间表已存在。
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
FEAT = OUT.parent / "features"

COLUMNS = ["source_big", "dataset", "cell_id", "channel", "family", "species",
           "temp_C", "parameter", "process", "voltage_mV", "value",
           "se_pcov", "prof_lo", "prof_hi", "model", "p_model", "r2",
           "qc_flag", "method", "drug", "blind", "A_peak_pA"]

parts = []

# ---- Batch1 Channelpedia（从 profile_all_rows 现构，逻辑同 batch1 脚本） ----
print("[1] Channelpedia ...")
prof = pd.read_csv(FEAT / "profile_all_rows.csv")
v5r = pd.read_csv(FEAT / "features_corpus_v5r.csv",
                  usecols=["cell_id", "parameter", "voltage_mV", "family"])
v5r["cell_id"] = v5r["cell_id"].astype(str)
prof["cell_id"] = prof["cell_id"].astype(str)
fam_map = v5r.drop_duplicates(subset=["cell_id", "parameter", "voltage_mV"])
prof = prof.merge(fam_map, on=["cell_id", "parameter", "voltage_mV"], how="left")
prof.loc[prof["family"].isna() & (prof["source"] == "rec"), "family"] = "Kv"
PARAM2PROC = {"tau_act": "activation", "Vhalf_act": "activation", "k_act": "activation",
              "tau_inact": "inactivation", "Vhalf_inact": "inactivation", "k_inact": "inactivation",
              "tau_deact": "deactivation", "tau_deact_slow2": "deactivation", "tau_rec": "recovery"}
cp = pd.DataFrame({
    "source_big": "channelpedia", "dataset": prof["source"], "cell_id": prof["cell_id"],
    "channel": prof["channel"], "family": prof["family"], "species": "",
    "temp_C": prof["temp"], "parameter": prof["parameter"],
    "process": prof["parameter"].map(PARAM2PROC).fillna("other"),
    "voltage_mV": prof["voltage_mV"], "value": prof["value"],
    "se_pcov": prof["se_pcov"], "prof_lo": prof["prof_lo"], "prof_hi": prof["prof_hi"],
    "model": prof["model"], "p_model": prof["p"], "r2": prof["r2"],
    "qc_flag": np.where(prof["prof_lo"].isna(), "profile_fail_fallback_pcov", "ok"),
    "method": "profile_v1", "drug": "", "blind": False, "A_peak_pA": np.nan})
parts.append(cp)
print("  ", len(cp), "行")

# ---- Batch2 CiPA sweeps ----
print("[2] CiPA 五实验室 ...")
sw = pd.read_csv(OUT / "cipa2_sweeps.csv")
ci = pd.DataFrame({
    "source_big": "cipa_5lab",
    "dataset": sw["lab"] + np.where(sw["blind"], "_blind", ""),
    "cell_id": sw["cell"], "channel": "hERG", "family": "Kv", "species": "",
    "temp_C": np.nan, "parameter": "tau_deact", "process": "drug_block",
    "voltage_mV": np.nan, "value": sw["tau_tail_ms"], "se_pcov": np.nan,
    "prof_lo": sw["tau_prof_lo"], "prof_hi": sw["tau_prof_hi"],
    "model": "exp1_tail", "p_model": np.nan, "r2": sw["tail_r2"],
    "qc_flag": np.where(sw["qc_pass"], "ok", "qc_fail") + "|seg_" + sw["seg"].astype(str),
    "method": "profile_v1", "drug": sw["drug"], "blind": sw["blind"], "A_peak_pA": sw["A_peak"]})
parts.append(ci)
print("  ", len(ci), "行")

# ---- Batch3 Lei ----
print("[3] Lei 五温度 ...")
lei = pd.read_csv(OUT / "lei3_unified.csv").reindex(columns=COLUMNS)
parts.append(lei)
print("  ", len(lei), "行")

# ---- Batch4 Beattie ----
print("[4] Beattie 九细胞 ...")
b9 = pd.read_csv(OUT / "beattie4_unified.csv").reindex(columns=COLUMNS)
parts.append(b9)
print("  ", len(b9), "行")

# ---- Batch5a Nav1.5（剖面网格顶 9ms 的MLE撞界行标 tau_edge） ----
print("[5a] Nav1.5 ...")
nav = pd.read_csv(OUT / "nav5_unified.csv").reindex(columns=COLUMNS)
m_nav_edge = nav["value"] >= 8.99
nav.loc[m_nav_edge, "qc_flag"] = "tau_edge"
print("   Nav 撞界标记:", int(m_nav_edge.sum()), "行")
parts.append(nav)
print("  ", len(nav), "行")

# ---- Batch5b CaV1.2（MLE 撞 500ms 拟合上限 -> tau_edge；含 Ba2 全部 5 行） ----
print("[5b] CaV1.2 ...")
cav = pd.read_csv(OUT / "cav5_unified.csv").reindex(columns=COLUMNS)
m_cav_edge = cav["value"] >= 499.0
cav.loc[m_cav_edge, "qc_flag"] = "tau_edge"
print("   CaV 撞界标记:", int(m_cav_edge.sum()), "行（其中 Ba2:",
      int((m_cav_edge & cav["dataset"].astype(str).str.contains("Ba2")).sum()), "）")
parts.append(cav)
print("  ", len(cav), "行")

# ---- Batch5c IKs ----
print("[5c] IKs ...")
iks = pd.read_csv(OUT / "iks5_unified.csv").reindex(columns=COLUMNS)
parts.append(iks)
print("  ", len(iks), "行")

# ---- Batch6 文献/组统计锚 ----
print("[6] 文献锚（newCiPA/跨物种/ICG） ...")
lit = pd.read_csv(OUT / "literature6_anchors.csv").reindex(columns=COLUMNS)
parts.append(lit)
print("  ", len(lit), "行")

master = pd.concat(parts, ignore_index=True)
n0 = len(master)
master = master.drop_duplicates().reset_index(drop=True)  # 整行去重（Lei双发86行、IKs 1行）
print(f"整行去重: {n0} -> {len(master)}")
master["cell_id"] = master["cell_id"].astype(str)
master["drug"] = master["drug"].fillna("").astype(str)
fp = OUT / "master_unified_v1.csv"
master.to_csv(fp, index=False, encoding="utf-8-sig")
print("\n统一总表 ->", fp)
print("总行数:", len(master))
print(master.groupby("source_big").agg(rows=("value", "size"), cells=("cell_id", "nunique")))
# 重复自检（子集需含 drug/temp，否则把合法的重复记录误判为重复）
dup = master.duplicated(subset=["source_big", "cell_id", "parameter", "voltage_mV",
                                "value", "drug", "temp_C"]).sum()
print("键值重复行(含drug/temp):", dup)
