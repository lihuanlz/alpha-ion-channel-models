# -*- coding: utf-8 -*-
"""
统一重算 Batch 1：Channelpedia 全家族装入统一总表
输入：features/profile_all_rows.csv（276,039 行，剖面似然封盘）
     features/features_corpus_v5r.csv（family 映射）
     features/features_recovery_v5.csv（rec 的 channel/family）
输出：统一重算_2026-11-11/master_unified_v1.csv（第一批）
schema:
  source_big   一级来源：channelpedia / cipa_5lab / lei2019 / nav15_tjuev / iks_lit / literature_anchor
  dataset      二级数据集：v2 / hcn / rec / lab_1..5 / herg27oc1 ...
  cell_id      细胞标识
  channel      通道名
  family       家族：Kv/K2P/Kir/HCN/Nav/CaV/IKs/other
  species      物种（本批从 census 补，缺则空）
  temp_C       温度
  parameter    参数名：tau_act/Vhalf_act/k_act/tau_inact/tau_deact/tau_rec/IC50 ...
  process      过程：activation/inactivation/deactivation/recovery/drug_block/steady_state
  voltage_mV   电压（电压无关参数为 NaN）
  value        点估计
  se_pcov      协方差 SE
  prof_lo      剖面似然95%CI下界
  prof_hi      剖面似然95%CI上界
  model        模型
  r2           拟合优度
  qc_flag      QC
  method       提取方法版本：profile_v1（剖面似然）/ legacy / literature
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # law_channelpedia
FEAT = ROOT / "features"
OUT = ROOT / "统一重算_2026-11-11"
OUT.mkdir(exist_ok=True)

PARAM2PROC = {
    "tau_act": "activation", "Vhalf_act": "activation", "k_act": "activation",
    "tau_inact": "inactivation", "Vhalf_inact": "inactivation", "k_inact": "inactivation",
    "tau_deact": "deactivation", "tau_deact_slow2": "deactivation",
    "tau_rec": "recovery",
}

print("[1/4] 读 profile_all_rows ...")
prof = pd.read_csv(FEAT / "profile_all_rows.csv")
print("  rows:", len(prof))

print("[2/4] 建 family 映射（v5r + recovery）...")
v5r = pd.read_csv(FEAT / "features_corpus_v5r.csv",
                  usecols=["cell_id", "parameter", "voltage_mV", "family"])
rec = pd.read_csv(FEAT / "features_recovery_v5.csv")
rec_fam = rec[["cell_id", "channel"]].drop_duplicates()
rec_fam["family"] = "Kv"  # recovery 表全为 Kv（存档口径）
# v5r join 键
v5r["cell_id"] = v5r["cell_id"].astype(str)
prof["cell_id"] = prof["cell_id"].astype(str)
fam_map = v5r.drop_duplicates(subset=["cell_id", "parameter", "voltage_mV"])
prof = prof.merge(fam_map.rename(columns={"family": "family"}),
                  on=["cell_id", "parameter", "voltage_mV"], how="left")
# rec 行补 family
m = prof["family"].isna() & (prof["source"] == "rec")
prof.loc[m, "family"] = "Kv"
print("  family 缺失:", prof["family"].isna().sum())

print("[3/4] 统一字段 ...")
prof["source_big"] = "channelpedia"
prof["dataset"] = prof["source"]
prof["species"] = ""
prof["temp_C"] = prof["temp"]
prof["process"] = prof["parameter"].map(PARAM2PROC).fillna("other")
prof["qc_flag"] = np.where(prof["prof_lo"].isna(), "profile_fail_fallback_pcov", "ok")
prof["method"] = "profile_v1"
prof["p_model"] = prof["p"]

keep = ["source_big", "dataset", "cell_id", "channel", "family", "species",
        "temp_C", "parameter", "process", "voltage_mV", "value",
        "se_pcov", "prof_lo", "prof_hi", "model", "p_model", "r2",
        "qc_flag", "method"]
master = prof[keep].copy()

print("[4/4] 落盘 ...")
out = OUT / "master_unified_v1.csv"
master.to_csv(out, index=False, encoding="utf-8-sig")
print("  ->", out, len(master), "行")
print()
print("== Batch1 质检 ==")
print(master.groupby("family")["cell_id"].nunique())
print(master["parameter"].value_counts().to_dict())
print("QC 非 ok 行:", (master["qc_flag"] != "ok").sum())
