# -*- coding: utf-8 -*-
"""
Stage2a：官方 meta-analysis 主表快分析
1) NominalConc vs CorrectedConc 重算跨实验室 fold —— lab_2 系统性偏移能否被浓度修正解释
2) 加药方式（perfusion/pipetting）对抑制率的影响
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
meta = pd.read_excel(os.path.join(OUT, "hERG_data_for_meta-analysis.xlsx"))
meta["Drug"] = meta["Drug"].astype(str).str.strip()
meta["Lab"] = meta["Lab"].astype(str).str.strip()

def hill_fn(logC, logIC50, hill):
    return 1.0 / (1.0 + 10 ** ((logIC50 - logC) * hill))

def fit_grp(conc, inhib):
    m = np.isfinite(conc) & np.isfinite(inhib) & (conc > 0)
    c, y = conc[m], inhib[m]
    if len(c) < 5 or len(np.unique(c)) < 3:
        return np.nan, np.nan, len(c)
    logC = np.log10(c)
    try:
        popt, _ = curve_fit(hill_fn, logC, y, p0=[np.median(logC), 1.0],
                            bounds=([logC.min() - 2, 0.1], [logC.max() + 2, 5.0]), maxfev=8000)
        return 10 ** popt[0], popt[1], len(c)
    except Exception:
        return np.nan, np.nan, len(c)

rows = []
for (drug, lab, method), g in meta.groupby(["Drug", "Lab", "ExperimentID"]):
    ic_n, h_n, n = fit_grp(g["NominalConc (uM)"].to_numpy(float), g["Fractional inhibition"].to_numpy(float))
    ic_c, h_c, _ = fit_grp(g["CorrectedConc (uM)"].to_numpy(float), g["Fractional inhibition"].to_numpy(float))
    rows.append({"Drug": drug, "Lab": lab, "Method": method, "n_meas": n,
                 "IC50_nominal_uM": ic_n, "hill_nominal": h_n,
                 "IC50_corrected_uM": ic_c, "hill_corrected": h_c})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "stage2a_meta_fits.csv"), index=False, encoding="utf-8-sig")

# ---- 1) 跨实验室 fold：nominal vs corrected（取每 药×实验室 的主方法组 = 测量数最多的组）----
main_rows = []
for (drug, lab), g in df.groupby(["Drug", "Lab"]):
    main_rows.append(g.loc[g["n_meas"].idxmax()])
dmain = pd.DataFrame(main_rows)

fold_rows = []
for drug, g in dmain.groupby("Drug"):
    gn = g.dropna(subset=["IC50_nominal_uM"])
    gc = g.dropna(subset=["IC50_corrected_uM"])
    if len(gn) >= 2:
        fold_rows.append({"Drug": drug, "n_labs": len(gn),
                          "fold_nominal": gn["IC50_nominal_uM"].max() / gn["IC50_nominal_uM"].min(),
                          "fold_corrected": gc["IC50_corrected_uM"].max() / gc["IC50_corrected_uM"].min() if len(gc) >= 2 else np.nan})
dff = pd.DataFrame(fold_rows).sort_values("fold_nominal", ascending=False)
dff.to_csv(os.path.join(OUT, "stage2a_fold_nominal_vs_corrected.csv"), index=False, encoding="utf-8-sig")

# ---- 各实验室偏移（几何平均，相对跨实验室中位）----
print("=== 各实验室系统性偏移（nominal → corrected）===")
labshift = []
for lab, gl in dmain.groupby("Lab"):
    rats_n, rats_c = [], []
    for drug, gd in dmain.groupby("Drug"):
        med_n = gd["IC50_nominal_uM"].median()
        med_c = gd["IC50_corrected_uM"].median()
        rn = gl[gl["Drug"] == drug]
        if len(rn) and pd.notna(rn["IC50_nominal_uM"].iloc[0]) and pd.notna(med_n):
            rats_n.append(rn["IC50_nominal_uM"].iloc[0] / med_n)
        if len(rn) and pd.notna(rn["IC50_corrected_uM"].iloc[0]) and pd.notna(med_c):
            rats_c.append(rn["IC50_corrected_uM"].iloc[0] / med_c)
    if rats_n:
        gn_ = 10 ** np.mean(np.log10(rats_n))
        gc_ = 10 ** np.mean(np.log10(rats_c)) if rats_c else np.nan
        labshift.append({"Lab": lab, "geo_shift_nominal": gn_, "geo_shift_corrected": gc_, "n_drugs": len(rats_n)})
        print(f"  {lab:8s}: {gn_:.2f}x → 修正后 {gc_:.2f}x  (n={len(rats_n)})")
pd.DataFrame(labshift).to_csv(os.path.join(OUT, "stage2a_lab_shift.csv"), index=False, encoding="utf-8-sig")

# ---- 2) 修正浓度/名义浓度比分布（各实验室浓度校准质量）----
meta["corr_ratio"] = meta["CorrectedConc (uM)"] / meta["NominalConc (uM)"]
print("\n=== CorrectedConc/NominalConc 比值（中位 [IQR]）===")
cr = meta.groupby("Lab")["corr_ratio"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "count"])
cr.columns = ["median", "q25", "q75", "n"]
print(cr.round(4).to_string())
cr.to_csv(os.path.join(OUT, "stage2a_conc_correction_ratio.csv"), encoding="utf-8-sig")

# ---- 3) 加药方式效应 ----
print("\n=== 加药方式分布 ===")
print(meta["ExperimentID"].value_counts().to_string())
# 同药同浓度下不同方法的抑制率差异
print("\n=== fold 总账 ===")
print(f"跨实验室 fold 中位：名义 {dff['fold_nominal'].median():.2f}x → 修正 {dff['fold_corrected'].median():.2f}x")
print(f"fold_nominal 最大: {dff.iloc[0]['Drug']} {dff.iloc[0]['fold_nominal']:.1f}x")
print("DONE")
