# -*- coding: utf-8 -*-
"""
Stage1 表级分析 v2：CiPA 全药池 134 个 concentration-inhibition.xlsx
关键事实：HESI BAA 每细胞只测 1-2 个浓度 → 浓度-反应曲线是跨细胞拼的。
故主拟合 = 目录级 pooled（对齐官方 ic50nh 方法），另做浓度中位数稳健对照。
输出：逐细胞抑制率表 / 浓度级统计 / 目录级汇总+官方对拍 / 跨实验室 fold / newCiPA 锚对拍 / 异常挖掘。
"""
import os, json, glob, time, argparse, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from multiprocessing import Pool
warnings.filterwarnings("ignore")

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
HUB = os.path.join(LAW, "数据总库")
CIPA = os.path.join(HUB, "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
OUT = os.path.join(LAW, "分析_全药池_2026-11-10")
os.makedirs(OUT, exist_ok=True)

def hill_fn(logC, logIC50, hill):
    return 1.0 / (1.0 + 10 ** ((logIC50 - logC) * hill))

def fit_pooled(logC, y, w=None):
    lo, hi = logC.min() - 2, logC.max() + 2
    try:
        popt, pcov = curve_fit(hill_fn, logC, y, p0=[np.median(logC), 1.0],
                               sigma=w, bounds=([lo, 0.1], [hi, 5.0]), maxfev=8000)
        perr = np.sqrt(np.diag(pcov))
        rss = float(np.sum((y - hill_fn(logC, *popt)) ** 2))
        return {"fit_ok": True, "logIC50": float(popt[0]), "logIC50_se": float(perr[0]),
                "IC50_uM": float(10 ** popt[0]), "hill": float(popt[1]),
                "hill_se": float(perr[1]), "rss": rss}
    except Exception as e:
        return {"fit_ok": False, "fail": type(e).__name__}

def proc_file(f):
    try:
        rel = os.path.relpath(f, CIPA)
        parts = rel.split(os.sep)
        lab, drug = parts[0], parts[2]
        blind = "blind" in rel.lower()
        xb = pd.read_excel(f, sheet_name="ByCell")
        xi = pd.read_excel(f, sheet_name="ic50nh")
        xb["SBFL"] = xb["SBFL"].astype(str)
        # ---- 单位统一：ic50nh 与 ByCell 单位一致（已全库验证），统一到 uM ----
        unit = str(xi["CONCU"].iloc[0]).strip() if len(xi) else str(xb["CONCU"].astype(str).mode().iloc[0]).strip()
        factor = {"uM": 1.0, "µM": 1.0, "μM": 1.0, "nM": 1e-3, "pM": 1e-6, "mM": 1e3}.get(unit)
        if factor is None:
            raise ValueError(f"未知浓度单位 {unit!r}")
        xb["CONC"] = xb["CONC"].astype(float) * factor
        dr = xb[xb["SBFL"] != "Y"].copy()
        cr = xb[xb["SBFL"] == "Y"]
        dr["inhib"] = 1.0 - dr["Drug"].astype(float) / dr["Control"].astype(float)
        e4031_inhib = float((1 - cr["Drug"].astype(float) / cr["Control"].astype(float)).mean()) if len(cr) else np.nan

        # ---- 逐浓度统计 ----
        conc_stats = []
        for conc, g in dr.groupby("CONC"):
            v = g["inhib"].to_numpy(float)
            v = v[np.isfinite(v)]
            conc_stats.append({"conc_uM": float(conc), "n": int(len(v)),
                               "med": float(np.median(v)), "mean": float(np.mean(v)),
                               "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                               "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0})
        cs = pd.DataFrame(conc_stats).sort_values("conc_uM")

        # ---- pooled 拟合（全部逐细胞点，对齐官方） ----
        Call = dr["CONC"].to_numpy(float); yall = dr["inhib"].to_numpy(float)
        m = np.isfinite(Call) & np.isfinite(yall) & (Call > 0)
        pooled = fit_pooled(np.log10(Call[m]), yall[m]) if m.sum() >= 4 else {"fit_ok": False, "fail": f"pts={m.sum()}"}
        # ---- 浓度中位数稳健拟合（IQR 加权） ----
        if len(cs) >= 3:
            w = 1.0 / np.maximum((cs["q75"] - cs["q25"]).to_numpy(), 0.02)
            medfit = fit_pooled(np.log10(cs["conc_uM"].to_numpy()), cs["med"].to_numpy(), w)
        else:
            medfit = {"fit_ok": False, "fail": f"concs={len(cs)}"}

        off = {}
        if len(xi):
            row = xi.iloc[0]
            off = {"off_IC50_uM": float(row["IC50"]) * factor, "off_IC50_LCL": float(row["IC50_LCL"]) * factor,
                   "off_IC50_UCL": float(row["IC50_UCL"]) * factor, "off_nh": float(row["nh"]),
                   "off_nh_LCL": float(row["nh_LCL"]), "off_nh_UCL": float(row["nh_UCL"]), "unit_orig": unit}
        return {"ok": True, "lab": lab, "drug": drug, "blind": blind, "file": rel, "unit": unit,
                "n_cells": int(dr[["EXPID", "CELLID"]].drop_duplicates().shape[0]),
                "n_meas": int(m.sum()), "n_e4031": int(len(cr)), "e4031_inhib": e4031_inhib,
                "conc_stats": conc_stats, "pooled": pooled, "medfit": medfit, "official": off}
    except Exception as e:
        return {"ok": False, "file": f, "err": f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(CIPA, "lab_*", "*", "*", "subtracted", "tables", "concentration-inhibition.xlsx")))
    print(f"subtracted={len(files)}", flush=True)
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(proc_file, files))
    print(f"解析 {time.time()-t0:.0f}s", flush=True)
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(f"成功 {len(ok)} / 失败 {len(bad)}")
    for b in bad[:10]:
        print("  FAIL:", b.get("file"), b.get("err"))
    with open(os.path.join(OUT, "stage1_raw_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False)

    # ---- 浓度级统计表 ----
    rows = []
    for r in ok:
        for c in r["conc_stats"]:
            rows.append({"lab": r["lab"], "drug": r["drug"], "blind": r["blind"], **c})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stage1_conc_level_stats.csv"), index=False, encoding="utf-8-sig")

    # ---- 目录级汇总 ----
    dl = []
    for r in ok:
        p, mf, off = r["pooled"], r["medfit"], r["official"]
        cs = pd.DataFrame(r["conc_stats"])
        row = {"lab": r["lab"], "drug": r["drug"], "blind": r["blind"],
               "n_cells": r["n_cells"], "n_meas": r["n_meas"], "n_conc": len(cs),
               "e4031_inhib": r["e4031_inhib"],
               "IC50_uM": p.get("IC50_uM"), "logIC50_se": p.get("logIC50_se"),
               "hill": p.get("hill"), "hill_se": p.get("hill_se"), "fit_ok": p.get("fit_ok"),
               "medfit_IC50_uM": mf.get("IC50_uM"), "medfit_hill": mf.get("hill"),
               "plateau_top_med": float(cs["med"].iloc[-1]) if len(cs) else np.nan,
               "top_conc_uM": float(cs["conc_uM"].iloc[-1]) if len(cs) else np.nan,
               "low_conc_med": float(cs["med"].iloc[0]) if len(cs) else np.nan,
               "mono_spearman": float(spearmanr(cs["conc_uM"], cs["med"]).statistic) if len(cs) >= 3 else np.nan,
               "max_conc_spread_IQR": float((cs["q75"] - cs["q25"]).iloc[-1]) if len(cs) else np.nan}
        row.update(off)
        dl.append(row)
    dfl = pd.DataFrame(dl)
    dfl.to_csv(os.path.join(OUT, "stage1_drug_lab_summary.csv"), index=False, encoding="utf-8-sig")

    # ---- 跨实验室 fold-range ----
    xr = []
    for drug, g in dfl[~dfl["blind"]].groupby("drug"):
        g2 = g.dropna(subset=["IC50_uM"])
        if len(g2) >= 2:
            meds = g2["IC50_uM"].to_numpy()
            off = g2["off_IC50_uM"].dropna().to_numpy()
            xr.append({"drug": drug, "n_labs": int(len(g2)),
                       "fold_ours": float(meds.max() / meds.min()),
                       "fold_official": float(off.max() / off.min()) if len(off) >= 2 else np.nan,
                       "pool_med_uM": float(np.median(meds)),
                       "hill_range": f"{g2['hill'].min():.2f}-{g2['hill'].max():.2f}"})
    dfx = pd.DataFrame(xr)
    if len(dfx):
        dfx = dfx.sort_values("fold_ours", ascending=False)
    dfx.to_csv(os.path.join(OUT, "stage1_crosslab_fold.csv"), index=False, encoding="utf-8-sig")

    # ---- newCiPA 锚对拍 ----
    anchor_p = os.path.join(HUB, "06_CiPA官方锚_7通道IC50_29药", "newCiPA.csv")
    if os.path.exists(anchor_p):
        anc = pd.read_csv(anchor_p)
        anc.columns = [c.strip() for c in anc.columns]
        herg_col = [c for c in anc.columns if c.strip().lower() == "hergic50"]
        drug_col = anc.columns[0]
        amap = {}
        if herg_col:
            for _, row in anc.iterrows():
                try:
                    v = float(row[herg_col[0]])
                    if 0 < v < 1e7:
                        amap[str(row[drug_col]).strip().lower()] = v / 1000.0
                except Exception:
                    pass
        comp = []
        for drug, g in dfl[~dfl["blind"]].groupby("drug"):
            key = drug.strip().lower()
            if key in amap:
                ours = float(np.nanmedian(g["IC50_uM"]))
                comp.append({"drug": drug, "newCiPA_hERG_uM": amap[key], "ours_med_uM": ours,
                             "fold": float(max(ours, amap[key]) / min(ours, amap[key]))})
        dfa = pd.DataFrame(comp)
        dfa.to_csv(os.path.join(OUT, "stage1_vs_newCiPA_anchor.csv"), index=False, encoding="utf-8-sig")
        if len(dfa):
            print(f"newCiPA 锚对拍 {len(dfa)} 药，中位 fold={dfa['fold'].median():.2f}，最大={dfa['fold'].max():.2f}")

    # ---- 异常挖掘 ----
    an = []
    for _, r in dfl.iterrows():
        flags = []
        # 官方 hill≠1（>5 或 CI 退化为零宽 = 拟合数值灾难，单独标注）
        if pd.notna(r.get("off_nh")) and (r["off_nh"] > 5 or (pd.notna(r.get("off_nh_LCL")) and abs(r["off_nh_UCL"] - r["off_nh_LCL"]) < 1e-6)):
            flags.append(f"官方拟合数值灾难(hill={r['off_nh']:.1f} CI退化)")
        elif pd.notna(r.get("off_nh_LCL")) and not (r["off_nh_LCL"] <= 1 <= r["off_nh_UCL"]):
            flags.append(f"官方hill={r['off_nh']:.2f}[{r['off_nh_LCL']:.2f},{r['off_nh_UCL']:.2f}]≠1")
        if pd.notna(r.get("hill")) and abs(r["hill"] - 1) > 0.5:
            flags.append(f"pooled_hill={r['hill']:.2f}±{r['hill_se']:.2f}")
        # 平台判据：仅当最高测试浓度 ≥10× IC50（曲线已走完）时 <80% 才算真不完全阻断
        if (pd.notna(r.get("plateau_top_med")) and r["plateau_top_med"] < 0.8
                and pd.notna(r.get("IC50_uM")) and pd.notna(r.get("top_conc_uM"))
                and r["top_conc_uM"] >= 10 * r["IC50_uM"]):
            flags.append(f"不完全阻断:顶浓度({r['top_conc_uM']:.0f}uM≥10xIC50)抑制仅={r['plateau_top_med']*100:.0f}%")
        if pd.notna(r.get("low_conc_med")) and r["low_conc_med"] < -0.05:
            flags.append(f"低浓度抑制={r['low_conc_med']*100:.1f}%")
        if pd.notna(r.get("e4031_inhib")) and r["e4031_inhib"] < 0.8:
            flags.append(f"E4031对照={r['e4031_inhib']*100:.0f}%")
        if pd.notna(r.get("off_IC50_uM")) and pd.notna(r.get("IC50_uM")) and r["off_IC50_uM"] > 0:
            fo = max(r["IC50_uM"], r["off_IC50_uM"]) / min(r["IC50_uM"], r["off_IC50_uM"])
            if fo > 3:
                flags.append(f"我们vs官方差{fo:.1f}x")
        if pd.notna(r.get("max_conc_spread_IQR")) and r["max_conc_spread_IQR"] > 0.5:
            flags.append(f"顶浓度细胞间IQR={r['max_conc_spread_IQR']:.2f}")
        if flags:
            an.append({"lab": r["lab"], "drug": r["drug"], "blind": r["blind"], "flags": " | ".join(flags)})
    pd.DataFrame(an).to_csv(os.path.join(OUT, "stage1_anomalies.csv"), index=False, encoding="utf-8-sig")

    # ---- 实验室离群 ----
    out2 = []
    for drug, g in dfl[~dfl["blind"]].groupby("drug"):
        g2 = g.dropna(subset=["IC50_uM"])
        if len(g2) >= 3:
            med_all = g2["IC50_uM"].median()
            for _, rr in g2.iterrows():
                if max(rr["IC50_uM"], med_all) / min(rr["IC50_uM"], med_all) > 3:
                    out2.append({"drug": drug, "lab": rr["lab"], "IC50_uM": rr["IC50_uM"],
                                 "pool_med_uM": float(med_all)})
    if out2:
        pd.DataFrame(out2).to_csv(os.path.join(OUT, "stage1_lab_outliers.csv"), index=False, encoding="utf-8-sig")

    # ---- 盲法 vs 非盲 ----
    bl = dfl[dfl["drug"].str.lower().isin(["dofetilide", "ondansetron"])]
    bl_out = []
    for (lab, drug), g in bl.groupby(["lab", "drug"]):
        if g["blind"].nunique() == 2:
            nb = g[~g["blind"]]["IC50_uM"]
            bd = g[g["blind"]]["IC50_uM"]
            if len(nb) and len(bd) and pd.notna(nb.iloc[0]) and pd.notna(bd.iloc[0]):
                a_, b_ = float(nb.iloc[0]), float(bd.iloc[0])
                bl_out.append({"lab": lab, "drug": drug, "unblind_uM": a_, "blind_uM": b_,
                               "fold": float(max(a_, b_) / min(a_, b_))})
    if bl_out:
        pd.DataFrame(bl_out).to_csv(os.path.join(OUT, "stage1_blind_vs_unblind.csv"), index=False, encoding="utf-8-sig")

    print("\n========== Stage1 v2 总览 ==========")
    print(f"目录 {len(ok)}；pooled 拟合成功 {int(dfl['fit_ok'].sum())}/{len(dfl)}")
    if len(dfx):
        print(f"跨实验室药 {len(dfx)}；fold_ours 中位={dfx['fold_ours'].median():.2f} 最大={dfx['fold_ours'].max():.2f}")
        print(f"fold_official 中位={dfx['fold_official'].median():.2f}")
    print(f"异常目录 {len(an)}/{len(dfl)}；实验室离群 {len(out2)}；盲法对拍 {len(bl_out)}")
    print("DONE")

if __name__ == "__main__":
    main()
