# -*- coding: utf-8 -*-
# stage4a_前向预言_稳态阻断.py
# ============================================================================
# Tier-1 第2条：模型前向预言全药池 1,546 细胞（皇冠证据升级）· Stage4a 稳态阻断
#
# 设计（口径在跑前冻结）：
#   目标量：逐细胞稳态阻断分数（stage2_final_cell_metrics.csv, qc_pass=True）
#   预言式：block_pred = C^h / (C^h + IC50^h)   （纯占据缩放，τ 不动——本模型核心设定）
#   浓度 C：官方 concentration-inhibition.xlsx 逐细胞累积末浓度（block_frac 对应末段）
#   三个预言臂（防循环）：
#     A  = 外部锚 newCiPA（hERGIC50[nM→µM], hERGh），名义浓度
#     A2 = 外部锚 + 实测修正浓度（meta 主表 Drug×Lab×Nominal→Corrected 中位映射）
#     B  = 留一实验室交叉（其余四室逐细胞 pooled Hill 拟合，预测被留出的室）
#   终审指标：Spearman ρ(pred,meas)、MAE、|err|≤0.15 占比、逐室分解
#   Stage3 联动终审：残差(pred−meas) vs tail_slope——模型预言"真平台"，
#     未平台化细胞的测量值应系统性低于预言（ρ>0 为机制确认）
# ============================================================================
import os, json, glob
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # law_channelpedia
DB = os.path.join(ROOT, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
ANCHOR_CSV = os.path.join(ROOT, "数据总库", "06_CiPA官方锚_7通道IC50_29药", "newCiPA.csv")

UNIT = {"um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3, "pm": 1e-6}

# ---------------------------------------------------------- 1. 全库巡览官方表
def inventory():
    rows = []
    hits = glob.glob(os.path.join(DB, "**", "concentration-inhibition.xlsx"), recursive=True)
    for p in hits:
        rel = os.path.relpath(p, DB).replace("\\", "/")
        parts = rel.split("/")
        lab = parts[0]
        drugdir = parts[2] if len(parts) > 3 else parts[-2]     # lab/phase/drugdir/...
        try:
            x = pd.read_excel(p)
        except Exception as e:
            rows.append(dict(lab=lab, drugdir=drugdir, err=str(e))); continue
        for _, r in x.iterrows():
            rows.append(dict(lab=lab, drugdir=drugdir,
                             CELLID=str(r.get("CELLID", "")).strip(),
                             LIQUID=str(r.get("LIQUID", "")),
                             CONC=r.get("CONC", np.nan), CONCU=str(r.get("CONCU", "")).strip(),
                             SBFL=str(r.get("SBFL", "")).strip(),
                             Change=r.get("Change", np.nan)))
    return pd.DataFrame(rows)

# ---------------------------------------------------------- 2. Hill 工具
def hill_occ(C, ic50, h):
    C = np.asarray(C, float)
    return C**h / (C**h + ic50**h)

def fit_hill(C, y):
    """pooled 变比例 Hill；返回 (ic50_uM, h) 或 None"""
    C = np.asarray(C, float); y = np.asarray(y, float)
    m = np.isfinite(C) & np.isfinite(y) & (C > 0)
    if m.sum() < 6 or len(np.unique(C[m])) < 2:
        return None
    Cc, yy = C[m], y[m]
    def resid(p):
        lic, h = p
        return hill_occ(Cc, 10**lic, h) - yy
    best = None
    for lic0 in (-3, -1, 1):
        for h0 in (0.7, 1.0, 1.5):
            try:
                r = least_squares(resid, [lic0, h0],
                                  bounds=([-6, 0.2], [3, 3.5]), method="trf")
                if best is None or r.cost < best.cost:
                    best = r
            except Exception:
                pass
    if best is None or not best.success:
        return None
    return 10**best.x[0], float(best.x[1])

# ---------------------------------------------------------- 3. 主流程
def main():
    print("[1/5] 巡览官方 concentration-inhibition.xlsx ...", flush=True)
    inv = inventory()
    bad = inv[inv.get("err", pd.Series(dtype=object)).notna()] if "err" in inv else pd.DataFrame()
    inv = inv[inv.get("err", pd.Series(dtype=object)).isna()] if "err" in inv else inv
    print(f"    官方行={len(inv)}  目录读取失败={len(bad)}", flush=True)
    print("    CONCU 取值:", inv.CONCU.str.lower().value_counts().to_dict(), flush=True)

    inv["conc_uM"] = [c * UNIT.get(str(u).lower(), np.nan) for c, u in zip(inv.CONC, inv.CONCU)]
    drug_rows = inv[(inv.SBFL.str.upper() != "Y") & np.isfinite(inv.conc_uM)].copy()

    # 逐细胞末浓度（累积给药：取该细胞最大浓度档）
    cellconc = (drug_rows.groupby(["lab", "drugdir", "CELLID"])
                .agg(conc_final_uM=("conc_uM", "max"),
                     conc_all=("conc_uM", lambda s: sorted(set(np.round(s, 6)))),
                     official_change_at_final=("Change", lambda s: float(s.iloc[np.argmax(drug_rows.loc[s.index, "conc_uM"].values)])))
                .reset_index())

    print("[2/5] 载入 stage2 终表并连接浓度 ...", flush=True)
    s2 = pd.read_csv(os.path.join(HERE, "stage2_final_cell_metrics.csv"))
    qc = s2[s2.qc_pass == True].copy()
    qc["basename"] = qc.cell.str.split("/").str[-1]
    qc["drug_real"] = qc.drug.str.replace("_blinded", "", regex=False)

    # 目录名映射：stage2 的 (lab,drug) ↔ 官方表的 (lab,drugdir)
    dirmap = {}
    for (lab, drug), g in qc.groupby(["lab", "drug"]):
        cand = inv[(inv.lab == lab) & (inv.drugdir.str.lower().str.contains(drug.replace("_blinded", "").lower()))].drugdir.unique()
        dirmap[(lab, drug)] = list(cand)
    # 连接
    recs = []
    miss = 0
    for _, r in qc.iterrows():
        dirs = dirmap[(r.lab, r.drug)]
        hit = cellconc[(cellconc.lab == r.lab) & (cellconc.drugdir.isin(dirs)) & (cellconc.CELLID == r.basename)]
        if len(hit) == 0:
            miss += 1
            recs.append(dict(cell=r.cell, lab=r.lab, drug=r.drug, drug_real=r.drug_real,
                             block_meas=r.block_frac, ref_A=r.ref_A_peak, conc_uM=np.nan, join="miss"))
        else:
            h0 = hit.iloc[0]
            recs.append(dict(cell=r.cell, lab=r.lab, drug=r.drug, drug_real=r.drug_real,
                             block_meas=r.block_frac, ref_A=r.ref_A_peak,
                             conc_uM=h0.conc_final_uM, conc_all=str(h0.conc_all),
                             official_change=h0.official_change_at_final,
                             join="ok" if len(hit) == 1 else "dup%d" % len(hit)))
    df = pd.DataFrame(recs)
    print(f"    QC 细胞={len(qc)}  浓度连接 miss={miss}  dup={(df["join"].str.startswith('dup')).sum()}", flush=True)
    print("    join 分布:", df["join"].value_counts().to_dict(), flush=True)

    # ---------------------------------------------------------- 3b. 修正浓度映射（meta 主表）
    meta = pd.read_excel(os.path.join(HERE, "hERG_data_for_meta-analysis.xlsx"))
    meta["drug_l"] = meta.Drug.str.lower()
    meta["lab_l"] = "lab_" + meta.Lab.str.extract(r"(\d+)")[0]
    corr = (meta.groupby(["lab_l", "drug_l", "NominalConc (uM)"])["CorrectedConc (uM)"]
            .median().reset_index())
    def corrected(row):
        m = corr[(corr.lab_l == row.lab) & (corr.drug_l == row.drug_real.lower()) &
                 (np.isclose(corr["NominalConc (uM)"], row.conc_uM, rtol=0.02, atol=1e-9))]
        return float(m["CorrectedConc (uM)"].iloc[0]) if len(m) else np.nan
    okc = df["join"] == "ok"
    df.loc[okc, "conc_corr_uM"] = df[okc].apply(corrected, axis=1)
    ncorr = np.isfinite(df.conc_corr_uM).sum()
    print(f"    修正浓度命中 {ncorr}/{okc.sum()}", flush=True)

    # ---------------------------------------------------------- 4. 三个预言臂
    print("[3/5] 构建预言臂 ...", flush=True)
    anch = pd.read_csv(ANCHOR_CSV)
    anch["drug_real"] = anch.drug.str.lower()
    A = {r.drug_real: (r.hERGIC50 / 1000.0, r.hERGh) for r in anch.itertuples()
         if np.isfinite(r.hERGIC50) and r.hERGIC50 > 0 and np.isfinite(r.hERGh)}
    print(f"    锚表覆盖 {len(A)} 药", flush=True)

    # 臂 A / A2
    df["pred_A"] = [hill_occ(c, *A[d]) if (d in A and np.isfinite(c)) else np.nan
                    for c, d in zip(df.conc_uM, df.drug_real)]
    df["pred_A2"] = [hill_occ(cc if np.isfinite(cc) else c, *A[d]) if (d in A and np.isfinite(c)) else np.nan
                     for c, cc, d in zip(df.conc_uM, df.conc_corr_uM, df.drug_real)]

    # 臂 B：留一实验室（同测量管线：stage2 逐细胞 block_frac，名义浓度）
    print("[4/5] 臂 B：留一实验室 pooled Hill ...", flush=True)
    df["pred_B"] = np.nan
    loo_log = []
    for (drug, lab), g in df[df["join"] == "ok"].groupby(["drug_real", "lab"]):
        train = df[(df.drug_real == drug) & (df.lab != lab) & (df["join"] == "ok") &
                   np.isfinite(df.block_meas) & np.isfinite(df.conc_uM)]
        fit = fit_hill(train.conc_uM.values, train.block_meas.values)
        if fit is None:
            loo_log.append(dict(drug=drug, lab=lab, status="fit_fail", n_train=len(train)))
            continue
        ic50, h = fit
        df.loc[g.index, "pred_B"] = hill_occ(g.conc_uM.values, ic50, h)
        loo_log.append(dict(drug=drug, lab=lab, status="ok", ic50=ic50, h=h, n_train=len(train)))

    # ---------------------------------------------------------- 5. 指标 + Stage3 联动
    print("[5/5] 指标汇总 ...", flush=True)
    def metrics(pred_col, name):
        m = df[np.isfinite(df[pred_col]) & np.isfinite(df.block_meas)]
        if len(m) < 10:
            return dict(arm=name, n=len(m))
        err = m[pred_col] - m.block_meas
        out = dict(arm=name, n=len(m),
                   rho=float(spearmanr(m[pred_col], m.block_meas).statistic),
                   MAE=float(np.abs(err).mean()),
                   within_015=float((np.abs(err) <= 0.15).mean()),
                   bias=float(err.mean()))
        per_lab = {}
        for lab, gg in m.groupby("lab"):
            e = gg[pred_col] - gg.block_meas
            per_lab[lab] = dict(n=len(gg), rho=float(spearmanr(gg[pred_col], gg.block_meas).statistic),
                                MAE=float(np.abs(err[gg.index - m.index[0]].mean())) if False else float(np.abs(e).mean()),
                                bias=float(e.mean()))
        out["per_lab"] = per_lab
        return out
    summ = [metrics("pred_A", "A_锚表_名义浓度"),
            metrics("pred_A2", "A2_锚表_修正浓度"),
            metrics("pred_B", "B_留一室交叉")]

    # Stage3 联动：残差 vs tail_slope
    s3 = pd.read_csv(os.path.join(HERE, "stage3_onset_metrics.csv"))
    df = df.merge(s3[["cell", "tail_slope", "tau_e4031_sweeps", "drug_span_sweeps"]], on="cell", how="left")
    tie = {}
    for arm in ("pred_A", "pred_A2", "pred_B"):
        m = df[np.isfinite(df[arm]) & np.isfinite(df.block_meas) & np.isfinite(df.tail_slope)]
        if len(m) > 30:
            resid = m[arm] - m.block_meas
            tie[arm] = dict(n=len(m),
                            rho_resid_tailslope=float(spearmanr(resid, m.tail_slope).statistic),
                            p=float(spearmanr(resid, m.tail_slope).pvalue))
            print(f"    [{arm}] 残差 vs 未平台化斜率: ρ={tie[arm]['rho_resid_tailslope']:+.3f} "
                  f"(p={tie[arm]['p']:.1e}, n={tie[arm]['n']})", flush=True)

    for s in summ:
        print(f"    [{s['arm']}] n={s.get('n')} ρ={s.get('rho', float('nan')):.3f} "
              f"MAE={s.get('MAE', float('nan')):.3f} ≤0.15占{s.get('within_015', float('nan')):.2f} "
              f"bias={s.get('bias', float('nan')):+.3f}", flush=True)
        for lab, l in s.get("per_lab", {}).items():
            print(f"        {lab}: n={l['n']} ρ={l['rho']:.3f} MAE={l['MAE']:.3f} bias={l['bias']:+.3f}", flush=True)

    df.to_csv(os.path.join(HERE, "stage4a_per_cell_predictions.csv"), index=False)
    json.dump(dict(arms=summ, loo_log=loo_log, stage3_tie=tie,
                   join_counts=df["join"].value_counts().to_dict()),
              open(os.path.join(HERE, "stage4a_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
