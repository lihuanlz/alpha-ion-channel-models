# -*- coding: utf-8 -*-
"""
Stage2 波形级分析：30 药全池，检验药物是否动 hERG 动力学常数（τ）还是只动占据（幅度）
协议（已目验确认）：1s sweep = 基线(0-280ms) + 去极化阶跃(~290-800ms) + 复极化斜坡(~800-890ms) + 去激活尾(890-1000ms)
每细胞：对照期(vehicle) → 药物递增浓度块 → E-4031 满阻断
指标：A_plateau(700-790ms均值) / A_peak(820-900ms峰) / τ_tail(峰后10ms到990ms单指数去激活τ) / 形状比 A_peak/A_plateau / 阻断起效τ(峰幅度 vs ELTIME)
QC：对照漂移>30% 或对照 A_peak<50pA 的细胞剔除
断点：按药物目录 ckpt.jsonl 追加；--budget 自停可续跑
"""
import os, json, glob, time, argparse, warnings, zipfile, io
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool
warnings.filterwarnings("ignore")

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
CIPA = os.path.join(LAW, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
OUT = os.path.join(LAW, "分析_全药池_2026-11-10")
CKPT = os.path.join(OUT, "stage2_ckpt.jsonl")

UNIT = {"uM": 1.0, "µM": 1.0, "μM": 1.0, "nM": 1e-3, "pM": 1e-6, "mM": 1e3}

def exp_off(t, a, tau, c):
    return a * np.exp(-(t - t[0]) / tau) + c

def trace_metrics(t, y):
    """单条 1s sweep 的指标。t: ms 数组, y: pA 数组"""
    m_plateau = (t >= 700) & (t <= 790)
    m_peak = (t >= 820) & (t <= 905)
    if m_plateau.sum() < 5 or m_peak.sum() < 5:
        return None
    A_plateau = float(np.mean(y[m_plateau]))
    ipk = np.argmax(y[m_peak])
    t_peak = float(t[m_peak][ipk]); A_peak = float(y[m_peak][ipk])
    # 去激活尾：峰后 +8ms 到 995ms
    m_tail = (t >= t_peak + 8) & (t <= 995)
    tau_tail = np.nan; tail_r2 = np.nan
    if m_tail.sum() >= 20:
        tt = t[m_tail]; yy = y[m_tail]
        try:
            p0 = [yy[0] - yy[-1], 150.0, yy[-1]]
            popt, _ = curve_fit(exp_off, tt, yy, p0=p0, maxfev=3000,
                                bounds=([-1e5, 5, -1e5], [1e5, 3000, 1e5]))
            pred = exp_off(tt, *popt)
            ss_res = float(np.sum((yy - pred) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            tau_tail = float(popt[1]); tail_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            pass
    return {"A_plateau": A_plateau, "A_peak": A_peak, "t_peak": t_peak,
            "tau_tail_ms": tau_tail, "tail_r2": tail_r2,
            "shape_ratio": A_peak / A_plateau if abs(A_plateau) > 1e-9 else np.nan}

def proc_cell(args):
    """处理一个细胞 zip + 其 resultsWide 行 → 逐 trace 指标 + 块级汇总"""
    zpath, rw_csv, lab, drug, blind, unit_factor = args
    try:
        cell = os.path.basename(zpath).replace(".csv.zip", "")
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        t = df["t_ms"].to_numpy(float)
        rw = rw_csv.sort_values("TRACENUM")
        n_tr = df.shape[1] - 1
        recs = []
        for _, row in rw.iterrows():
            tr = int(row["TRACENUM"])
            if tr < 1 or tr > n_tr:
                continue
            col = f"trace_#{tr}_current_pa"
            y = df[col].to_numpy(float)
            mm = trace_metrics(t, y)
            if mm is None:
                continue
            mm.update({"TRACENUM": tr, "ELTIME_s": float(row["ELTIME"]),
                       "LIQUID": row["LIQUID"], "CONC_uM": float(row["CONC"]) * unit_factor,
                       "CTLFL": str(row.get("CTLFL", "")) == "Y"})
            recs.append(mm)
        if not recs:
            return {"ok": False, "cell": cell, "err": "no traces matched"}
        tr_df = pd.DataFrame(recs)
        # ---- 对照参考：vehicle 期（CTLFL==Y 或 LIQUID 含 Bath/Vehicle）最后 5 条 ----
        veh = tr_df[(tr_df["CTLFL"]) | (tr_df["LIQUID"].str.contains("Bath|Vehicle", caseless=True, na=False))]
        drug_rows = tr_df[tr_df["LIQUID"].str.strip().str.lower() == drug.strip().lower()]
        e4031 = tr_df[tr_df["LIQUID"].str.contains("E-4031", caseless=True, na=False)]
        if len(veh) < 3:
            return {"ok": False, "cell": cell, "err": f"vehicle traces={len(veh)}"}
        veh_last = veh.tail(5)
        ref = {"A_peak": float(veh_last["A_peak"].median()), "A_plateau": float(veh_last["A_plateau"].median()),
               "tau_tail_ms": float(veh_last["tau_tail_ms"].median()),
               "shape_ratio": float(veh_last["shape_ratio"].median())}
        drift = abs(float(veh.head(3)["A_peak"].median()) - ref["A_peak"]) / max(abs(ref["A_peak"]), 1e-9)
        qc_pass = (drift <= 0.30) and (ref["A_peak"] >= 50)
        # ---- 浓度块：稳态(每块最后3条) + 起效τ(块内 A_peak vs ELTIME 单指数) ----
        blocks = []
        for conc, g in drug_rows.groupby("CONC_uM"):
            g = g.sort_values("ELTIME_s")
            ss = g.tail(3)
            onset_tau = np.nan
            if len(g) >= 6 and g["ELTIME_s"].ptp() > 5:
                try:
                    gg = g[g["A_peak"] > -1e9]
                    y0 = gg["A_peak"].iloc[0]; yinf = gg["A_peak"].iloc[-3:].mean()
                    def decay(tt, tau):
                        return yinf + (y0 - yinf) * np.exp(-(tt - gg["ELTIME_s"].iloc[0]) / tau)
                    popt, _ = curve_fit(decay, gg["ELTIME_s"].to_numpy(), gg["A_peak"].to_numpy(),
                                        p0=[60.0], bounds=([1.0], [5000.0]), maxfev=3000)
                    onset_tau = float(popt[0])
                except Exception:
                    pass
            blocks.append({"conc_uM": float(conc), "n_traces": int(len(g)),
                           "A_peak_ss": float(ss["A_peak"].median()),
                           "A_plateau_ss": float(ss["A_plateau"].median()),
                           "tau_tail_ss": float(ss["tau_tail_ms"].median()),
                           "shape_ratio_ss": float(ss["shape_ratio"].median()),
                           "onset_tau_s": onset_tau,
                           "block_frac": 1 - float(ss["A_peak"].median()) / ref["A_peak"] if ref["A_peak"] > 0 else np.nan})
        e4031_resid = float(e4031["A_peak"].median()) / ref["A_peak"] if len(e4031) and ref["A_peak"] > 0 else np.nan
        return {"ok": True, "lab": lab, "drug": drug, "blind": blind, "cell": cell,
                "n_traces": int(len(tr_df)), "qc_pass": bool(qc_pass), "control_drift": float(drift),
                "ref": ref, "blocks": blocks, "e4031_resid": e4031_resid}
    except Exception as e:
        return {"ok": False, "cell": zpath, "err": f"{type(e).__name__}: {e}"}

def collect_tasks():
    """遍历全部药物目录，构建 (zpath, rw_rows, lab, drug, blind, factor) 任务列表"""
    tasks = []
    dirs = sorted(glob.glob(os.path.join(CIPA, "lab_*", "*", "*", "subtracted")))
    for d in dirs:
        rel = os.path.relpath(d, CIPA); parts = rel.split(os.sep)
        lab, drug = parts[0], parts[2]
        blind = "blind" in rel.lower()
        rw_p = os.path.join(d, "resultsWide_converted.csv")
        if not os.path.exists(rw_p):
            continue
        rw = pd.read_csv(rw_p)
        unit = str(rw["CONCU"].dropna().mode().iloc[0]) if rw["CONCU"].notna().any() else "uM"
        factor = UNIT.get(unit.strip(), 1.0)
        ted = os.path.join(d, "ted")
        if not os.path.isdir(ted):
            continue
        for zp in sorted(glob.glob(os.path.join(ted, "*.csv.zip"))):
            cell = os.path.basename(zp).replace(".csv.zip", "")
            # resultsWide 的 CELLID 与 zip 文件名对应（azimilide_7_A Azi_1 → CELLID=7_A Azi_1）
            sub = rw[rw["CELLID"] == cell]
            if len(sub) == 0:
                # 尝试去掉实验前缀的匹配
                sub = rw[rw["CELLID"].str.contains(cell.split(" ", 0)[0], na=False)]
            if len(sub):
                tasks.append((zp, sub[["TRACENUM", "ELTIME", "LIQUID", "CONC", "CTLFL"]].copy(), lab, drug, blind, factor))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tasks", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    print("收集任务...", flush=True)
    tasks = collect_tasks()
    print(f"细胞任务总数 {len(tasks)}", flush=True)
    done_cells = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        done_cells.add(r["cell"])
                except Exception:
                    pass
        print(f"续跑：已完成 {len(done_cells)}", flush=True)
    todo = [t for t in tasks if os.path.basename(t[0]).replace(".csv.zip", "") not in done_cells]
    if a.max_tasks:
        todo = todo[:a.max_tasks]
    print(f"待处理 {len(todo)}", flush=True)

    results = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(proc_cell, todo, chunksize=4):
            results.append(r)
            with open(CKPT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n = len(results)
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("预算到点，自停（可续跑）", flush=True)
                pool.terminate()
                pool.join()
                return
    summarize()

def summarize():
    rows = []
    with open(CKPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r.get("ok"):
                    rows.append(r)
            except Exception:
                pass
    print(f"\n========== Stage2 汇总（细胞数 {len(rows)}）==========")
    df = pd.DataFrame(rows)
    print(f"QC 通过 {int(df['qc_pass'].sum())}/{len(df)}")
    # 逐块记录
    blk = []
    for r in rows:
        for b in r["blocks"]:
            blk.append({"lab": r["lab"], "drug": r["drug"], "blind": r["blind"], "cell": r["cell"],
                        "qc_pass": r["qc_pass"], **b,
                        "ref_tau_tail": r["ref"]["tau_tail_ms"], "ref_A_peak": r["ref"]["A_peak"],
                        "tau_ratio": b["tau_tail_ss"] / r["ref"]["tau_tail_ms"] if r["ref"]["tau_tail_ms"] and np.isfinite(r["ref"]["tau_tail_ms"]) else np.nan})
    dfb = pd.DataFrame(blk)
    dfb.to_csv(os.path.join(OUT, "stage2_block_metrics.csv"), index=False, encoding="utf-8-sig")
    # 药×实验室汇总：最高浓度块
    q = dfb[dfb["qc_pass"]]
    summ = []
    for (lab, drug), g in q.groupby(["lab", "drug"]):
        top = g.loc[g.groupby("cell")["conc_uM"].idxmax()]
        summ.append({"lab": lab, "drug": drug, "n_cells": int(g["cell"].nunique()),
                     "top_conc_med_uM": float(top["conc_uM"].median()),
                     "block_frac_med": float(top["block_frac"].median()),
                     "tau_ratio_med": float(top["tau_ratio"].median()),
                     "tau_ratio_q25": float(top["tau_ratio"].quantile(0.25)),
                     "tau_ratio_q75": float(top["tau_ratio"].quantile(0.75)),
                     "shape_change_med": float((top["shape_ratio_ss"]).median()),
                     "onset_tau_med_s": float(g["onset_tau_s"].median())})
    dfs = pd.DataFrame(summ)
    dfs.to_csv(os.path.join(OUT, "stage2_drug_kinetics_summary.csv"), index=False, encoding="utf-8-sig")
    tr = dfs["tau_ratio_med"].dropna()
    print(f"最高浓度块 τ_tail 比值（药/对照）：中位 {tr.median():.3f}，IQR [{tr.quantile(0.25):.3f}, {tr.quantile(0.75):.3f}]")
    print(f"偏离 >20% 的目录数：{int(((tr < 0.8) | (tr > 1.25)).sum())}/{len(tr)}")
    print("DONE")

if __name__ == "__main__":
    main()
