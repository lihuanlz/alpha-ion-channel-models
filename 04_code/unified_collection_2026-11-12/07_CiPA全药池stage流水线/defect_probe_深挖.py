# -*- coding: utf-8 -*-
"""数据缺陷深挖：温度列、电压实现值、E-4031残余、采样率、文件完整性
每实验室分层抽样 zip，读原始列统计。"""
import os, glob, json, zipfile, argparse, time
import numpy as np
import pandas as pd
from multiprocessing import Pool

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
CIPA = os.path.join(LAW, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
OUT = os.path.join(LAW, "分析_全药池_2026-11-10")

def probe(zpath):
    rel = os.path.relpath(zpath, CIPA); parts = rel.split(os.sep)
    lab, drug = parts[0], parts[2]
    cell = f"{lab}/{drug}/" + os.path.basename(zpath).replace(".csv.zip", "")
    try:
        z = zipfile.ZipFile(zpath)
        bad = z.testzip()  # None=完整
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        t = df["t_ms"].to_numpy(float)
        dt = np.diff(t)
        dt_med = float(np.median(dt)); dt_jit = float(np.max(np.abs(dt - dt_med)) / max(dt_med, 1e-12))
        # 电压列
        vcols = [c for c in df.columns if "voltage" in c.lower()]
        vmin = vmax = np.nan
        if vcols:
            V = df[vcols[0]].to_numpy(float)
            vmin, vmax = float(np.min(V)), float(np.max(V))
        # 温度列
        tcols = [c for c in df.columns if "temperature" in c.lower()]
        temp = float(np.median(df[tcols[0]].to_numpy(float))) if tcols else np.nan
        # 电流列
        ccols = [c for c in df.columns if c.startswith("trace_#") and "_current_" in c]
        uf = {"pa":1.0,"na":1e3,"a":1e12,"ma":1e15,"ua":1e6,"µa":1e6}.get(
            ccols[0].rsplit("_current_",1)[-1].lower()) if ccols else None
        # E-4031 残余：末尾 10% 电流列的峰幅 / 前 20% 的峰幅
        e4_ratio = np.nan; head_amp = np.nan; tail_amp = np.nan
        if ccols and uf:
            amps = []
            for c in ccols:
                y = df[c].to_numpy(float) * uf
                if np.std(y) > 5:
                    amps.append(float(np.max(y)))
            amps = np.array(amps)
            if len(amps) >= 20:
                head_amp = float(np.median(amps[:max(3, len(amps)//5)]))
                tail_amp = float(np.median(amps[-max(3, len(amps)//10):]))
                if head_amp > 50:
                    e4_ratio = tail_amp / head_amp
        return {"cell": cell, "lab": lab, "drug": drug, "zip_ok": bad is None,
                "n_cols": len(df.columns), "n_cur": len(ccols), "has_V": len(vcols) > 0, "has_T": len(tcols) > 0,
                "sweep_ms": float(t.max()-t.min()), "dt_med_ms": dt_med, "dt_jitter": dt_jit,
                "V_min": vmin, "V_max": vmax, "temp_C": temp,
                "head_amp_pA": head_amp, "tail_amp_pA": tail_amp, "e4_residual": e4_ratio}
    except Exception as e:
        return {"cell": cell, "lab": lab, "drug": drug, "err": f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lab", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.RandomState(20261111)
    zips = []
    for lab in ["lab_1","lab_2","lab_3","lab_4","lab_5"]:
        zs = sorted(glob.glob(os.path.join(CIPA, lab, "*", "*", "subtracted", "ted", "*.csv.zip")))
        idx = rng.choice(len(zs), size=min(a.per_lab, len(zs)), replace=False)
        zips += [zs[i] for i in idx]
    print(f"抽样 {len(zips)} 个文件", flush=True)
    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(probe, zips, chunksize=4):
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "defect_probe.csv"), index=False, encoding="utf-8-sig")
    print(f"完成 {len(df)}，耗时 {time.time()-t0:.0f}s")
    # 汇总
    ok = df[df.get("zip_ok", pd.Series(dtype=bool)) == True] if "zip_ok" in df else df
    print("\n=== 按实验室汇总 ===")
    for lab, g in df.groupby("lab"):
        if "err" in g and g.get("err", pd.Series([np.nan]*len(g))).notna().any():
            pass
        print(f"{lab}: n={len(g)} | 温度 中位 {g['temp_C'].median():.1f}°C (有温度列 {g['has_T'].mean()*100:.0f}%) | "
              f"V范围 [{g['V_min'].median():.0f},{g['V_max'].median():.0f}] mV | "
              f"E4031残余 中位 {g['e4_residual'].median()*100:.1f}% (n={g['e4_residual'].notna().sum()}) | "
              f"dt {g['dt_med_ms'].median()*1000:.0f}µs 抖动 {g['dt_jitter'].median():.2e} | "
              f"zip坏 {100-g['zip_ok'].mean()*100:.0f}%")
    print("DONE")

if __name__ == "__main__":
    main()
