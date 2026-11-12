# -*- coding: utf-8 -*-
"""
统一重算 Batch 5a：Nav1.5（Nav1.5_27193878）失活快分量 τ_h(V) 剖面似然重提
方法基座 = Nα1 判决 α 臂（表观单指数，峰后 0.15ms 起拟，25 kHz）
升级：τ_h 加剖面似然 95% CI（固定 τ 时 A/Ip 线性最小二乘）
数据：Nav1.5_27193878_原始压缩包.zip（nav/batch{1,2}/.../NaIV_35C_{CP}CP.csv）
输出：nav5_unified.csv
"""
import os, zipfile, time
import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档\2026-10-16_清理\04_细胞线4\_归档\公开数据\Nav1.5_27193878_原始压缩包.zip"
EXTRACT = os.path.join(HERE, "_nav15_raw")

DT_MS = 0.04
SKIP_MS = 0.15
TAU_BND = (0.05, 3.0)
N1 = 225  # 保持段点数（975 点版）

def ensure_extract():
    if os.path.isdir(EXTRACT):
        return
    os.makedirs(EXTRACT, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(EXTRACT)
    print("解压完成 ->", EXTRACT)

def monoexp_tau_prof(tt, y, alpha=0.05):
    """y = A*exp(-t/tau) + Ip，τ 剖面（固定 τ 时 A,Ip 线性）。"""
    n = len(y); p = 3
    if n < p + 3:
        return None
    def solve(tau):
        X = np.column_stack([np.exp(-tt / tau), np.ones(n)])
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ sol
        return float(r @ r), sol
    best = None
    for tau in np.exp(np.linspace(np.log(TAU_BND[0]), np.log(TAU_BND[1] * 3), 50)):
        sse, sol = solve(tau)
        if best is None or sse < best[0]:
            best = (sse, tau, sol)
    sse0, th, sol = best
    A0, Ip0 = sol
    if A0 <= 0:
        return None
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    lo = TAU_BND[0]; hi = TAU_BND[1] * 3
    for cand in np.exp(np.linspace(np.log(TAU_BND[0]), np.log(th), 30))[::-1]:
        if solve(cand)[0] > thresh:
            lo = cand; break
    for cand in np.exp(np.linspace(np.log(th), np.log(TAU_BND[1] * 3), 30)):
        if solve(cand)[0] > thresh:
            hi = cand; break
    edge = abs(th - TAU_BND[0]) < 1e-3 or abs(th - TAU_BND[1]) < 1e-3
    return {"tau": th, "tau_lo": lo, "tau_hi": hi, "A": A0, "Ip": Ip0,
            "sse": sse0, "n": n, "tau_edge": edge}

def main():
    t0 = time.time()
    ensure_extract()
    nav_root = os.path.join(EXTRACT, "nav")
    rows = []
    n_file = 0
    for root, dirs, files in os.walk(nav_root):
        for f in files:
            if not (f.startswith("NaIV_") and f.endswith(".csv") and "meta" not in f):
                continue
            # 文件名：NaIV_35C_0CP.csv
            try:
                temp = float(f.split("_")[1].replace("C", ""))
                cp = float(f.split("_")[2].replace("CP.csv", ""))
            except Exception:
                continue
            cell = os.path.basename(root)
            fp = os.path.join(root, f)
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if len(df) not in (975, 1000):
                continue
            n1, n2, _ = {975: (225, 500, 250), 1000: (250, 500, 250)}[len(df)]
            volts = [float(c) for c in df.columns]
            for v in volts:
                cur = df[f"{v:.1f}"].to_numpy() * 1e3  # nA→pA
                seg = cur[n1: n1 + n2]
                if len(seg) < 100:
                    continue
                ipk = int(np.argmin(seg))  # Nav 内向电流，峰为负
                A_pk = -float(seg[ipk])
                if A0 := A_pk if False else None:
                    pass
                if not np.isfinite(A_pk) or A_pk < 200.0:
                    continue
                a0 = ipk + int(round(SKIP_MS / DT_MS))
                if a0 >= len(seg) - 30:
                    continue
                y = -seg[a0:]
                tt = np.arange(a0, len(seg)) * DT_MS - ipk * DT_MS
                r = monoexp_tau_prof(tt, y)
                if r is None or r["tau_edge"]:
                    continue
                rows.append({"source_big": "nav15_27193878", "dataset": f"batch_{cell}",
                             "cell_id": cell, "channel": "Nav1.5", "family": "Nav",
                             "species": "human|HEK", "temp_C": temp,
                             "parameter": "tau_inact", "process": "inactivation",
                             "voltage_mV": v, "value": r["tau"],
                             "se_pcov": np.nan, "prof_lo": r["tau_lo"], "prof_hi": r["tau_hi"],
                             "model": "monoexp_profile", "p_model": np.nan,
                             "r2": np.nan, "qc_flag": "ok", "method": "profile_v1",
                             "drug": f"CP{cp:g}" if cp else "", "blind": False,
                             "A_peak_pA": A_pk})
                n_file += 1
    df = pd.DataFrame(rows)
    fp = os.path.join(HERE, "nav5_unified.csv")
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"落盘 {fp}：{len(df)} 行，耗时 {time.time()-t0:.0f}s")
    if len(df):
        print(df.groupby("cell_id")["value"].agg(["count", "median"]).to_string())

if __name__ == "__main__":
    main()
