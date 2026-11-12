# -*- coding: utf-8 -*-
"""
统一重算 Batch 5b：CaV1.2（Ren2022）失活 τ 温度律剖面似然重提
方法基座 = Cα-1（scan_file：ch0 电流/ch1 电压回读/ch2 浴温；+47mV 段；G1-G4 QC）
升级：τ_inact 单指数/双指数（AICc 选型）+ 剖面似然 95% CI
输出：cav5_unified.csv
"""
import os, glob, time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as sst
import pyabf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档\2026-10-16_清理\04_细胞线4\_归档\公开数据\Cav12_Ren2022_g3msb"

DT = 1e-4
T47 = (0.42, 0.60)
G1_DV = 5.0; G2_PK = 100.0; G3_RATIO = 0.9; G4_R2 = 0.95
AICC_MARGIN = 10.0

def exp1(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def exp2(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

def aicc(n, k, rss):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def prof_tau_1exp(tt, yy, tau_hat, alpha=0.05):
    """单指数 τ 剖面（固定 τ，A/C 线性）。"""
    n = len(yy); p = 3
    def solve(tau):
        X = np.column_stack([np.exp(-tt / tau), np.ones(n)])
        sol, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ sol
        return float(r @ r)
    sse0 = solve(tau_hat)
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    grid = np.exp(np.linspace(np.log(1e-3), np.log(0.5), 60))
    lo = grid[0]; hi = grid[-1]
    for cand in grid[grid < tau_hat][::-1]:
        if solve(cand) > thresh:
            lo = cand; break
    for cand in grid[grid > tau_hat]:
        if solve(cand) > thresh:
            hi = cand; break
    return lo, hi

def prof_tauw_2exp(tt, yy, p2, alpha=0.05):
    """双指数加权 τw 的近似剖面：固定 τw 方向扰动。
    简化：对 τ1/τ2 分别做一维剖面，τw CI 由 (τ1,τ2) 极值组合给出。"""
    A1, t1, A2, t2, C = p2
    n = len(yy); p = 5
    def sse_given(t1_, t2_):
        X = np.column_stack([np.exp(-tt / t1_), np.exp(-tt / t2_), np.ones(n)])
        sol, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ sol
        return float(r @ r)
    sse0 = sse_given(t1, t2)
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    def w_of(t1_, t2_):
        X = np.column_stack([np.exp(-tt / t1_), np.exp(-tt / t2_), np.ones(n)])
        sol, *_ = np.linalg.lstsq(X, yy, rcond=None)
        a1_, a2_ = abs(sol[0]), abs(sol[1])
        return (a1_ * t1_ + a2_ * t2_) / max(a1_ + a2_, 1e-12)
    ws = []
    g1 = np.exp(np.linspace(np.log(1e-3), np.log(0.5), 24))
    g2 = np.exp(np.linspace(np.log(1e-3), np.log(0.5), 24))
    for t1_ in g1:
        for t2_ in g2:
            if t2_ > t1_ and sse_given(t1_, t2_) <= thresh:
                ws.append(w_of(t1_, t2_))
    if ws:
        return min(ws), max(ws)
    return np.nan, np.nan

def fit_segment_prof(t, y):
    i_pk = int(np.argmin(y))
    if i_pk >= len(y) - 30:
        return None
    tt = t[i_pk:] - t[i_pk]
    yy = y[i_pk:]
    pk = y[i_pk]
    tail = np.mean(yy[-200:])
    if not (pk <= -G2_PK and abs(tail) < G3_RATIO * abs(pk)):
        return {"qc_fail": "G2G3"}
    C0 = tail
    try:
        p1, _ = curve_fit(exp1, tt, yy, p0=[pk - C0, 0.03, C0],
                          bounds=([-2e4, 1e-3, -5e3], [0, 0.5, 5e3]), maxfev=20000)
        rss1 = float(np.sum((yy - exp1(tt, *p1)) ** 2))
    except Exception:
        return {"qc_fail": "fit1"}
    r2_1 = 1 - rss1 / float(np.sum((yy - yy.mean()) ** 2))
    best = {"model": "1", "tau": float(p1[1] * 1e3), "r2": float(r2_1)}
    lo, hi = prof_tau_1exp(tt, yy, p1[1])
    best["tau_lo"] = lo * 1e3; best["tau_hi"] = hi * 1e3
    try:
        p2, _ = curve_fit(exp2, tt, yy,
                          p0=[(pk - C0) * 0.6, 0.01, (pk - C0) * 0.4, 0.08, C0],
                          bounds=([-2e4, 1e-3, -2e4, 1e-3, -5e3], [0, 0.5, 0, 0.5, 5e3]),
                          maxfev=40000)
        rss2 = float(np.sum((yy - exp2(tt, *p2)) ** 2))
        r2_2 = 1 - rss2 / float(np.sum((yy - yy.mean()) ** 2))
        a2 = aicc(len(yy), 5, rss2)
        if a2 < aicc(len(yy), 3, rss1) - AICC_MARGIN:
            A1, t1, A2, t2, C = p2
            tw = (abs(A1) * t1 + abs(A2) * t2) / (abs(A1) + abs(A2))
            lo2, hi2 = prof_tauw_2exp(tt, yy, p2)
            best = {"model": "2", "tau": float(tw * 1e3), "r2": float(r2_2),
                    "tau_lo": lo2 * 1e3, "tau_hi": hi2 * 1e3}
    except Exception:
        pass
    best["qc_fail"] = None if best["r2"] >= G4_R2 else "G4"
    return best

def main():
    t0 = time.time()
    rows = []
    for grp in ["Ca2", "Ba2"]:
        files = sorted(glob.glob(os.path.join(DATA, "Temperature_" + grp, "*.abf")))
        for f in files:
            name = os.path.basename(f).replace(".abf", "")
            a = pyabf.ABF(f)
            for s in range(a.sweepCount):
                a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
                a.setSweep(s, channel=1); v = a.sweepY.astype(float)
                a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
                T = float(bt.mean())
                m47 = (t >= T47[0]) & (t < T47[1])
                v47 = float(v[m47].mean())
                if abs(v47 - 47.0) > G1_DV:
                    continue
                fr = fit_segment_prof(t[m47], i[m47])
                if fr is None or fr.get("qc_fail"):
                    continue
                rows.append({"source_big": "cav12_ren2022",
                             "dataset": f"Temperature_{grp}",
                             "cell_id": name, "channel": "CaV1.2", "family": "CaV",
                             "species": "", "temp_C": T,
                             "parameter": "tau_inact", "process": "inactivation",
                             "voltage_mV": 47.0, "value": fr["tau"],
                             "se_pcov": np.nan, "prof_lo": fr["tau_lo"], "prof_hi": fr["tau_hi"],
                             "model": f"exp{fr['model']}_profile", "p_model": np.nan,
                             "r2": fr["r2"], "qc_flag": "ok", "method": "profile_v1",
                             "drug": "", "blind": False, "A_peak_pA": np.nan})
    df = pd.DataFrame(rows)
    fp = os.path.join(HERE, "cav5_unified.csv")
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"落盘 {fp}：{len(df)} 行，耗时 {time.time()-t0:.0f}s")
    if len(df):
        print(df.groupby("dataset")["value"].agg(["count", "median"]).to_string())

if __name__ == "__main__":
    main()
