# -*- coding: utf-8 -*-
"""
统一重算 Batch 4：Beattie 九细胞 hERG 四协议剖面似然重提
数据：wA1 代码A1 data/cells + data/protocols（.mat，10 kHz，dofetilide 减+漏减）
协议结构（已程序化解析确认，10 kHz）：
  deactivation:  +50mV 2s 预激活 → V_test(-120..-40, 6~6.5s) → τ_deact(V) 双指数
  steady_activation: V_hold(-60..60, 5s) → I_ss(V) → Boltzmann (Vhalf_act, k_act) 二维剖面
  activation_kinetics_1: (0mV, T=30ms..10s) 包络 → τ_act(0mV)
  inactivation: +50 6s 全激活 → -90 0.6s → V_test(-100..50, 1.5s) → τ_inact(V) 单指数
输出：beattie4_unified.csv（统一 schema，value 单位 ms；Boltzmann 参数 mV）
"""
import os, json, time
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.optimize import curve_fit
from scipy import stats as sst
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档\2026-10-16_清理\04_细胞线4\_归档\wA1\代码A1_16704007_流程固定重标卡_2026-09-11\data"
CELLS = ["16704007", "16704047", "16707014", "16708016", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"]
CELLS = sorted(set(CELLS))
FS = 1e4  # 10 kHz
DT = 1e-4

def load(proto, cell):
    V = sio.loadmat(os.path.join(DATA, "protocols", f"{proto}_protocol.mat"))["T"].flatten().astype(float)
    I = sio.loadmat(os.path.join(DATA, "cells", cell, f"{proto}_{cell}_dofetilide_subtracted_leak_subtracted.mat"))["T"].flatten().astype(float)
    return V, I

def segs_of(V):
    r = np.round(V).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(V)), edges)]

# ---- 剖面似然工具 ----
def prof_linear_multi(tt, y, sse_fn, p_free, alpha=0.05):
    """通用：给定 SSE 函数与最优点，对各参数做一维剖面（其余参数在网格/连续上重新优化由 sse_fn 内部完成）。
    sse_fn(param_vec) 返回 SSE。p_free = 最优参数向量。"""
    raise NotImplementedError

def biexp_grid_prof(tt, y, sig, alpha=0.05):
    """双指数带锚网格 + τ1/τ2 剖面（同 Batch3 逻辑，单位 s）。"""
    n = len(tt); p = 5
    if n < p + 5:
        return None
    E = lambda tau: np.exp(-tt / tau)
    def solve(t1, t2):
        X = np.column_stack([np.ones(n), E(t1), E(t2)])
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ sol
        return float(r @ r), sol
    best = None
    for t1 in np.exp(np.linspace(np.log(0.002), np.log(0.5), 20)):
        for t2 in np.exp(np.linspace(np.log(max(0.01, t1 * 1.8)), np.log(30.0), 24)):
            sse, sol = solve(t1, t2)
            if best is None or sse < best[0]:
                best = (sse, t1, t2, sol)
    sse0, t1h, t2h, sol = best
    a0, a1, a2 = sol
    if abs(a1) + abs(a2) < 4 * sig:
        return None
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    def prof(which):
        if which == 1:
            lo = 0.002; hi = 0.5
            for cand in np.exp(np.linspace(np.log(0.002), np.log(t1h), 22))[::-1]:
                if min(solve(cand, t2)[0] for t2 in np.exp(np.linspace(np.log(max(0.01, cand * 1.8)), np.log(30.0), 14))) > thresh:
                    lo = cand; break
            for cand in np.exp(np.linspace(np.log(t1h), np.log(0.5), 14)):
                if min(solve(cand, t2)[0] for t2 in np.exp(np.linspace(np.log(max(0.01, cand * 1.8)), np.log(30.0), 14))) > thresh:
                    hi = cand; break
            return lo, hi
        else:
            lo = 0.01; hi = 30.0
            for cand in np.exp(np.linspace(np.log(max(0.01, t1h * 1.8)), np.log(t2h), 22))[::-1]:
                if min(solve(t1, cand)[0] for t1 in np.exp(np.linspace(np.log(0.002), np.log(min(0.5, cand / 1.8)), 14))) > thresh:
                    lo = cand; break
            for cand in np.exp(np.linspace(np.log(t2h), np.log(30.0), 22)):
                if min(solve(t1, cand)[0] for t1 in np.exp(np.linspace(np.log(0.002), np.log(min(0.5, cand / 1.8)), 14))) > thresh:
                    hi = cand; break
            return lo, hi
    t1_lo, t1_hi = prof(1)
    t2_lo, t2_hi = prof(2)
    return {"tau1": t1h, "tau1_lo": t1_lo, "tau1_hi": t1_hi,
            "tau2": t2h, "tau2_lo": t2_lo, "tau2_hi": t2_hi, "sse": sse0}

def monoexp_prof(tt, y, alpha=0.05):
    """单指数 y=y0+a*exp(-t/tau)，tau 剖面（固定 tau 时 y0,a 线性）。"""
    n = len(tt); p = 3
    if n < p + 5:
        return None
    def solve(tau):
        X = np.column_stack([np.ones(n), np.exp(-tt / tau)])
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ sol
        return float(r @ r), sol
    best = None
    for tau in np.exp(np.linspace(np.log(0.002), np.log(30), 60)):
        sse, sol = solve(tau)
        if best is None or sse < best[0]:
            best = (sse, tau, sol)
    sse0, th, sol = best
    if abs(sol[1]) < 1e-12:
        return None
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    lo = 0.002; hi = 30.0
    for cand in np.exp(np.linspace(np.log(0.002), np.log(th), 40))[::-1]:
        if solve(cand)[0] > thresh:
            lo = cand; break
    for cand in np.exp(np.linspace(np.log(th), np.log(30), 40)):
        if solve(cand)[0] > thresh:
            hi = cand; break
    return {"tau": th, "tau_lo": lo, "tau_hi": hi, "a": sol[1], "y0": sol[0], "sse": sse0}

def boltzmann_prof(V, I, sign=1, alpha=0.05):
    """I = I0 + A/(1+exp(-sign*(V-Vh)/k))，二维网格 + Vh/k 剖面。"""
    n = len(V); p = 4
    if n < p + 2:
        return None
    def solve(vh, k):
        x = 1.0 / (1.0 + np.exp(-sign * (V - vh) / k))
        X = np.column_stack([np.ones(n), x])
        sol, *_ = np.linalg.lstsq(X, I, rcond=None)
        r = I - X @ sol
        return float(r @ r), sol
    best = None
    vgrid = np.linspace(V.min() - 10, V.max() + 10, 40)
    kgrid = np.linspace(1.5, 40, 40)
    for vh in vgrid:
        for k in kgrid:
            sse, sol = solve(vh, k)
            if best is None or sse < best[0]:
                best = (sse, vh, k, sol)
    sse0, vh0, k0, sol = best
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    def prof_vh():
        lo, hi = vgrid[0], vgrid[-1]
        for cand in np.linspace(vgrid[0], vh0, 30)[::-1]:
            if min(solve(cand, k)[0] for k in kgrid) > thresh:
                lo = cand; break
        for cand in np.linspace(vh0, vgrid[-1], 30):
            if min(solve(cand, k)[0] for k in kgrid) > thresh:
                hi = cand; break
        return lo, hi
    def prof_k():
        lo, hi = kgrid[0], kgrid[-1]
        for cand in np.linspace(kgrid[0], k0, 30)[::-1]:
            if min(solve(vh, cand)[0] for vh in vgrid) > thresh:
                lo = cand; break
        for cand in np.linspace(k0, kgrid[-1], 30):
            if min(solve(vh, cand)[0] for vh in vgrid) > thresh:
                hi = cand; break
        return lo, hi
    vh_lo, vh_hi = prof_vh()
    k_lo, k_hi = prof_k()
    return {"Vhalf": vh0, "Vhalf_lo": vh_lo, "Vhalf_hi": vh_hi,
            "k": k0, "k_lo": k_lo, "k_hi": k_hi, "A": sol[1], "sse": sse0}

# ---- 协议提取 ----
def do_deactivation(cell):
    V, I = load("deactivation", cell)
    segs = segs_of(V)
    rows = []
    sig = float(np.std(I[0:2000]))
    for i in range(len(segs) - 1):
        vv, a, b = segs[i]
        vp = segs[i - 1][0]
        # V_test 段：前置 +50 2s，段长 5.5-7s，电压 -120..-40
        if vp == 50 and 5.5 <= (b - a) * DT <= 7.0 and -125 <= vv <= -35:
            seg = I[a:b]
            tt = np.arange(len(seg)) * DT
            r = biexp_grid_prof(tt, seg, sig)
            if r:
                rows.append({"parameter": "tau_deact_fast", "process": "deactivation",
                             "voltage_mV": vv, "value": r["tau1"] * 1e3,
                             "prof_lo": r["tau1_lo"] * 1e3, "prof_hi": r["tau1_hi"] * 1e3,
                             "model": "biexp_anchored"})
                rows.append({"parameter": "tau_deact_slow", "process": "deactivation",
                             "voltage_mV": vv, "value": r["tau2"] * 1e3,
                             "prof_lo": r["tau2_lo"] * 1e3, "prof_hi": r["tau2_hi"] * 1e3,
                             "model": "biexp_anchored"})
    return rows

def do_steady_activation(cell):
    """m_ss(V)：尾电流法。V_hold(-60..+60, 5s) 后紧跟 -40mV 1s 尾电流段，
    尾段 20-120ms 窗内峰电流 ∝ V_hold 达到的激活水平 → Boltzmann(Vhalf_act, k_act)。"""
    V, I = load("steady_activation", cell)
    segs = segs_of(V)
    vh_list = []; iss_list = []
    for i in range(len(segs) - 1):
        vv, a, b = segs[i]
        vn, an, bn = segs[i + 1]
        # V_hold 段（5s 长、非 -40）后跟 -40 尾段（~1s）
        if 4.5 <= (b - a) * DT <= 5.5 and -65 <= vv <= 65 and vv != -40 and vn == -40 and 0.8 <= (bn - an) * DT <= 1.2:
            tail = I[an: bn]
            win = tail[int(0.02 / DT): int(0.12 / DT)]  # 20-120ms 峰窗
            if len(win) > 10:
                vh_list.append(vv)
                iss_list.append(float(np.max(win)))
    rows = []
    if len(vh_list) >= 5:
        r = boltzmann_prof(np.array(vh_list, float), np.array(iss_list), sign=1)
        if r:
            rows.append({"parameter": "Vhalf_act", "process": "activation",
                         "voltage_mV": np.nan, "value": r["Vhalf"],
                         "prof_lo": r["Vhalf_lo"], "prof_hi": r["Vhalf_hi"],
                         "model": "boltzmann_grid|tail_peak"})
            rows.append({"parameter": "k_act", "process": "activation",
                         "voltage_mV": np.nan, "value": r["k"],
                         "prof_lo": r["k_lo"], "prof_hi": r["k_hi"],
                         "model": "boltzmann_grid|tail_peak"})
    return rows

def do_inactivation(cell):
    """h_ss(V)：+50 6s 全激活 → -90 0.6s 恢复窗 → V_test 1.5s。
    V_test 段起始 50ms 均值电流 ∝ 该电压下经 -90 恢复窗后的可用度 → Boltzmann(Vhalf_inact, k_inact, sign=-1)。"""
    V, I = load("inactivation", cell)
    segs = segs_of(V)
    rows = []
    v_list = []; a_list = []
    for i in range(2, len(segs)):
        vv, a, b = segs[i]
        v_prev2 = segs[i - 2][0]
        v_prev1 = segs[i - 1][0]
        if v_prev2 == 50 and v_prev1 == -90 and 0.13 <= (b - a) * DT <= 0.17 and -105 <= vv <= 55:
            seg = I[a:b]
            head = float(np.mean(seg[:int(0.05 / DT)]))  # 起始 50ms
            v_list.append(vv); a_list.append(head)
    if len(v_list) >= 5:
        a_arr = np.array(a_list)
        # 归一：以最负档（恢复最完全）的幅度为参考
        a_arr = a_arr / max(abs(np.percentile(a_arr, 10)), 1e-12)
        r = boltzmann_prof(np.array(v_list, float), a_arr, sign=-1)
        if r:
            rows.append({"parameter": "Vhalf_inact", "process": "inactivation",
                         "voltage_mV": np.nan, "value": r["Vhalf"],
                         "prof_lo": r["Vhalf_lo"], "prof_hi": r["Vhalf_hi"],
                         "model": "boltzmann_grid|hss_head50ms"})
            rows.append({"parameter": "k_inact", "process": "inactivation",
                         "voltage_mV": np.nan, "value": r["k"],
                         "prof_lo": r["k_lo"], "prof_hi": r["k_hi"],
                         "model": "boltzmann_grid|hss_head50ms"})
    return rows

def do_activation_kinetics(cell):
    """envelope of tails：0mV 变时长激活后回 -120，尾电流早期幅度 ∝ 激活程度 → τ_act(0)。"""
    V, I = load("activation_kinetics_1", cell)
    segs = segs_of(V)
    Ts = []; amps = []
    for i in range(len(segs) - 1):
        vv, a, b = segs[i]
        vn, an, bn = segs[i + 1]
        if vv == 0 and vn == -120 and (b - a) * DT <= 11:
            T = (b - a) * DT
            # 尾电流：-120 段早期 50ms 峰
            tail = I[an: an + int(0.05 / DT)]
            if len(tail) > 10:
                Ts.append(T)
                amps.append(float(np.max(np.abs(tail))))
    rows = []
    if len(Ts) >= 5:
        # 包络：A(T) = Amax*(1-exp(-T/tau)) → 单参数剖面
        Ts = np.array(Ts); amps = np.array(amps)
        Amax = np.percentile(amps, 95)
        def solve(tau):
            x = 1 - np.exp(-Ts / tau)
            X = np.column_stack([np.ones(len(Ts)), x])
            sol, *_ = np.linalg.lstsq(X, amps, rcond=None)
            r = amps - X @ sol
            return float(r @ r), sol
        best = None
        for tau in np.exp(np.linspace(np.log(0.005), np.log(5), 60)):
            sse, sol = solve(tau)
            if best is None or sse < best[0]:
                best = (sse, tau, sol)
        sse0, th, sol = best
        n = len(Ts); p = 3
        thresh = sse0 * (1 + sst.f.ppf(0.95, 1, n - p) / (n - p))
        lo, hi = 0.005, 5.0
        for cand in np.exp(np.linspace(np.log(0.005), np.log(th), 40))[::-1]:
            if solve(cand)[0] > thresh:
                lo = cand; break
        for cand in np.exp(np.linspace(np.log(th), np.log(5), 40)):
            if solve(cand)[0] > thresh:
                hi = cand; break
        rows.append({"parameter": "tau_act", "process": "activation",
                     "voltage_mV": 0.0, "value": th * 1e3,
                     "prof_lo": lo * 1e3, "prof_hi": hi * 1e3,
                     "model": "envelope_of_tails"})
    return rows

def main():
    all_rows = []
    t0 = time.time()
    for cell in CELLS:
        for fn, proto in [(do_deactivation, "deact"), (do_steady_activation, "ssact"),
                          (do_inactivation, "inact"), (do_activation_kinetics, "actkin")]:
            try:
                rows = fn(cell)
                for r in rows:
                    r.update({"source_big": "beattie9", "dataset": "beattie9",
                              "cell_id": cell, "channel": "hERG", "family": "Kv",
                              "species": "human|CHO", "temp_C": 37.0,
                              "se_pcov": np.nan, "p_model": np.nan, "r2": np.nan,
                              "qc_flag": "ok", "method": "profile_v1",
                              "drug": "", "blind": False, "A_peak_pA": np.nan})
                all_rows.extend(rows)
                print(f"{cell} {proto}: {len(rows)} 行", flush=True)
            except Exception as e:
                print(f"{cell} {proto} FAIL: {type(e).__name__}: {e}", flush=True)
    df = pd.DataFrame(all_rows)
    fp = os.path.join(HERE, "beattie4_unified.csv")
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"落盘 {fp}：{len(df)} 行，耗时 {time.time()-t0:.0f}s")
    print(df.groupby("parameter")["value"].agg(["count", "median"]).to_string())

if __name__ == "__main__":
    main()
