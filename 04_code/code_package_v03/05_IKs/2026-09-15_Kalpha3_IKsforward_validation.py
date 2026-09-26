# -*- coding: utf-8 -*-
"""
Kα-3 · IKs α-model forward validation (preregistration v1.0 · 2026-09-15)
Pool: the 8 cells passing Kα-1 crit-0; the only free quantity per cell is G; two-arm tau_deact (A=0.40 / B=2.0 s).
Smoke: %runfile '.../2026-09-15_Kα3_IKs前向验证.py' --wdir   (first line SMOKE=True)
Full: set SMOKE to False, same command
"""
import json, os
import numpy as np
import pyabf
from scipy.optimize import curve_fit

SMOKE = False  # full run (smoke already green: R2act 0.971, inst diff 0.078, arm-B tail 0.669/0.020, arm-A rejected)

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
D = os.path.join(ROOT, "数据", "iks_chan_8226585")
POOL_CSV = os.path.join(ROOT, "α模型", "2026-09-15_Kα1_IKs激活去激活恒定性判决_逐文件表.csv")
FEDIDA_TAU = os.path.join(ROOT, "α模型", "2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv")
TAG = "冒烟" if SMOKE else "结果"
OUT_JSON = os.path.join(ROOT, "α模型", f"2026-09-15_Kα3_IKs前向验证_{TAG}.json")
OUT_CSV = os.path.join(ROOT, "α模型", f"2026-09-15_Kα3_IKs前向验证_逐细胞表_{TAG}.csv")
OUT_PNG = os.path.join(ROOT, "α模型", f"2026-09-15_Kα3_IKs前向验证_{TAG}.png")

# ---------------- model-card entries (preregistration section 1 pinned) ----------------
VH, KK = 25.05, 22.6          # G-V Boltzmann (pool 8-cell median)
E_K = -85.0                   # mV, registered assumption
TAU_D_ARMS = {"A": 0.40, "B": 2.0}   # tau_deact(-40) two arms
BETA = 0.30                          # v1.1: instant-component pool constant (inst_med median 0.302)

def load_fedida_tau():
    import csv as _csv
    vs, ts = [], []
    for r in _csv.DictReader(open(FEDIDA_TAU, encoding="utf-8-sig")):
        vs.append(float(r["V_mV"])); ts.append(float(r["tau_s"]))
    return np.array(vs), np.array(ts)

FV, FT = load_fedida_tau()
def tau_act(V):
    return float(np.interp(V, FV, FT))

DV = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
DT = np.array([1.17, 1.01, 0.91, 0.79, 0.73, 0.54, 0.53, 0.42, 0.31])
def delay(V):
    return float(np.interp(V, DV, DT))

def a_ss(V):
    return 1.0 / (1.0 + np.exp(-(V - VH) / KK))

# ---------------- protocol parsing (same gates as Ka-1) ----------------
def load_cell(path):
    """Return a list of sweeps: dict(V, t_act, y_act, t_tail, y_tail), baseline subtracted."""
    a = pyabf.ABF(path)
    dt = 1.0 / a.sampleRate
    out = []
    for s in range(a.sweepCount):
        a.setSweep(s)
        e = a.sweepEpochs
        if len(e.types) < 4:
            continue
        v_step = float(e.levels[2]); v_tail = float(e.levels[3])
        if abs(v_tail + 40) > 5:
            continue
        y = a.sweepY.astype(float)
        p2a, p2b = int(e.p1s[2]), int(e.p2s[2])
        p3a, p3b = int(e.p1s[3]), int(e.p2s[3])
        base = float(np.mean(y[max(0, p2a - int(0.02 / dt)):p2a]))
        yc = y - base
        out.append({"V": v_step,
                    "t_act": np.arange(p2b - p2a) * dt, "y_act": yc[p2a:p2b],
                    "t_tail": np.arange(p3b - p3a) * dt, "y_tail": yc[p3a:p3b],
                    "V_pre": float(e.levels[1]) if len(e.levels) > 1 else -80.0})
    return out

# ---------------- forward simulation (unit G) ----------------
def sim_sweep(sw, tau_d):
    """v1.1: instant + delayed two components. Returns a_act, a_tail (gating-variable trajectories, G=1)."""
    V = sw["V"]
    ass = a_ss(V); ass_pre = a_ss(sw["V_pre"])
    ass_d = (1.0 - BETA) * ass           # delayed-component steady state
    a0d = (1.0 - BETA) * ass_pre         # delayed-component initial value
    d = delay(V); tau = tau_act(V)
    t = sw["t_act"]
    a_dyn = np.where(t <= d, a0d, ass_d + (a0d - ass_d) * np.exp(-(t - d) / tau))
    a_act = BETA * ass + a_dyn
    a_end_dyn = a_dyn[-1] if len(a_dyn) else a0d
    a_tail = BETA * a_ss(-40.0) + a_end_dyn * np.exp(-sw["t_tail"] / tau_d)
    return a_act, a_tail

def unit_current(sw, tau_d):
    a_act, a_tail = sim_sweep(sw, tau_d)
    i_act = a_act * (sw["V"] - E_K)
    i_tail = a_tail * (-40.0 - E_K)
    return i_act, i_tail

# ---------------- metrics ----------------
def r2_of(y, yhat):
    ss = np.sum((y - yhat) ** 2); st = np.sum((y - y.mean()) ** 2)
    return 1 - ss / st if st > 0 else np.nan

def decay_frac(y, dt):
    n0 = slice(int(0.005 / dt), int(0.025 / dt))
    n1 = slice(len(y) - int(0.05 / dt), len(y))
    start = float(np.mean(y[n0])); end = float(np.mean(y[n1]))
    return (start - end) / start if start > 0 else np.nan

def inst_of(y, dt):
    """v1.1 crit-C metric (same as Ka-1 v1.4): first 50 ms (from 2 ms) / last 100 ms."""
    head = float(np.mean(y[int(0.002 / dt):int(0.002 / dt) + int(0.05 / dt)]))
    iss = float(np.mean(y[-int(0.1 / dt):]))
    return head / iss if iss > 0 else np.nan

# ---------------- per cell ----------------
def run_cell(path, tau_d, fit_taud=False):
    sweeps = load_cell(path)
    if len(sweeps) < 8:
        return None
    dt_cell = sweeps[0]["t_act"][1] - sweeps[0]["t_act"][0]
    if fit_taud:
        # diagnostic: tau_deact free grid fit (minimize tail RMS, G by simultaneous least squares)
        best = (np.inf, None, None)
        for td in np.arange(0.05, 8.001, 0.05):  # v1.1: grid extended to 8 s
            u, m = [], []
            for sw in sweeps:
                ia, it = unit_current(sw, td)
                u += [ia, it]; m += [sw["y_act"], sw["y_tail"]]
            u = np.concatenate(u); m = np.concatenate(m)
            G = float(np.dot(u, m) / np.dot(u, u))
            r = float(np.sqrt(np.mean((m - G * u) ** 2)))
            if r < best[0]:
                best = (r, td, G)
        return {"tau_d_free": best[1], "G_free": best[2]}
    # main flow: G least squares (all sweeps, activation + tail)
    u, m = [], []
    for sw in sweeps:
        ia, it = unit_current(sw, tau_d)
        u += [ia, it]; m += [sw["y_act"], sw["y_tail"]]
    u = np.concatenate(u); m = np.concatenate(m)
    G = float(np.dot(u, m) / np.dot(u, u))
    # per-segment R2
    ya, yha, yt, yht = [], [], [], []
    df_sim, df_mea, inst_d = [], [], []
    for sw in sweeps:
        ia, it = unit_current(sw, tau_d)
        ya.append(sw["y_act"]); yha.append(G * ia)
        yt.append(sw["y_tail"]); yht.append(G * it)
        df_mea.append(decay_frac(sw["y_tail"], dt_cell))
        df_sim.append(decay_frac(G * it, dt_cell))
        if sw["V"] >= 40:   # v1.1 crit-C metric: only V>=+40 (no signal at low voltages)
            im = inst_of(sw["y_act"], dt_cell); ism = inst_of(G * ia, dt_cell)
            if not (np.isnan(im) or np.isnan(ism)):
                inst_d.append(abs(ism - im))
    return {"G": G, "n_sweeps": len(sweeps),
            "r2_act": float(r2_of(np.concatenate(ya), np.concatenate(yha))),
            "r2_tail": float(r2_of(np.concatenate(yt), np.concatenate(yht))),
            "df_diff_med": float(np.nanmedian(np.abs(np.array(df_sim) - np.array(df_mea)))),
            "inst_diff_med": float(np.median(inst_d)) if inst_d else np.nan,
            "df_mea_med": float(np.nanmedian(df_mea)), "df_sim_med": float(np.nanmedian(df_sim))}

# ---------------- main flow ----------------
def main():
    print("=" * 72)
    print(" Ka-3: IKs alpha-model forward validation (preregistration v1.0)")
    print("=" * 72, flush=True)
    import csv
    pool = [r for r in csv.DictReader(open(POOL_CSV, encoding="utf-8-sig")) if r["in_pool"] == "1"]
    if SMOKE:
        pool = [r for r in pool if r["file"].startswith("2020_10_09_0039")]
        print(f"[smoke] single cell {pool[0]['file']}", flush=True)
    res = {"smoke": SMOKE, "arms": {}, "diag_taud_free": {}, "sensitivity_去边界": {}}
    rows_csv = []
    for arm, td in TAU_D_ARMS.items():
        print(f"\n[arm {arm}] tau_deact(-40) = {td} s", flush=True)
        recs = []
        for r in pool:
            path = os.path.join(D, r["file"])
            if not os.path.exists(path):
                print(f"  missing file {r['file']}", flush=True); continue
            rec = run_cell(path, td)
            if rec is None:
                print(f"  {r['file'][:28]} insufficient sweeps", flush=True); continue
            rec["file"] = r["file"]; recs.append(rec)
            print(f"  {r['file'][:28]} R²act={rec['r2_act']:.3f} R²tail={rec['r2_tail']:.3f} "
                  f"df_diff={rec['df_diff_med']:.3f} inst_diff={rec['inst_diff_med']:.3f} G={rec['G']:.1f}", flush=True)
            rows_csv.append({"arm": arm, "file": r["file"], **{k: rec[k] for k in
                             ("G", "n_sweeps", "r2_act", "r2_tail", "df_diff_med", "inst_diff_med",
                              "df_mea_med", "df_sim_med")}})
        okA = sum(1 for x in recs if x["r2_act"] >= 0.80)
        okB = sum(1 for x in recs if x["r2_tail"] >= 0.50 and x["df_diff_med"] <= 0.10)
        okC = [x["inst_diff_med"] for x in recs if not np.isnan(x["inst_diff_med"])]
        res["arms"][arm] = {
            "tau_d": td, "n": len(recs),
            "判A_激活R2≥0.80细胞数": okA, "判A_过": okA >= 6,
            "判B_尾巴双标细胞数": okB, "判B_过": okB >= 6,
            "判C_inst差中位": float(np.median(okC)) if okC else None,
            "判C_过": (float(np.median(okC)) <= 0.10) if okC else None,
            "cells": recs}
    # diagnostic: tau_deact free fit
    print("\n[diagnostic] tau_deact(-40) per-cell free fit", flush=True)
    for r in pool:
        path = os.path.join(D, r["file"])
        if not os.path.exists(path):
            continue
        dg = run_cell(path, None, fit_taud=True)
        if dg:
            res["diag_taud_free"][r["file"]] = dg
            print(f"  {r['file'][:28]} τ_d_free={dg['tau_d_free']:.2f}s", flush=True)
    tds = [v["tau_d_free"] for v in res["diag_taud_free"].values()]
    res["diag_taud_med"] = float(np.median(tds)) if tds else None
    # sensitivity: recompute without boundary cells (arm-A verdicts only)
    bdr = "2022_03_11_0005 after LS.abf"
    for arm in res["arms"]:
        cells = [c for c in res["arms"][arm]["cells"] if c["file"] != bdr]
        if len(cells) < len(res["arms"][arm]["cells"]):
            res["sensitivity_去边界"][arm] = {
                "n": len(cells),
                "判A": sum(1 for x in cells if x["r2_act"] >= 0.80),
                "判B": sum(1 for x in cells if x["r2_tail"] >= 0.50 and x["df_diff_med"] <= 0.10)}
    # summary verdicts
    A = res["arms"]["A"]; B = res["arms"]["B"]
    res["判词"] = {
        "判A(A臂)": "过" if A["判A_过"] else "未过",
        "判B_A臂": "过" if A["判B_过"] else "未过",
        "判B_B臂": "过" if B["判B_过"] else "未过",
        "判C(A臂)": "过" if A["判C_过"] else ("未过" if A["判C_过"] is not None else "无票"),
        "τ_deact归属": ("A 表观0.40" if A["判B_过"] and not B["判B_过"] else
                     "B 绝对2.0" if B["判B_过"] and not A["判B_过"] else
                     "双过取A" if A["判B_过"] and B["判B_过"] else "双不过缺口坐实")}
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("arm,file,G,n_sweeps,r2_act,r2_tail,df_diff_med,inst_diff_med,df_mea_med,df_sim_med\n")
        for r in rows_csv:
            f.write(",".join(str(r[k]) for k in
                             ("arm", "file", "G", "n_sweeps", "r2_act", "r2_tail",
                              "df_diff_med", "inst_diff_med", "df_mea_med", "df_sim_med")) + "\n")
    print("\n[verdicts]", json.dumps(res["判词"], ensure_ascii=False), flush=True)
    print(f"[saved] {OUT_JSON}\n[saved] {OUT_CSV}", flush=True)
    # figure: arm-A per-cell R2 scatter + example-cell waveforms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(_sys.executable).parent.parent.parent))
        from daimon_runtime import setup_plot; setup_plot()
        import matplotlib.pyplot as plt
        cells = res["arms"]["A"]["cells"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        xs = [c["r2_act"] for c in cells]; ys = [c["r2_tail"] for c in cells]
        ax[0].scatter(xs, ys, s=60)
        for c in cells:
            ax[0].annotate(c["file"][:12], (c["r2_act"], c["r2_tail"]), fontsize=7,
                           textcoords="offset points", xytext=(4, 4))
        ax[0].axvline(0.8, color="r", ls="--", lw=1); ax[0].axhline(0.5, color="r", ls="--", lw=1)
        ax[0].set_xlabel("R2 activation segment"); ax[0].set_ylabel("R2 tail segment"); ax[0].set_title("(a) arm A per cell")
        if cells:
            best = max(cells, key=lambda c: c["r2_act"])
            sw = load_cell(os.path.join(D, best["file"]))[-1]
            ia, it = unit_current(sw, TAU_D_ARMS["A"])
            ax[1].plot(sw["t_act"], sw["y_act"], "k", lw=1, label="measured")
            ax[1].plot(sw["t_act"], best["G"] * ia, "r", lw=1, label="simulated")
            ax[1].set_title(f"(b) example {best['file'][:16]} V={sw['V']:.0f}mV"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT_PNG, bbox_inches="tight", dpi=130)
        print(f"[saved] {OUT_PNG}", flush=True)
    except Exception as e:
        print(f"[figure] failed (verdicts unaffected): {e}", flush=True)

if __name__ == "__main__":
    main()
