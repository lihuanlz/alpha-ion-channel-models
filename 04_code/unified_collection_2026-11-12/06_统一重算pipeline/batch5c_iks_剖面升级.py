# -*- coding: utf-8 -*-
"""
统一重算 Batch 5c：IKs（iks_chan_8226585，Fedida 系）激活/去激活 τ 剖面似然重提
方法基座 = Kα1（epoch 解析：epoch2=阶跃段、epoch3=-40mV 尾段；带延迟单指数；锚C表观τ）
升级：τ_act 二维(d,τ)网格+Iss线性+剖面；τ_app 单参数剖面
输出去重后逐细胞逐电压行：iks5_unified.csv
"""
import os, re, glob, time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as sst
import pyabf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档\2026-10-16_清理\04_细胞线4\_归档\数据\iks_chan_8226585"
TAU_FIT_MAX = 30.0

def rec_id(fn):
    s = fn[:-4] if fn.lower().endswith(".abf") else fn
    for pat in [" after LS and HMR subtraction", r"_reduced_\d+",
                " after LS w sweep 1 and 4", " after LS",
                " LS updated control trace selection", " LS updated2", " LS updated",
                " LS endo", " LS MC", " YD LS", " LS copy", " LS 2",
                r" LS \+60", r" \+60 LS", r" \+60", " LS"]:
        s = re.sub(pat, "", s, flags=re.I)
    return re.sub(r"[ \-]", "_", s.strip().lower())

def act_model(t, Iss, d, tau):
    y = np.zeros_like(t)
    m = t > d
    y[m] = Iss * (1.0 - np.exp(-(t[m] - d) / tau))
    return y

def prof_tau_act(tw, yw, tau_hat, d_hat, alpha=0.05):
    """act_model 的 τ 剖面：固定 τ，d 网格 + Iss 线性。"""
    n = len(yw); p = 3
    def solve(tau):
        best = None
        for d in np.linspace(0, 1.5, 25):
            x = np.where(tw > d, 1.0 - np.exp(-(np.maximum(tw - d, 0)) / tau), 0.0)
            X = np.column_stack([x, np.ones(n)])
            sol, *_ = np.linalg.lstsq(X, yw, rcond=None)
            r = yw - X @ sol
            sse = float(r @ r)
            if best is None or sse < best[0]:
                best = (sse, d)
        return best
    sse0, _ = solve(tau_hat)
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    grid = np.exp(np.linspace(np.log(0.01), np.log(TAU_FIT_MAX), 40))
    lo = grid[0]; hi = grid[-1]
    for cand in grid[grid < tau_hat][::-1]:
        if solve(cand)[0] > thresh:
            lo = cand; break
    for cand in grid[grid > tau_hat]:
        if solve(cand)[0] > thresh:
            hi = cand; break
    return lo, hi

def prof_tau_app(tt, tseg, tau_hat, Cfix, alpha=0.05):
    """锚 C 单指数 τ 剖面（A 线性）。"""
    n = len(tt); p = 2
    def solve(tau):
        X = np.exp(-tt / tau)[:, None]
        sol, *_ = np.linalg.lstsq(X, tseg - Cfix, rcond=None)
        r = tseg - (sol[0] * np.exp(-tt / tau) + Cfix)
        return float(r @ r)
    sse0 = solve(tau_hat)
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    grid = np.exp(np.linspace(np.log(0.02), np.log(60.0), 40))
    lo = grid[0]; hi = grid[-1]
    for cand in grid[grid < tau_hat][::-1]:
        if solve(cand) > thresh:
            lo = cand; break
    for cand in grid[grid > tau_hat]:
        if solve(cand) > thresh:
            hi = cand; break
    return lo, hi

def analyze(path):
    a = pyabf.ABF(path)
    dt = 1.0 / a.sampleRate
    rows = []
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
        seg = yc[p2a:p2b]
        t = np.arange(len(seg)) * dt
        if len(seg) > int(0.2 / dt):
            iss0 = float(np.mean(seg[-int(0.1 / dt):]))
        else:
            iss0 = float(seg.max())
        if iss0 > 0:
            try:
                i15 = int(np.argmax(seg > 0.15 * iss0))
                tw = t[i15:]
                popt, _ = curve_fit(act_model, tw, seg[i15:],
                                    p0=[iss0, max(tw[0] - 0.05, 0.0), 0.5],
                                    bounds=([0, 0, 0.01], [np.inf, 1.5, TAU_FIT_MAX]),
                                    maxfev=20000)
                Iss, d0, tau0 = popt
                if tau0 < TAU_FIT_MAX - 1e-6:
                    lo, hi = prof_tau_act(tw, seg[i15:], tau0, d0)
                    rows.append({"parameter": "tau_act", "process": "activation",
                                 "voltage_mV": v_step, "value": tau0 * 1e3,
                                 "prof_lo": lo * 1e3, "prof_hi": hi * 1e3,
                                 "model": "delay_monoexp_profile", "r2": np.nan})
            except Exception:
                pass
        # 尾段去激活
        tseg = yc[p3a:p3b]
        tt = np.arange(len(tseg)) * dt
        if len(tseg) > 50:
            i0 = slice(int(0.005 / dt), int(0.025 / dt))
            i1 = slice(len(tseg) - int(0.05 / dt), len(tseg))
            start = float(np.mean(tseg[i0])); end = float(np.mean(tseg[i1]))
            if start > 0:
                try:
                    popt, _ = curve_fit(lambda x, A, tau: A * np.exp(-x / tau) + end,
                                        tt, tseg, p0=[max(start - end, 1.0), 0.5],
                                        bounds=([0, 0.02], [np.inf, 60.0]), maxfev=20000)
                    tau_app = float(popt[1])
                    lo, hi = prof_tau_app(tt, tseg, tau_app, end)
                    rows.append({"parameter": "tau_deact", "process": "deactivation",
                                 "voltage_mV": v_step, "value": tau_app * 1e3,
                                 "prof_lo": lo * 1e3, "prof_hi": hi * 1e3,
                                 "model": "anchoredC_monoexp_profile", "r2": np.nan})
                except Exception:
                    pass
    return rows

def process_one(f):
    """worker：分析单文件，返回 (rid, rows, err)。"""
    rid = rec_id(os.path.basename(f))
    try:
        rows = analyze(f)
        for r in rows:
            r.update({"source_big": "iks_fedida", "dataset": "iks_chan_8226585",
                      "cell_id": rid, "channel": "IKs", "family": "Kv",
                      "species": "", "temp_C": np.nan,
                      "se_pcov": np.nan, "p_model": np.nan,
                      "qc_flag": "ok", "method": "profile_v1",
                      "drug": "", "blind": False, "A_peak_pA": np.nan})
        return rid, rows, None
    except Exception as e:
        return rid, [], f"{type(e).__name__}: {e}"

def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(DATA, "*.abf")))
    print(f"ABF 文件 {len(files)}")
    # 去重：按 rec_id 选最优版本
    groups = {}
    for f in files:
        groups.setdefault(rec_id(os.path.basename(f)), []).append(f)
    picked = []
    for rid, grp in groups.items():
        def score(x):
            fn = os.path.basename(x).lower()
            return (2 if "updated" in fn else 0) - (1 if "reduced" in fn else 0)
        grp2 = sorted(grp, key=lambda x: (score(x), os.path.getsize(x)))
        sel = grp2[-1]
        picked.append(sel)
    print(f"去重后 {len(picked)} 个记录", flush=True)
    # 断点续跑：逐文件 part csv
    pdir = os.path.join(HERE, "iks5_parts")
    os.makedirs(pdir, exist_ok=True)
    todo = [f for f in picked
            if not os.path.exists(os.path.join(pdir, rec_id(os.path.basename(f)) + ".csv"))]
    print(f"待跑 {len(todo)} / {len(picked)}", flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_done = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(process_one, f): f for f in todo}
        for fut in as_completed(futs):
            rid, rows, err = fut.result()
            pf = os.path.join(pdir, rid + ".csv")
            if rows:
                pd.DataFrame(rows).to_csv(pf, index=False, encoding="utf-8-sig")
            else:
                pd.DataFrame({"cell_id": [rid], "err": [err or "no rows"]}).to_csv(
                    pf, index=False, encoding="utf-8-sig")
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                print(f"  进度 {n_done}/{len(todo)}  耗时 {time.time()-t0:.0f}s", flush=True)
    # 汇总
    parts = sorted(glob.glob(os.path.join(pdir, "*.csv")))
    frames = []
    for pf in parts:
        d = pd.read_csv(pf)
        if "parameter" in d.columns:
            frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fp = os.path.join(HERE, "iks5_unified.csv")
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"落盘 {fp}：{len(df)} 行，记录 {len(parts)}，耗时 {time.time()-t0:.0f}s")
    if len(df):
        print(df.groupby("parameter")["value"].agg(["count", "median"]).to_string())

if __name__ == "__main__":
    main()
