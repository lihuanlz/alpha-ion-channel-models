# -*- coding: utf-8 -*-
"""
Stage2b 波形级 τ 守恒分析：30 药全池
问题：药物是只动占据（幅度）还是动 τ（动力学）？
设计：逐细胞内部分段（vehicle 对照 / drug 稳态 / E-4031），对照锚=车辆期，检验 τ_tail 比值。
各实验室协议不同（sweep 1s~4s）→ 相对窗口 + 自动相位识别；不做跨实验室绝对 τ 比较。
断点续跑：stage2b_ckpt.jsonl；--budget 自停。
"""
import os, json, glob, time, argparse, warnings, zipfile
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from multiprocessing import Pool
warnings.filterwarnings("ignore")

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
CIPA = os.path.join(LAW, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
OUT = os.path.join(LAW, "分析_全药池_2026-11-10")
CKPT = os.path.join(OUT, "stage2b_ckpt.jsonl")

def exp_off(t, a, tau, c):
    return a * np.exp(-(t - t[0]) / tau) + c

def phase_metrics(t, y, sweep_ms):
    """单条 trace：自动相位识别 + τ_tail。返回 dict 或 None"""
    # 全局峰（复极化峰）：限定在后 60% 时程内找（避开阶跃初期伪峰）
    i0 = int(0.4 * len(t))
    ipk = i0 + int(np.argmax(y[i0:]))
    A_peak = float(y[ipk]); t_peak = float(t[ipk])
    # plateau：峰前 25%~10% 时程窗均值
    m_pl = (t >= t_peak - 0.25 * sweep_ms) & (t <= t_peak - 0.10 * sweep_ms)
    if m_pl.sum() < 5:
        return None
    A_plateau = float(np.mean(y[m_pl]))
    # 去激活尾：峰后 3% 到 22% sweep 时程
    m_tail = (t >= t_peak + 0.03 * sweep_ms) & (t <= t_peak + 0.22 * sweep_ms)
    tau_tail = np.nan; tail_r2 = np.nan
    if m_tail.sum() >= 15:
        tt = t[m_tail]; yy = y[m_tail]
        # 尾窗衰减幅度太小 → 不可拟合，标记退化
        if abs(yy[0] - yy[-1]) < 5.0:  # pA
            return {"A_peak": A_peak, "t_peak_frac": t_peak / sweep_ms, "A_plateau": A_plateau,
                    "tau_tail_ms": np.nan, "tail_r2": np.nan, "tau_suspect": True,
                    "shape_ratio": A_peak / A_plateau if abs(A_plateau) > 1e-9 else np.nan}
        try:
            p0 = [yy[0] - yy[-1], 0.08 * sweep_ms, yy[-1]]
            popt, _ = curve_fit(exp_off, tt, yy, p0=p0, maxfev=2500,
                                bounds=([-1e7, 0.002 * sweep_ms, -1e7], [1e7, 1.5 * sweep_ms, 1e7]))
            pred = exp_off(tt, *popt)
            ss_res = float(np.sum((yy - pred) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            lo_b, hi_b = 0.002 * sweep_ms, 1.5 * sweep_ms
            suspect = bool(abs(popt[1] - hi_b) < 0.02 * hi_b or abs(popt[1] - lo_b) < 0.05 * lo_b)
            if popt[1] > 0:
                tau_tail = float(popt[1]); tail_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                if suspect:
                    tau_tail = np.nan  # 撞界即弃
        except Exception:
            pass
    return {"A_peak": A_peak, "t_peak_frac": t_peak / sweep_ms, "A_plateau": A_plateau,
            "tau_tail_ms": tau_tail, "tail_r2": tail_r2,
            "shape_ratio": A_peak / A_plateau if abs(A_plateau) > 1e-9 else np.nan}

def proc_cell(zpath):
    try:
        rel = os.path.relpath(zpath, CIPA)
        parts = rel.split(os.sep)
        lab, drug = parts[0], parts[2]
        blind = "blind" in rel.lower()
        cell = f"{lab}/{drug}/" + os.path.basename(zpath).replace(".csv.zip", "")
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        t = df["t_ms"].to_numpy(float)
        sweep_ms = float(t.max() - t.min())
        cols = [c for c in df.columns if c.startswith("trace_#")]
        if len(cols) < 10:
            return {"ok": False, "cell": cell, "err": f"traces={len(cols)}"}
        # 单位统一：列名后缀决定（_current_pa / _current_A / _current_na ...）
        suffix = cols[0].rsplit("_current_", 1)[-1].lower()
        ufactor = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(suffix)
        if ufactor is None:
            return {"ok": False, "cell": cell, "err": f"未知电流单位 {suffix}"}
        # 全部 trace 指标（单位无关——比值化）
        recs = []
        for c in cols:
            tr = int(c.split("#")[1].split("_")[0])
            y = df[c].to_numpy(float) * ufactor  # 统一 pA
            m = phase_metrics(t, y, sweep_ms)
            if m:
                m["trace"] = tr
                recs.append(m)
        if len(recs) < 10:
            return {"ok": False, "cell": cell, "err": "phase fail"}
        td = pd.DataFrame(recs).sort_values("trace")
        Amax = td["A_peak"].max()
        if not np.isfinite(Amax) or Amax <= 0:
            return {"ok": False, "cell": cell, "err": "Amax<=0"}
        # ---- 三段自动分段 ----
        frac = td["A_peak"] / Amax
        n = len(td)
        # E-4031 段：末尾连续 frac<0.15
        tail_low = (frac < 0.15).to_numpy()
        i_e4 = n
        for i in range(n - 1, -1, -1):
            if tail_low[i]:
                i_e4 = i
            else:
                break
        # vehicle 段：开头连续 frac>0.8
        head_hi = (frac > 0.8).to_numpy()
        i_veh_end = 0
        for i in range(n):
            if head_hi[i]:
                i_veh_end = i
            else:
                break
        veh = td.iloc[:i_veh_end + 1]
        e4 = td.iloc[i_e4:] if i_e4 < n else td.iloc[0:0]
        drug_seg = td.iloc[i_veh_end + 1:i_e4] if i_e4 > i_veh_end + 1 else td.iloc[0:0]
        if len(veh) < 3 or len(drug_seg) < 5:
            return {"ok": False, "cell": cell, "err": f"seg veh={len(veh)} drug={len(drug_seg)}"}
        # 对照参考：vehicle 段后一半
        ref = veh.tail(max(3, len(veh) // 2))
        ref_tau = float(ref["tau_tail_ms"].median())
        ref_A = float(ref["A_peak"].median())
        ref_shape = float(ref["shape_ratio"].median())
        drift = abs(float(veh["A_peak"].iloc[:max(2, len(veh)//3)].mean()) - ref_A) / max(ref_A, 1e-12)
        # drug 稳态：drug 段最后 5 条
        ss = drug_seg.tail(5)
        block_frac = 1 - float(ss["A_peak"].median()) / ref_A if ref_A > 0 else np.nan
        tau_ratio = float(ss["tau_tail_ms"].median()) / ref_tau if ref_tau and np.isfinite(ref_tau) else np.nan
        shape_ratio_chg = float(ss["shape_ratio"].median()) / ref_shape if ref_shape and np.isfinite(ref_shape) else np.nan
        # 起效 τ：drug 段 A_peak vs trace 序号单指数
        onset_tau_tr = np.nan
        if len(drug_seg) >= 8:
            try:
                gg = drug_seg
                y0 = float(gg["A_peak"].iloc[0]); yinf = float(gg["A_peak"].tail(3).mean())
                x = gg["trace"].to_numpy(float) - gg["trace"].iloc[0]
                def dc(xx, tau):
                    return yinf + (y0 - yinf) * np.exp(-xx / tau)
                popt, _ = curve_fit(dc, x, gg["A_peak"].to_numpy(float), p0=[15.0], bounds=([0.5], [2000.0]), maxfev=2000)
                onset_tau_tr = float(popt[0])
            except Exception:
                pass
        qc_pass = (drift <= 0.35) and (ref_A >= 50)
        return {"ok": True, "cell": cell, "lab": lab, "drug": drug, "blind": blind,
                "n_traces": int(n), "sweep_ms": sweep_ms,
                "qc_pass": bool(qc_pass), "control_drift": float(drift),
                "ref_tau_ms": ref_tau, "ref_A_peak": ref_A,
                "block_frac": float(block_frac), "tau_ratio": float(tau_ratio),
                "shape_ratio_chg": float(shape_ratio_chg),
                "onset_tau_traces": onset_tau_tr,
                "e4031_resid": float(e4["A_peak"].median() / ref_A) if len(e4) and ref_A > 0 else np.nan}
    except Exception as e:
        return {"ok": False, "cell": zpath, "err": f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-tasks", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    zips = sorted(glob.glob(os.path.join(CIPA, "lab_*", "*", "*", "subtracted", "ted", "*.csv.zip")))
    print(f"zip 总数 {len(zips)}", flush=True)
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["cell"])
                except Exception:
                    pass
        print(f"续跑：已完成 {len(done)}", flush=True)
    todo = []
    for z in zips:
        rel = os.path.relpath(z, CIPA); parts = rel.split(os.sep)
        key = f"{parts[0]}/{parts[2]}/" + os.path.basename(z).replace(".csv.zip", "")
        if key not in done:
            todo.append(z)
    if a.max_tasks:
        todo = todo[:a.max_tasks]
    print(f"待处理 {len(todo)}", flush=True)
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(proc_cell, todo, chunksize=4), 1):
            with open(CKPT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("预算到点自停", flush=True)
                pool.terminate(); pool.join()
                return
    print(f"本轮完成 {len(todo)}，耗时 {time.time()-t0:.0f}s")
    print("DONE")

if __name__ == "__main__":
    main()
