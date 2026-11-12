# -*- coding: utf-8 -*-
"""
Stage2c 波形级 τ 守恒分析 v2（五实验室三协议自适应）
协议事实（已目验）：
  lab_1: 1s sweep，阶跃+斜坡峰，尾窗 ~130ms
  lab_2: 3.9s，协议乱、有伪迹巨峰
  lab_3: 2s，阶跃+峰+长尾（~1100ms 窗）
  lab_4: 4s，方波阶跃+稀疏测试 trace
  lab_5: 4.5s，阶跃+尖峰+超长尾（~1500ms 窗）
策略：测试 trace 筛选（std 门槛）→ MAD 去伪迹 → E-4031 段锚定 →
      drug 稳态 = E4031 前最后 5 条测试 trace；vehicle = 前段连续高幅段。
      τ 窗 = 峰后 3% → 50% sweep（自适应长尾），r²≥0.5，撞界弃。
输出：stage2c_ckpt.jsonl（逐细胞），汇总另跑 stage2c_汇总出图.py
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
CKPT = os.path.join(OUT, "stage2c_ckpt.jsonl")

def exp_off(t, a, tau, c):
    return a * np.exp(-(t - t[0]) / tau) + c

def trace_metrics(t, y, sweep_ms):
    """单条测试 trace 指标"""
    i0 = int(0.05 * len(t))  # 只排除开头 5% 伪影；峰位置全自适应（lab_1 峰在87%，lab_5 在20%）
    ipk = i0 + int(np.argmax(y[i0:]))
    A_peak = float(y[ipk]); t_peak = float(t[ipk])
    # plateau：峰前 20%~8% sweep；若峰太靠前（lab_5 峰在20%处）plateau 无定义 → NaN
    m_pl = (t >= t_peak - 0.20 * sweep_ms) & (t <= t_peak - 0.08 * sweep_ms)
    if m_pl.sum() >= 5 and (t_peak - 0.20 * sweep_ms) > t[0] + 0.25 * sweep_ms:
        A_plateau = float(np.mean(y[m_pl]))
    else:
        A_plateau = np.nan
    # τ 尾窗：峰后 3% → 50% sweep（受 sweep 末尾限制）
    t_end = float(t[-1])
    t1 = t_peak + 0.03 * sweep_ms
    t2 = min(t_peak + 0.50 * sweep_ms, t_end - 0.01 * sweep_ms)
    m_tail = (t >= t1) & (t <= t2)
    tau_tail = np.nan; tail_r2 = np.nan
    if m_tail.sum() >= 25:
        tt = t[m_tail]; yy = y[m_tail]
        decay_amp = yy[0] - yy[-1]
        if decay_amp > max(3.0, 0.05 * abs(A_peak)):  # 至少有像样衰减
            try:
                p0 = [decay_amp, 0.1 * sweep_ms, yy[-1]]
                hi_b = 2.0 * sweep_ms
                popt, _ = curve_fit(exp_off, tt, yy, p0=p0, maxfev=3000,
                                    bounds=([-1e7, 0.001 * sweep_ms, -1e7], [1e7, hi_b, 1e7]))
                pred = exp_off(tt, *popt)
                ss_res = float(np.sum((yy - pred) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                lo_b = 0.001 * sweep_ms
                hit_bound = abs(popt[1] - hi_b) < 0.02 * hi_b or abs(popt[1] - lo_b) < 0.05 * lo_b
                if popt[1] > 0 and r2 >= 0.5 and not hit_bound:
                    tau_tail = float(popt[1]); tail_r2 = float(r2)
            except Exception:
                pass
    return {"A_peak": A_peak, "t_peak_frac": t_peak / sweep_ms, "A_plateau": A_plateau,
            "tau_tail_ms": tau_tail, "tail_r2": tail_r2,
            "shape_ratio": A_peak / A_plateau if np.isfinite(A_plateau) and abs(A_plateau) > 1.0 else np.nan}

def proc_cell(zpath):
    try:
        rel = os.path.relpath(zpath, CIPA); parts = rel.split(os.sep)
        lab, drug = parts[0], parts[2]
        blind = "blind" in rel.lower()
        cell = f"{lab}/{drug}/" + os.path.basename(zpath).replace(".csv.zip", "")
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        t = df["t_ms"].to_numpy(float)
        sweep_ms = float(t.max() - t.min())
        cols = [c for c in df.columns if c.startswith("trace_#")]
        if len(cols) < 8:
            return {"ok": False, "cell": cell, "err": f"traces={len(cols)}"}
        uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(
            cols[0].rsplit("_current_", 1)[-1].lower())
        if uf is None:
            return {"ok": False, "cell": cell, "err": "单位未知"}
        # ---- 测试 trace 筛选：std 分层（保持平线 std≈0.06，测试波形 std≈40-160）----
        ys = {int(c.split("#")[1].split("_")[0]): df[c].to_numpy(float) * uf for c in cols}
        stds = {k: float(np.std(v)) for k, v in ys.items()}
        std_arr = np.array(list(stds.values()))
        thr = max(5.0, 0.05 * np.percentile(std_arr, 95))
        test_trs = sorted([k for k, s in stds.items() if s > thr])
        if len(test_trs) < 6:
            return {"ok": False, "cell": cell, "err": f"test_traces={len(test_trs)}"}
        # ---- 逐 trace 指标；仅剔极端伪迹（ptp>4000 或 |max|>3000 pA）----
        recs = []
        for tr in test_trs:
            y = ys[tr]
            if np.ptp(y) > 4000 or abs(np.max(y)) > 3000:
                continue
            m = trace_metrics(t, y, sweep_ms)
            m["trace"] = tr
            recs.append(m)
        td = pd.DataFrame(recs).sort_values("trace").reset_index(drop=True)
        if len(td) < 6:
            return {"ok": False, "cell": cell, "err": "post-artifact too few"}
        # 稳健幅度参考：p90（抗单点伪迹尖峰）
        Amax = float(np.percentile(td["A_peak"], 90))
        if Amax <= 20:
            return {"ok": False, "cell": cell, "err": f"Amax={Amax:.1f}pA 太低"}
        frac = td["A_peak"] / Amax
        n = len(td)
        # ---- E-4031 段锚：末尾连续段 幅度 < 0.2×前1/3段中位（相对前段水平而非全局最大）----
        head_level = float(td["A_peak"].iloc[:max(3, n // 3)].median())
        if head_level <= 0:
            head_level = Amax
        tail_low = (td["A_peak"] < 0.2 * head_level).to_numpy()
        i_e4 = n
        for i in range(n - 1, -1, -1):
            if tail_low[i]:
                i_e4 = i
            else:
                break
        e4_found = (n - i_e4) >= 3
        # ---- drug 稳态：E4031 段前最后 5 条 ----
        ss_end = i_e4 if e4_found else n
        ss = td.iloc[max(0, ss_end - 5):ss_end]
        # ---- vehicle：前 1/3 内 frac>0.55 的连续开头段 ----
        i_veh_end = -1
        for i in range(min(n // 3 + 1, n)):
            if frac.iloc[i] > 0.55:
                i_veh_end = i
            else:
                break
        veh = td.iloc[:i_veh_end + 1] if i_veh_end >= 2 else td.iloc[0:0]
        seg_warn = ""
        if len(veh) < 3:
            # 退路：取开头 3 条且标注
            veh = td.iloc[:3]
            seg_warn = "veh_fallback"
        if not e4_found:
            seg_warn = (seg_warn + "|" if seg_warn else "") + "no_e4031"
        ref_tau = float(veh["tau_tail_ms"].dropna().median()) if veh["tau_tail_ms"].notna().any() else np.nan
        ref_A = float(veh["A_peak"].median())
        ref_shape = float(veh["shape_ratio"].dropna().median()) if veh["shape_ratio"].notna().any() else np.nan
        drift = abs(float(td["A_peak"].iloc[:3].mean()) - ref_A) / max(ref_A, 1e-12)
        ss_tau = float(ss["tau_tail_ms"].dropna().median()) if ss["tau_tail_ms"].notna().any() else np.nan
        ss_A = float(ss["A_peak"].median())
        ss_shape = float(ss["shape_ratio"].dropna().median()) if ss["shape_ratio"].notna().any() else np.nan
        return {"ok": True, "cell": cell, "lab": lab, "drug": drug, "blind": blind,
                "sweep_ms": sweep_ms, "n_test_traces": int(n), "seg_warn": seg_warn,
                "control_drift": float(drift),
                "ref_A_peak": ref_A, "ref_tau_ms": ref_tau, "ref_shape": ref_shape,
                "ss_A_peak": ss_A, "ss_tau_ms": ss_tau, "ss_shape": ss_shape,
                "block_frac": float(1 - ss_A / ref_A) if ref_A > 0 else np.nan,
                "tau_ratio": ss_tau / ref_tau if np.isfinite(ss_tau) and np.isfinite(ref_tau) and ref_tau > 0 else np.nan,
                "shape_chg": ss_shape / ref_shape if np.isfinite(ss_shape) and np.isfinite(ref_shape) and ref_shape > 0 else np.nan,
                "qc_pass": bool(drift <= 0.35 and ref_A >= 50 and len(veh) >= 3)}
    except Exception as e:
        return {"ok": False, "cell": zpath, "err": f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    t0 = time.time()
    zips = sorted(glob.glob(os.path.join(CIPA, "lab_*", "*", "*", "subtracted", "ted", "*.csv.zip")))
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        done.add(r["cell"])  # 只把成功的算作 done，fail 的重跑
                except Exception:
                    pass
    todo = []
    for z in zips:
        rel = os.path.relpath(z, CIPA); parts = rel.split(os.sep)
        key = f"{parts[0]}/{parts[2]}/" + os.path.basename(z).replace(".csv.zip", "")
        if key not in done:
            todo.append(z)
    print(f"总 {len(zips)}，已完成 {len(done)}，待处理 {len(todo)}", flush=True)
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(proc_cell, todo, chunksize=4), 1):
            with open(CKPT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("预算到点自停", flush=True)
                pool.terminate(); pool.join()
                return
    print(f"完成 {len(todo)}，耗时 {time.time()-t0:.0f}s")
    print("DONE")

if __name__ == "__main__":
    main()
