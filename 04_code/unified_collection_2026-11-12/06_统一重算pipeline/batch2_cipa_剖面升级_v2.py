# -*- coding: utf-8 -*-
"""
统一重算 Batch 2：CiPA 五实验室全药池剖面似然重提（v2）
方法基座 = stage2d（无元数据分段启发式 + 单位自适应 + _current_ 列过滤）
升级点：
  1) 逐 sweep τ_tail 单指数拟合 + 剖面似然 95% CI（固定τ时 a,c 线性最小二乘）
  2) 逐 sweep 全量留存（cipa2_sweeps.csv）
  3) 细胞级 ref/ss τ 中位数 + 逐 sweep IQR + 剖面似然合并带
  4) τ_ratio 及其 bootstrap CI（veh vs ss 逐 sweep 重抽样 2000 次）
输出：统一重算_2026-11-11/cipa2_ckpt.jsonl（断点续跑）
      cipa2_sweeps.csv / cipa2_cells.csv
跑法：python batch2_cipa_剖面升级_v2.py --budget 250 --workers 20，反复至 ALL DONE
"""
import os, json, glob, time, argparse, warnings, zipfile
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as sst
from multiprocessing import Pool
warnings.filterwarnings("ignore")

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
CIPA = os.path.join(LAW, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
OUT = os.path.join(LAW, "统一重算_2026-11-11")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, "cipa2_ckpt.jsonl")

def exp_off(t, a, tau, c):
    return a * np.exp(-(t - t[0]) / tau) + c

def profile_tau_linear(t, y, tau_hat, sweep_ms, alpha=0.05):
    """exp_off 中 tau 的剖面似然 CI；固定 tau 时 (a,c) 线性。"""
    n = len(y); p = 3
    if n < p + 3:
        return np.nan, np.nan
    t0 = t[0]
    lo_b = 0.001 * sweep_ms; hi_b = 2.0 * sweep_ms
    def sse_given_tau(tau):
        E = np.exp(-(t - t0) / tau)
        X = np.column_stack([E, np.ones_like(E)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - (X @ coef)
        return float(r @ r)
    sse_best = sse_given_tau(tau_hat)
    thresh = sse_best * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    grid_lo = np.linspace(max(lo_b, tau_hat / 40), tau_hat, 30)
    grid_hi = np.linspace(tau_hat, min(hi_b, tau_hat * 40), 30)
    lo = grid_lo[0]
    for tau in grid_lo[::-1]:
        if sse_given_tau(tau) > thresh:
            lo = tau; break
    hi = grid_hi[-1]
    for tau in grid_hi:
        if sse_given_tau(tau) > thresh:
            hi = tau; break
    def bisect(ta, tb):
        for _ in range(20):
            tm = 0.5 * (ta + tb)
            if sse_given_tau(tm) > thresh: ta = tm
            else: tb = tm
        return 0.5 * (ta + tb)
    if lo < tau_hat and lo > grid_lo[0]:
        lo = bisect(lo, min(lo * 1.3, tau_hat))
    if hi > tau_hat and hi < grid_hi[-1]:
        hi = bisect(max(hi / 1.3, tau_hat), hi)
    return float(lo), float(hi)

def trace_metrics(t, y, sweep_ms):
    i0 = int(0.05 * len(t))
    ipk = i0 + int(np.argmax(y[i0:]))
    A_peak = float(y[ipk]); t_peak = float(t[ipk])
    m_pl = (t >= t_peak - 0.20 * sweep_ms) & (t <= t_peak - 0.08 * sweep_ms)
    A_plateau = float(np.mean(y[m_pl])) if m_pl.sum() >= 5 and (t_peak - 0.20 * sweep_ms) > t[0] + 0.25 * sweep_ms else np.nan
    t_end = float(t[-1])
    t1 = t_peak + 0.03 * sweep_ms
    t2 = min(t_peak + 0.50 * sweep_ms, t_end - 0.01 * sweep_ms)
    m_tail = (t >= t1) & (t <= t2)
    tau_tail = np.nan; tail_r2 = np.nan; tau_lo = np.nan; tau_hi = np.nan
    if m_tail.sum() >= 25:
        tt = t[m_tail]; yy = y[m_tail]
        decay_amp = yy[0] - yy[-1]
        if decay_amp > max(3.0, 0.05 * abs(A_peak)):
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
                    tau_lo, tau_hi = profile_tau_linear(tt, yy, tau_tail, sweep_ms)
            except Exception:
                pass
    return {"A_peak": A_peak, "t_peak_frac": t_peak / sweep_ms, "A_plateau": A_plateau,
            "tau_tail_ms": tau_tail, "tail_r2": tail_r2,
            "tau_prof_lo": tau_lo, "tau_prof_hi": tau_hi,
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
        cols = [c for c in df.columns if c.startswith("trace_#") and "_current_" in c]
        if len(cols) < 8:
            return {"ok": False, "cell": cell, "err": f"traces={len(cols)}"}
        uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(
            cols[0].rsplit("_current_", 1)[-1].lower())
        if uf is None:
            return {"ok": False, "cell": cell, "err": "单位未知"}
        ys = {int(c.split("#")[1].split("_")[0]): df[c].to_numpy(float) * uf for c in cols}
        stds = {k: float(np.std(v)) for k, v in ys.items()}
        std_arr = np.array(list(stds.values()))
        thr = max(5.0, 0.05 * np.percentile(std_arr, 95))
        test_trs = sorted([k for k, s in stds.items() if s > thr])
        if len(test_trs) < 12:
            return {"ok": False, "cell": cell, "err": f"test_traces={len(test_trs)}"}
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
        Amax = float(np.percentile(td["A_peak"], 90))
        if Amax <= 20:
            return {"ok": False, "cell": cell, "err": f"Amax={Amax:.1f}pA 太低"}
        frac = td["A_peak"] / Amax
        n = len(td)
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
        ss_end = i_e4 if e4_found else n
        ss = td.iloc[max(0, ss_end - 5):ss_end]
        i_veh_end = -1
        for i in range(min(n // 3 + 1, n)):
            if frac.iloc[i] > 0.55:
                i_veh_end = i
            else:
                break
        veh = td.iloc[:i_veh_end + 1] if i_veh_end >= 2 else td.iloc[0:0]
        seg_warn = ""
        if len(veh) < 3:
            veh = td.iloc[:3]
            seg_warn = "veh_fallback"
        if not e4_found:
            seg_warn = (seg_warn + "|" if seg_warn else "") + "no_e4031"
        veh_tau = veh["tau_tail_ms"].dropna().to_numpy()
        ss_tau_arr = ss["tau_tail_ms"].dropna().to_numpy()
        ref_tau = float(np.median(veh_tau)) if len(veh_tau) else np.nan
        ss_tau = float(np.median(ss_tau_arr)) if len(ss_tau_arr) else np.nan
        ref_A = float(veh["A_peak"].median())
        ss_A = float(ss["A_peak"].median())
        # τ_ratio bootstrap CI
        tr_lo = np.nan; tr_hi = np.nan; tau_ratio = np.nan
        if len(veh_tau) >= 3 and len(ss_tau_arr) >= 2:
            rng = np.random.default_rng(20261111)
            boots = []
            for _ in range(2000):
                vb = rng.choice(veh_tau, len(veh_tau), replace=True)
                sb = rng.choice(ss_tau_arr, len(ss_tau_arr), replace=True)
                mb = np.median(vb)
                if mb > 0:
                    boots.append(np.median(sb) / mb)
            if boots:
                tr_lo, tr_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            tau_ratio = ss_tau / ref_tau if ref_tau > 0 else np.nan
        drift = abs(float(td["A_peak"].iloc[:3].mean()) - ref_A) / max(ref_A, 1e-12)
        # 逐 sweep 留存（τ 有效行）
        sweeps = td[np.isfinite(td["tau_tail_ms"])][
            ["trace", "tau_tail_ms", "tau_prof_lo", "tau_prof_hi", "tail_r2", "A_peak"]].to_dict("records")
        # veh/ss 段标记进 sweep
        veh_tr = set(veh["trace"].tolist()); ss_tr = set(ss["trace"].tolist())
        for s in sweeps:
            s["seg"] = "veh" if s["trace"] in veh_tr else ("ss" if s["trace"] in ss_tr else "mid")
        return {"ok": True, "cell": cell, "lab": lab, "drug": drug, "blind": blind,
                "sweep_ms": sweep_ms, "n_test_traces": int(n), "seg_warn": seg_warn,
                "control_drift": float(drift),
                "ref_A_peak": ref_A, "ref_tau_ms": ref_tau,
                "ref_tau_iqr_lo": float(np.quantile(veh_tau, 0.25)) if len(veh_tau) else np.nan,
                "ref_tau_iqr_hi": float(np.quantile(veh_tau, 0.75)) if len(veh_tau) else np.nan,
                "ss_A_peak": ss_A, "ss_tau_ms": ss_tau,
                "ss_tau_iqr_lo": float(np.quantile(ss_tau_arr, 0.25)) if len(ss_tau_arr) else np.nan,
                "ss_tau_iqr_hi": float(np.quantile(ss_tau_arr, 0.75)) if len(ss_tau_arr) else np.nan,
                "block_frac": float(1 - ss_A / ref_A) if ref_A > 0 else np.nan,
                "tau_ratio": tau_ratio, "tau_ratio_boot_lo": tr_lo, "tau_ratio_boot_hi": tr_hi,
                "qc_pass": bool(drift <= 0.35 and ref_A >= 50 and len(veh) >= 3),
                "sweeps": sweeps}
    except Exception as e:
        return {"ok": False, "cell": zpath, "err": f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-tasks", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    zips = sorted(glob.glob(os.path.join(CIPA, "lab_*", "*", "*", "subtracted", "ted", "*.csv.zip")))
    print(f"总 zip {len(zips)}", flush=True)
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        done.add(r["cell"])
                except Exception:
                    pass
    todo = []
    for z in zips:
        rel = os.path.relpath(z, CIPA); parts = rel.split(os.sep)
        key = f"{parts[0]}/{parts[2]}/" + os.path.basename(z).replace(".csv.zip", "")
        if key not in done:
            todo.append(z)
    if a.max_tasks:
        todo = todo[:a.max_tasks]
    print(f"已完成 {len(done)}，待处理 {len(todo)}", flush=True)
    if not todo:
        print("ALL DONE"); finalize(); return
    n0 = 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(proc_cell, todo, chunksize=2):
            with open(CKPT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n0 += 1
            if n0 % 100 == 0:
                print(f"  {n0}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("预算到点自停（可续跑）", flush=True)
                pool.terminate(); pool.join(); return
    print("ALL DONE", flush=True)
    finalize()

def finalize():
    seen = {}
    with open(CKPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r.get("ok"):
                    seen[r["cell"]] = r
            except Exception:
                pass
    rows = list(seen.values())
    sw_rows = []; cell_rows = []
    for r in rows:
        for s in r.get("sweeps", []):
            sw_rows.append({"lab": r["lab"], "drug": r["drug"], "blind": r["blind"],
                            "cell": r["cell"], "qc_pass": r["qc_pass"], **s})
        cr = {k: v for k, v in r.items() if k != "sweeps"}
        cell_rows.append(cr)
    pd.DataFrame(sw_rows).to_csv(os.path.join(OUT, "cipa2_sweeps.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cell_rows).to_csv(os.path.join(OUT, "cipa2_cells.csv"), index=False, encoding="utf-8-sig")
    print(f"落盘：sweeps {len(sw_rows)} 行，cells {len(cell_rows)} 行")

if __name__ == "__main__":
    main()
