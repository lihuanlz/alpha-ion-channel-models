# -*- coding: utf-8 -*-
"""
统一重算 Batch 2：CiPA 五实验室全药池——剖面似然升级重提
基于 stage2_波形级分析.py 的方法学升级：
  1) τ_tail 逐 sweep 单指数拟合 + 剖面似然 95% CI（固定 τ 时 a,c 线性最小二乘秒解）
  2) onset_tau 阻断起效 τ + 剖面似然 CI（单参数 SSE 网格扫描）
  3) 逐 sweep τ_tail 全量留存（供细胞级中位数 + bootstrap CI）
  4) τ_ratio = τ_ss / τ_ref，细胞级 CI 用逐 sweep 分布 bootstrap
输出：
  stage2p_ckpt.jsonl（断点续跑）
  cipa_profile_sweeps.csv（逐 sweep 级，进统一总表的行级源）
  cipa_profile_cells.csv（细胞×浓度块级汇总 + CI）
跑法：python batch2_cipa_剖面升级.py --budget 250 --workers 20（反复调用至 ALL DONE）
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
CKPT = os.path.join(OUT, "stage2p_ckpt.jsonl")

UNIT = {"uM": 1.0, "µM": 1.0, "μM": 1.0, "nM": 1e-3, "pM": 1e-6, "mM": 1e3}

# ---------- 剖面似然工具 ----------
def profile_tau_linear(t, y, tau_hat, grid_lo=5, grid_hi=3000, n_grid=40, alpha=0.05):
    """exp_off(t,a,tau,c) 中 tau 的剖面似然 CI。
    固定 tau 时模型对 (a,c) 线性：y = a*E + c，E=exp(-(t-t0)/tau)。最小二乘秒解。
    判据：SSE(tau) <= SSE_best*(1 + F(1,n-p;1-alpha)/(n-p))"""
    n = len(y); p = 3
    if n < p + 3:
        return np.nan, np.nan
    t0 = t[0]
    def sse_given_tau(tau):
        E = np.exp(-(t - t0) / tau)
        X = np.column_stack([E, np.ones_like(E)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_, c_ = coef[0], coef[1]
        r = y - (a_ * E + c_)
        return float(r @ r)
    sse_best = sse_given_tau(tau_hat)
    thresh = sse_best * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    lo_grid = np.linspace(grid_lo, tau_hat, n_grid)
    hi_grid = np.linspace(tau_hat, grid_hi, n_grid)
    lo = np.nan
    for tau in lo_grid[::-1]:
        if sse_given_tau(tau) > thresh:
            lo = tau
            break
    else:
        lo = grid_lo
    hi = np.nan
    for tau in hi_grid:
        if sse_given_tau(tau) > thresh:
            hi = tau
            break
    else:
        hi = grid_hi
    # 细化（二分）
    def bisect(ta, tb):
        for _ in range(24):
            tm = 0.5 * (ta + tb)
            if sse_given_tau(tm) > thresh:
                ta = tm
            else:
                tb = tm
        return 0.5 * (ta + tb)
    if np.isfinite(lo) and lo < tau_hat and lo > grid_lo:
        lo = bisect(lo, min(lo * 1.35, tau_hat))
    if np.isfinite(hi) and hi > tau_hat and hi < grid_hi:
        hi = bisect(max(hi / 1.35, tau_hat), hi)
    return lo, hi

def profile_onset_tau(t, y, tau_hat, t0, alpha=0.05):
    """decay(tt,tau)=yinf+(y0-yinf)*exp(-(tt-t0)/tau) 单参数剖面。
    y0,yinf 由数据预取（首点/末3点均值），只有 tau 自由。"""
    n = len(y); p = 1
    if n < 6:
        return np.nan, np.nan
    y0 = y[0]; yinf = np.mean(y[-3:])
    def sse(tau):
        r = y - (yinf + (y0 - yinf) * np.exp(-(t - t0) / tau))
        return float(r @ r)
    sse_best = sse(tau_hat)
    thresh = sse_best * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    grid = np.geomspace(1, 5000, 60)
    lo = np.nan; hi = np.nan
    below = grid[grid < tau_hat]
    for tau in below[::-1]:
        if sse(tau) > thresh:
            lo = tau; break
    else:
        lo = 1.0
    above = grid[grid > tau_hat]
    for tau in above:
        if sse(tau) > thresh:
            hi = tau; break
    else:
        hi = 5000.0
    def bisect(ta, tb):
        for _ in range(24):
            tm = 0.5 * (ta + tb)
            if sse(tm) > thresh:
                ta = tm
            else:
                tb = tm
        return 0.5 * (ta + tb)
    if lo < tau_hat and lo > 1.0:
        lo = bisect(lo, min(lo * 1.4, tau_hat))
    if hi > tau_hat and hi < 5000.0:
        hi = bisect(max(hi / 1.4, tau_hat), hi)
    return lo, hi

# ---------- 单 sweep 指标（含剖面 CI） ----------
def exp_off(t, a, tau, c):
    return a * np.exp(-(t - t[0]) / tau) + c

def trace_metrics(t, y, do_profile=True):
    m_plateau = (t >= 700) & (t <= 790)
    m_peak = (t >= 820) & (t <= 905)
    if m_plateau.sum() < 5 or m_peak.sum() < 5:
        return None
    A_plateau = float(np.mean(y[m_plateau]))
    ipk = np.argmax(y[m_peak])
    t_peak = float(t[m_peak][ipk]); A_peak = float(y[m_peak][ipk])
    m_tail = (t >= t_peak + 8) & (t <= 995)
    tau_tail = np.nan; tail_r2 = np.nan; tau_lo = np.nan; tau_hi = np.nan
    if m_tail.sum() >= 20:
        tt = t[m_tail]; yy = y[m_tail]
        try:
            p0 = [yy[0] - yy[-1], 150.0, yy[-1]]
            popt, _ = curve_fit(exp_off, tt, yy, p0=p0, maxfev=3000,
                                bounds=([-1e5, 5, -1e5], [1e5, 3000, 1e5]))
            pred = exp_off(tt, *popt)
            ss_res = float(np.sum((yy - pred) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            tau_tail = float(popt[1]); tail_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            if do_profile and np.isfinite(tau_tail) and 5 < tau_tail < 3000:
                tau_lo, tau_hi = profile_tau_linear(tt, yy, tau_tail)
        except Exception:
            pass
    return {"A_plateau": A_plateau, "A_peak": A_peak, "t_peak": t_peak,
            "tau_tail_ms": tau_tail, "tail_r2": tail_r2,
            "tau_prof_lo": tau_lo, "tau_prof_hi": tau_hi,
            "shape_ratio": A_peak / A_plateau if abs(A_plateau) > 1e-9 else np.nan}

def proc_cell(args):
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
            mm = trace_metrics(t, y, do_profile=True)
            if mm is None:
                continue
            mm.update({"TRACENUM": tr, "ELTIME_s": float(row["ELTIME"]),
                       "LIQUID": row["LIQUID"], "CONC_uM": float(row["CONC"]) * unit_factor,
                       "CTLFL": str(row.get("CTLFL", "")) == "Y"})
            recs.append(mm)
        if not recs:
            return {"ok": False, "cell": cell, "err": "no traces matched"}
        tr_df = pd.DataFrame(recs)
        veh = tr_df[(tr_df["CTLFL"]) | (tr_df["LIQUID"].str.contains("Bath|Vehicle", caseless=True, na=False))]
        drug_rows = tr_df[tr_df["LIQUID"].str.strip().str.lower() == drug.strip().lower()]
        e4031 = tr_df[tr_df["LIQUID"].str.contains("E-4031", caseless=True, na=False)]
        if len(veh) < 3:
            return {"ok": False, "cell": cell, "err": f"vehicle traces={len(veh)}"}
        veh_last = veh.tail(5)
        ref = {"A_peak": float(veh_last["A_peak"].median()),
               "tau_tail_ms": float(veh_last["tau_tail_ms"].median()),
               "tau_tail_iqr_lo": float(veh_last["tau_tail_ms"].quantile(0.25)),
               "tau_tail_iqr_hi": float(veh_last["tau_tail_ms"].quantile(0.75)),
               "tau_prof_lo": float(np.nanmedian(veh_last["tau_prof_lo"])),
               "tau_prof_hi": float(np.nanmedian(veh_last["tau_prof_hi"]))}
        drift = abs(float(veh.head(3)["A_peak"].median()) - ref["A_peak"]) / max(abs(ref["A_peak"]), 1e-9)
        qc_pass = (drift <= 0.30) and (ref["A_peak"] >= 50)
        blocks = []
        for conc, g in drug_rows.groupby("CONC_uM"):
            g = g.sort_values("ELTIME_s")
            ss = g.tail(3)
            onset_tau = np.nan; onset_lo = np.nan; onset_hi = np.nan
            if len(g) >= 6 and g["ELTIME_s"].ptp() > 5:
                try:
                    gg = g[g["A_peak"] > -1e9]
                    y0 = gg["A_peak"].iloc[0]; yinf = gg["A_peak"].iloc[-3:].mean()
                    t0_ = gg["ELTIME_s"].iloc[0]
                    def decay(tt, tau):
                        return yinf + (y0 - yinf) * np.exp(-(tt - t0_) / tau)
                    popt, _ = curve_fit(decay, gg["ELTIME_s"].to_numpy(), gg["A_peak"].to_numpy(),
                                        p0=[60.0], bounds=([1.0], [5000.0]), maxfev=3000)
                    onset_tau = float(popt[0])
                    onset_lo, onset_hi = profile_onset_tau(
                        gg["ELTIME_s"].to_numpy(), gg["A_peak"].to_numpy(), onset_tau, t0_)
                except Exception:
                    pass
            blocks.append({"conc_uM": float(conc), "n_traces": int(len(g)),
                           "A_peak_ss": float(ss["A_peak"].median()),
                           "tau_tail_ss": float(ss["tau_tail_ms"].median()),
                           "tau_tail_ss_lo": float(ss["tau_tail_ms"].quantile(0.25)),
                           "tau_tail_ss_hi": float(ss["tau_tail_ms"].quantile(0.75)),
                           "onset_tau_s": onset_tau,
                           "onset_prof_lo": onset_lo, "onset_prof_hi": onset_hi,
                           "block_frac": 1 - float(ss["A_peak"].median()) / ref["A_peak"] if ref["A_peak"] > 0 else np.nan})
        e4031_resid = float(e4031["A_peak"].median()) / ref["A_peak"] if len(e4031) and ref["A_peak"] > 0 else np.nan
        # 逐 sweep 留存（供统一总表行级）：只留 τ_tail 有限行
        sweeps = tr_df[np.isfinite(tr_df["tau_tail_ms"])][
            ["TRACENUM", "ELTIME_s", "LIQUID", "CONC_uM", "tau_tail_ms", "tau_prof_lo", "tau_prof_hi", "tail_r2", "A_peak"]].to_dict("records")
        return {"ok": True, "lab": lab, "drug": drug, "blind": blind, "cell": cell,
                "n_traces": int(len(tr_df)), "qc_pass": bool(qc_pass), "control_drift": float(drift),
                "ref": ref, "blocks": blocks, "e4031_resid": e4031_resid, "sweeps": sweeps}
    except Exception as e:
        return {"ok": False, "cell": zpath, "err": f"{type(e).__name__}: {e}"}

def collect_tasks():
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
            sub = rw[rw["CELLID"] == cell]
            if len(sub) == 0:
                sub = rw[rw["CELLID"].str.contains(cell.split(" ", 0)[0], na=False)]
            if len(sub):
                tasks.append((zp, sub[["TRACENUM", "ELTIME", "LIQUID", "CONC", "CTLFL"]].copy(), lab, drug, blind, factor))
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=20)
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
    if not todo:
        print("ALL DONE"); finalize(); return
    n0 = 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(proc_cell, todo, chunksize=2):
            with open(CKPT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n0 += 1
            if n0 % 20 == 0:
                print(f"  {n0}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("预算到点，自停（可续跑）", flush=True)
                pool.terminate(); pool.join()
                return
    print("ALL DONE", flush=True)
    finalize()

def finalize():
    """从 ckpt 汇总落盘两张表"""
    rows = []
    with open(CKPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r.get("ok"):
                    rows.append(r)
            except Exception:
                pass
    # 去重（续跑可能重复）：保最后一个
    seen = {}
    for r in rows:
        seen[r["cell"]] = r
    rows = list(seen.values())
    sw_rows = []; blk_rows = []
    for r in rows:
        for s in r.get("sweeps", []):
            sw_rows.append({"lab": r["lab"], "drug": r["drug"], "blind": r["blind"],
                            "cell": r["cell"], "qc_pass": r["qc_pass"], **s})
        for b in r["blocks"]:
            ref_tau = r["ref"]["tau_tail_ms"]
            blk = {"lab": r["lab"], "drug": r["drug"], "blind": r["blind"], "cell": r["cell"],
                   "qc_pass": r["qc_pass"], **b,
                   "ref_tau_tail": ref_tau,
                   "ref_tau_prof_lo": r["ref"]["tau_prof_lo"],
                   "ref_tau_prof_hi": r["ref"]["tau_prof_hi"],
                   "tau_ratio": b["tau_tail_ss"] / ref_tau if ref_tau and np.isfinite(ref_tau) else np.nan}
            blk_rows.append(blk)
    pd.DataFrame(sw_rows).to_csv(os.path.join(OUT, "cipa_profile_sweeps.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(blk_rows).to_csv(os.path.join(OUT, "cipa_profile_cells.csv"), index=False, encoding="utf-8-sig")
    print(f"落盘：sweeps {len(sw_rows)} 行，blocks {len(blk_rows)} 行，细胞 {len(rows)}")

if __name__ == "__main__":
    main()
