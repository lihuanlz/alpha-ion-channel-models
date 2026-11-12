# -*- coding: utf-8 -*-
"""
Stage3 阻断时间进程挖掘：每细胞 wash-in 曲线 → 逐药结合速率 + 稳态完成度
动机：HESI BAA 每细胞录了 15-20 min 加药洗入全程，此前只用首尾两点。
      τ_on（以 sweep 序号为时间轴）= 阻断发展速率；完成度 = 末段/拟合平台。
      慢药未达稳态 → IC50 系统性高估 → 可能解释锚表离群与跨室偏移。
列过滤精确匹配 _current_（列污染教训，见方法学冻结档 1.1/2.1）。
输出：stage3_ckpt.jsonl（逐细胞）+ stage3_onset_metrics.csv
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
CKPT = os.path.join(OUT, "stage3_ckpt.jsonl")

def onset_model(s, binf, tau_on, s0):
    return binf * (1 - np.exp(-(s - s0) / tau_on)) * (s >= s0)

def onset_rundown(s, binf, tau_on, s0, r):
    return binf * (1 - np.exp(-(s - s0) / tau_on)) * (s >= s0) + r * s

def proc_cell(zpath):
    try:
        rel = os.path.relpath(zpath, CIPA); parts = rel.split(os.sep)
        lab, drug = parts[0], parts[2]
        blind = "blind" in rel.lower()
        cell = f"{lab}/{drug}/" + os.path.basename(zpath).replace(".csv.zip", "")
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("trace_#") and "_current_" in c]
        if len(cols) < 8:
            return {"ok": False, "cell": cell, "err": f"traces={len(cols)}"}
        uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(
            cols[0].rsplit("_current_", 1)[-1].lower())
        if uf is None:
            return {"ok": False, "cell": cell, "err": "单位未知"}
        # 逐 trace 幅度（自适应峰，剔极端伪迹）
        recs = []
        for c in cols:
            y = df[c].to_numpy(float) * uf
            if np.std(y) <= 5:
                continue
            if np.ptp(y) > 4000 or abs(np.max(y)) > 3000:
                continue
            tr = int(c.split("#")[1].split("_")[0])
            i0 = int(0.05 * len(y))
            recs.append((tr, float(y[i0:].max())))
        if len(recs) < 10:
            return {"ok": False, "cell": cell, "err": f"test_traces={len(recs)}"}
        recs.sort()
        s_all = np.array([r[0] for r in recs], float)
        a_all = np.array([r[1] for r in recs], float)
        n = len(a_all)
        Amax = float(np.percentile(a_all, 90))
        if Amax <= 50:
            return {"ok": False, "cell": cell, "err": f"Amax={Amax:.0f}pA"}
        # E-4031 段锚定（同 stage2 口径）
        head_level = float(np.median(a_all[:max(3, n // 3)]))
        tail_low = a_all < 0.2 * head_level
        i_e4 = n
        for i in range(n - 1, -1, -1):
            if tail_low[i]:
                i_e4 = i
            else:
                break
        e4_found = (n - i_e4) >= 3
        # vehicle 段：开头连续 >0.55×Amax
        i_veh_end = -1
        for i in range(min(n // 3 + 1, n)):
            if a_all[i] > 0.55 * Amax:
                i_veh_end = i
            else:
                break
        if i_veh_end < 2:
            return {"ok": False, "cell": cell, "err": "veh 段太短"}
        A_ref = float(np.median(a_all[:i_veh_end + 1]))
        # 加药段 = vehicle 之后 → E-4031 之前
        i_drug0 = i_veh_end + 1
        i_drug_end = i_e4 if e4_found else n
        if (i_drug_end - i_drug0) < 6:
            return {"ok": False, "cell": cell, "err": "加药段<6点"}
        s_d = s_all[i_drug0:i_drug_end] - s_all[i_drug0]
        b_d = 1 - a_all[i_drug0:i_drug_end] / A_ref
        # 加药段：指数 onset + 线性 rundown 联合拟合（rundown 不剔除则 τ_on 被漂移吃掉）
        try:
            p0 = [0.6, max(3.0, 0.2 * s_d[-1]), 0.0, 0.0]
            popt, _ = curve_fit(onset_rundown, s_d, b_d, p0=p0, maxfev=6000,
                                bounds=([0.0, 0.5, -1e9, -0.02], [1.05, 3 * max(s_d[-1], 1), 1e9, 0.02]))
            pred = onset_rundown(s_d, *popt)
            ss_res = float(np.sum((b_d - pred) ** 2)); ss_tot = float(np.sum((b_d - b_d.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except Exception:
            return {"ok": False, "cell": cell, "err": "onset 拟合失败"}
        binf, tau_on, s0, r_rd = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
        hit_hi = abs(tau_on - 3 * s_d[-1]) < 0.02 * 3 * s_d[-1]
        tail5 = float(np.median(b_d[-5:]))
        completeness = tail5 / binf if binf > 0.05 else np.nan
        # 模型自由的末段斜率（平台化判据）：末 25% 线性斜率，单位 frac/sweep
        n_tail = max(4, len(b_d) // 4)
        sl = np.polyfit(s_d[-n_tail:], b_d[-n_tail:], 1)
        tail_slope = float(sl[0])
        # E-4031 段 onset（灌注时钟参照）
        tau_e4 = np.nan
        if e4_found:
            s_e = s_all[i_e4:] - s_all[i_e4]
            c_e = 1 - a_all[i_e4:] / max(float(np.median(a_all[max(0, i_e4 - 5):i_e4])), 1e-9)
            if len(s_e) >= 5 and c_e[-1] > 0.3:
                try:
                    pe, _ = curve_fit(onset_model, s_e, c_e, p0=[0.9, max(2.0, 0.2 * s_e[-1]), 0.0],
                                      maxfev=3000, bounds=([0.0, 0.3, -1e9], [1.2, 3 * max(s_e[-1], 1), 1e9]))
                    pre = onset_model(s_e, *pe)
                    r2e = 1 - float(np.sum((c_e - pre) ** 2)) / max(float(np.sum((c_e - c_e.mean()) ** 2)), 1e-12)
                    if r2e >= 0.5:
                        tau_e4 = float(pe[1])
                except Exception:
                    pass
        return {"ok": True, "cell": cell, "lab": lab, "drug": drug, "blind": blind,
                "n_traces": int(n), "e4_found": bool(e4_found),
                "A_ref_pA": A_ref, "n_veh": int(i_veh_end + 1),
                "drug_span_sweeps": float(s_d[-1]),
                "B_inf": binf, "tau_on_sweeps": tau_on, "s0": s0, "rundown_per_sweep": r_rd,
                "onset_r2": r2, "hit_hi": bool(hit_hi), "tau_e4031_sweeps": tau_e4,
                "tail_block": tail5, "completeness": completeness, "tail_slope": tail_slope,
                "qc_pass": bool(r2 >= 0.7 and not hit_hi and binf > 0.15 and tau_on > 0.5)}
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
                        done.add(r["cell"])
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
