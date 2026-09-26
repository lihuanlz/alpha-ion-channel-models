# -*- coding: utf-8 -*-
"""
Kα药 · IKs 药物表分解判决（预注册 v1.0 · 2026-09-15）
臂1 Mef（18）/ 臂2 DIDS（26）判1-3；臂3 HMR（6）登记不判。
对照 = Kα-1 WT 池 8 细胞（跨批登记）。
冒烟：%runfile '.../2026-09-15_Kα药_IKs药物表分解判决.py' --wdir  （首行 SMOKE=True）
正式：SMOKE 改 False 后同一命令
"""
import json, os
import numpy as np
import pyabf
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon

SMOKE = False  # 正式跑（冒烟已绿：v1.1 tpk 修正生效，毛刺排除）

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
D = os.path.join(ROOT, "数据", "iks_chan_8226585")
POOL_CSV = os.path.join(ROOT, "α模型", "2026-09-15_Kα1_IKs激活去激活恒定性判决_逐文件表.csv")
INV = os.path.join(D, "inventory.json")
TAG = "冒烟" if SMOKE else "结果"
OUT_JSON = os.path.join(ROOT, "α模型", f"2026-09-15_Kα药_IKs药物表分解判决_{TAG}.json")
OUT_CSV = os.path.join(ROOT, "α模型", f"2026-09-15_Kα药_IKs药物表分解判决_逐文件表_{TAG}.csv")
OUT_PNG = os.path.join(ROOT, "α模型", f"2026-09-15_Kα药_IKs药物表分解判决_{TAG}.png")

GRID = np.arange(0.0, 4.0, 0.00005)   # 20 kHz × 4 s 公共波形网格
WT_INST = 0.302                        # 池 inst 中位（Kα-1）

# ---------------- 提取（Kα-1 同码口径） ----------------
def boltz(V, Vh, k):
    return 1.0 / (1.0 + np.exp(-(V - Vh) / k))

def r2_of(y, yhat):
    ss = np.sum((y - yhat) ** 2); st = np.sum((y - y.mean()) ** 2)
    return 1 - ss / st if st > 0 else np.nan

def analyze(path):
    """返回 dict(gv=(Vh,k,r2), inst, decay_frac, wf60, iss60) 或 None。"""
    try:
        a = pyabf.ABF(path)
    except Exception:
        return None
    dt = 1.0 / a.sampleRate
    Vs, tpks, insts, dfs = [], [], [], []
    wf60 = None; iss60 = None; best_dv = 1e9
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
        seg = yc[p2a:p2b]; tseg = yc[p3a:p3b]
        # 尾巴峰（v1.1：3–30 ms 均值窗，跳变沿毛刺为单采样点，必须排除）
        w0, w1 = int(0.003 / dt), int(0.03 / dt)
        tpk = float(np.mean(tseg[w0:w1])) if len(tseg) > w1 else np.nan
        # inst（V≥+40）
        iss0 = float(np.mean(seg[-int(0.1 / dt):])) if len(seg) > int(0.2 / dt) else np.nan
        if iss0 and iss0 > 0 and v_step >= 40:
            head = float(np.mean(seg[int(0.002 / dt):int(0.002 / dt) + int(0.05 / dt)]))
            insts.append(head / iss0)
        # decay_frac
        if len(tseg) > int(0.1 / dt):
            start = float(np.mean(tseg[int(0.005 / dt):int(0.025 / dt)]))
            end = float(np.mean(tseg[-int(0.05 / dt):]))
            if start > 0:
                dfs.append((start - end) / start)
        # +60 波形（I_ss 归一，重采样到公共网格）
        if iss0 and iss0 > 20 and abs(v_step - 60) < best_dv and abs(v_step - 60) <= 15:
            t = np.arange(len(seg)) * dt
            m = t <= 4.0
            wf60 = np.interp(GRID, t[m], seg[m] / iss0)
            iss60 = iss0; best_dv = abs(v_step - 60)
        Vs.append(v_step); tpks.append(tpk)
    if len(Vs) < 8:
        return None
    Vs = np.array(Vs); tpks = np.array(tpks, dtype=float)
    ok = ~np.isnan(tpks)
    gv = None
    if ok.sum() >= 8:
        vv = Vs[ok]; aa = tpks[ok] - np.nanpercentile(tpks[ok], 5)
        mx = aa.max()
        if mx > 0:
            aa = aa / mx
            try:
                popt, _ = curve_fit(boltz, vv, aa, p0=[25, 19],
                                    bounds=([-150, 2], [120, 80]), maxfev=10000)
                r2 = float(r2_of(aa, boltz(vv, *popt)))
                gv = (float(popt[0]), float(popt[1]), r2)
            except Exception:
                pass
    return {"gv": gv, "inst": float(np.median(insts)) if insts else np.nan,
            "decay_frac": float(np.median(dfs)) if dfs else np.nan,
            "wf60": wf60, "iss60": iss60, "n_sweeps": len(Vs)}

def eps(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

# ---------------- 主流程 ----------------
def main():
    print("=" * 72)
    print(" Kα药：IKs 药物表分解判决（预注册 v1.0）")
    print("=" * 72, flush=True)
    import csv
    pool_rows = [r for r in csv.DictReader(open(POOL_CSV, encoding="utf-8-sig")) if r["in_pool"] == "1"]
    inv = json.load(open(INV, encoding="utf-8"))
    mef = [x["file"] for x in inv if "EQ+MA" in str(x.get("protocol", ""))]
    dids = [x["file"] for x in inv if "DIDs" in str(x.get("protocol", ""))]
    hmr = [x["file"] for x in inv if "HMR subtraction" in x["file"]]
    print(f"[清单] WT池 {len(pool_rows)} | Mef {len(mef)} | DIDS {len(dids)} | HMR {len(hmr)}", flush=True)

    # ---- WT 池：共识波形 + Null-B + 池统计 ----
    print("\n[WT 池]", flush=True)
    pool = {}
    for r in pool_rows:
        fn = r["file"]
        res = analyze(os.path.join(D, fn))
        if res and res["wf60"] is not None and res["gv"] and res["gv"][2] >= 0.9:
            pool[fn] = res
            print(f"  {fn[:30]} V½={res['gv'][0]:.1f} k={res['gv'][1]:.1f} inst={res['inst']:.3f} df={res['decay_frac']:.3f}", flush=True)
        else:
            print(f"  {fn[:30]} 质量门剔除", flush=True)
    names = list(pool.keys())
    W = np.array([pool[n]["wf60"] for n in names])
    consensus = W.mean(axis=0)
    null_ex = []
    for i, n in enumerate(names):
        loo = np.delete(W, i, axis=0).mean(axis=0)
        null_ex.append(eps(W[i], loo))
    null_q95 = float(np.percentile(null_ex, 95))
    wt_vh = float(np.median([pool[n]["gv"][0] for n in names]))
    wt_k = float(np.median([pool[n]["gv"][1] for n in names]))
    wt_inst = float(np.nanmedian([pool[n]["inst"] for n in names]))
    print(f"  池: n={len(names)} V½中位={wt_vh:.2f} k中位={wt_k:.2f} inst中位={wt_inst:.3f} Null Q95={null_q95:.4f}", flush=True)

    # ---- 药物臂 ----
    res = {"smoke": SMOKE, "wt": {"n": len(names), "vh_med": wt_vh, "k_med": wt_k,
                                  "inst_med": wt_inst, "null_ex": null_ex, "null_q95": null_q95},
           "arms": {}, "hmr": {}}
    rows_csv = []
    todo = {"Mef": mef, "DIDS": dids}
    if SMOKE:
        todo = {"Mef": mef[:1], "DIDS": dids[:1]}
        print("\n[冒烟] 各臂 1 文件试跑（不判）", flush=True)
    for arm, files in todo.items():
        print(f"\n[臂 {arm}] {len(files)} 文件", flush=True)
        recs = []
        for fn in files:
            r = analyze(os.path.join(D, fn))
            if r is None or r["wf60"] is None or r["gv"] is None or r["gv"][2] < 0.9:
                print(f"  {fn[:30]} 剔除（质量门）", flush=True)
                recs.append({"file": fn, "excluded": True}); continue
            ex = eps(r["wf60"], consensus)
            rec = {"file": fn, "excluded": False, "eps_x": ex,
                   "vh": r["gv"][0], "k": r["gv"][1], "gv_r2": r["gv"][2],
                   "dvh": r["gv"][0] - wt_vh, "dk": r["gv"][1] - wt_k,
                   "inst": r["inst"], "dinst": r["inst"] - wt_inst,
                   "decay_frac": r["decay_frac"], "iss60": r["iss60"]}
            recs.append(rec)
            print(f"  {fn[:30]} εx={ex:.3f} ΔV½={rec['dvh']:+.1f} Δk={rec['dk']:+.1f} inst={r['inst']:.3f} df={r['decay_frac']:.3f}", flush=True)
            rows_csv.append({"arm": arm, **{k: rec[k] for k in
                             ("file", "eps_x", "vh", "k", "gv_r2", "dvh", "dk", "inst", "dinst", "decay_frac", "iss60")}})
        ok = [x for x in recs if not x["excluded"]]
        v = {"n_files": len(files), "n_ok": len(ok)}
        if len(ok) >= 3:
            exs = np.array([x["eps_x"] for x in ok])
            dvhs = np.array([x["dvh"] for x in ok]); dks = np.array([x["dk"] for x in ok])
            insts = np.array([x["dinst"] for x in ok]); dfsm = np.array([x["decay_frac"] for x in ok])
            v.update({
                "eps_x_med": float(np.median(exs)),
                "frac_over_null": float(np.mean(exs > null_q95)),
                "wilcoxon_p": float(wilcoxon(exs - null_q95).pvalue) if len(ok) >= 5 else None,
                "判1_过": bool(np.median(exs) > null_q95 and np.mean(exs > null_q95) >= 0.70
                            and (wilcoxon(exs - null_q95).pvalue < 0.01 if len(ok) >= 5 else True)),
                "dvh_med": float(np.median(dvhs)), "dk_med": float(np.median(dks)),
                "dinst_med": float(np.median(insts)), "df_med": float(np.median(dfsm)),
                "判2a_|ΔV½|>20": bool(abs(np.median(dvhs)) > 20),
                "判2b_|Δk|>10": bool(abs(np.median(dks)) > 10),
                "判2c_inst差>0.2": bool(np.median(insts) > 0.2),
                "判2d_df<0.10": bool(np.median(dfsm) < 0.10),
                "cells": ok})
            # 判 3 外部对拍
            if arm == "Mef":
                v["判3_ΔV½带(-105.7±20)"] = bool(-125.7 <= np.median(dvhs) <= -85.7)
                v["判3_k带[24.5,58.1]"] = bool(24.5 <= wt_k + np.median(dks) <= 58.1)
            else:
                v["判3_ΔV½带(-46.6±20)"] = bool(-66.6 <= np.median(dvhs) <= -26.6)
                v["判3_k带[15,40]"] = bool(15 <= wt_k + np.median(dks) <= 40)
        res["arms"][arm] = v

    # ---- HMR 登记 ----
    if not SMOKE:
        print("\n[臂 HMR] 登记不判", flush=True)
        for fn in hmr:
            r = analyze(os.path.join(D, fn))
            if r:
                res["hmr"][fn] = {"gv": r["gv"], "inst": r["inst"],
                                  "decay_frac": r["decay_frac"], "n_sweeps": r["n_sweeps"]}
                g = r["gv"]
                print(f"  {fn[:36]} V½={g[0]:.1f} k={g[1]:.1f} inst={r['inst']:.3f} df={r['decay_frac']:.3f}", flush=True)

    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("arm,file,eps_x,vh,k,gv_r2,dvh,dk,inst,dinst,decay_frac,iss60\n")
        for r in rows_csv:
            f.write(",".join(str(r[k]) for k in
                             ("arm", "file", "eps_x", "vh", "k", "gv_r2", "dvh", "dk",
                              "inst", "dinst", "decay_frac", "iss60")) + "\n")
    print(f"\n[落盘] {OUT_JSON}\n[落盘] {OUT_CSV}", flush=True)
    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(_sys.executable).parent.parent.parent))
        from daimon_runtime import setup_plot; setup_plot()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        ax[0].plot(GRID, consensus, "k", lw=1.5, label="WT 池共识")
        for arm, mk in (("Mef", "r"), ("DIDS", "b")):
            for x in res["arms"].get(arm, {}).get("cells", [])[:6]:
                r = analyze(os.path.join(D, x["file"]))
                ax[0].plot(GRID, r["wf60"], mk, lw=0.6, alpha=0.5)
        ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("I/I_ss"); ax[0].set_title("(a) +60 波形：红=Mef 蓝=DIDS 黑=WT")
        for j, key in enumerate(("eps_x", "dvh")):
            data = [res["arms"].get(a, {}).get("cells", []) for a in ("Mef", "DIDS")]
            vals = [[x[key] for x in dd] for dd in data]
            ax[j + 1].boxplot([v for v in vals if v], labels=[a for a, v in zip(("Mef", "DIDS"), vals) if v])
            ax[j + 1].set_title(f"(b{j + 1}) {key}")
        ax[1].axhline(res["wt"]["null_q95"], color="r", ls="--", lw=1)
        ax[1].set_ylabel("ε_x vs WT 共识"); ax[2].axhline(-105.7, color="r", ls=":", lw=1); ax[2].axhline(-46.6, color="b", ls=":", lw=1)
        ax[2].set_ylabel("ΔV½ (mV)")
        fig.tight_layout(); fig.savefig(OUT_PNG, bbox_inches="tight", dpi=130)
        print(f"[落盘] {OUT_PNG}", flush=True)
    except Exception as e:
        print(f"[图] 失败（不影响判词）: {e}", flush=True)

if __name__ == "__main__":
    main()
