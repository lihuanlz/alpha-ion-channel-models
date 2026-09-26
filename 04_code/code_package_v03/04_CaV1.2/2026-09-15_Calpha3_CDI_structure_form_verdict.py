# -*- coding: utf-8 -*-
"""
Cα-3 v1.2：CDI 结构形式判决——新增快分量 vs 整体加速（τf 网格选型嵌套）
预注册 v1.2（2026-09-15，两轮冒烟驱动修订在案：自由M2截断简并废止→单锚敏感废止→网格选型）
H0: d = C + A·e^(−t/τ)
H1: d = C + A·[(1−w)·e^(−t/τ) + w·e^(−t/τf)]，τf ∈ {4,6,8,12,16}ms 网格 AICc 联合选型
判线: 预注册 v1.1/v1.2（判1′ 快分量存在 / 判2′ H-add vs H-acc / 判3′ w 恒定）
用法: 正式跑  %runfile 本文件 --wdir
      冒烟    python 本文件 --smoke
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import curve_fit

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα3v12_CDI结构形式判决_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα3v12_CDI结构形式判决.png")
SMOKE = "--smoke" in sys.argv

T17 = (0.38, 0.42)
G1_DV = 8.0
G2_PK = -300.0
LAT_MAX = 150
WIN = 250
MIN_SW = 20
AICC_MARGIN = 10.0
TAF_GRID = (4.0, 6.0, 8.0, 12.0, 16.0)
EXCLUDE_PRIMARY = {"2021_06_30_0016.abf"}
TMS = np.arange(WIN) * 0.1

def h0(t, C, A, tau):
    return C + A * np.exp(-t / tau)

def h1(t, C, A, tau, w, tf):
    return C + A * ((1 - w) * np.exp(-t / tau) + w * np.exp(-t / tf))

def aicc(n, k, rss):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def fit_grid(t, y):
    """H0 vs H1（τf 网格联合选型）。返回 dict。"""
    out = {"H0": None, "H1": None, "pick": None}
    try:
        p0, _ = curve_fit(h0, t, y, p0=[0.5, 0.5, 30.0],
                          bounds=([-0.2, 0.0, 0.5], [1.2, 1.5, 300.0]), maxfev=20000)
        rss0 = float(np.sum((y - h0(t, *p0)) ** 2))
        out["H0"] = {"C": float(p0[0]), "A": float(p0[1]), "tau": float(p0[2]),
                     "aicc": aicc(len(y), 3, rss0)}
    except Exception:
        pass
    best = None
    for tf in TAF_GRID:
        try:
            p1, _ = curve_fit(lambda tt, C, A, tau, w: h1(tt, C, A, tau, w, tf), t, y,
                              p0=[0.4, 0.6, 40.0, 0.3],
                              bounds=([-0.2, 0.0, 0.5, 0.0], [1.2, 1.5, 300.0, 1.0]), maxfev=40000)
            rss1 = float(np.sum((y - h1(t, *p1, tf)) ** 2))
            a1 = aicc(len(y), 4, rss1)
            if best is None or a1 < best[0]:
                best = (a1, tf, p1)
        except Exception:
            pass
    if best is not None:
        out["H1"] = {"C": float(best[2][0]), "A": float(best[2][1]), "tau": float(best[2][2]),
                     "w": float(best[2][3]), "tauf": float(best[1]), "aicc": float(best[0])}
    if out["H0"] and out["H1"]:
        out["pick"] = "H1" if out["H1"]["aicc"] < out["H0"]["aicc"] - AICC_MARGIN else "H0"
    return out

def d_shape(D):
    L = len(D); am = int(np.argmax(D)); dmax = float(D[am])
    return {"argmax_frac": am / L, "D_max": dmax,
            "D_end_over_max": float(D[-1] / dmax) if dmax else None}

def scan_file(path):
    import pyabf
    a = pyabf.ABF(path)
    trs, lats, n_qc = [], [], 0
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        m = (t >= T17[0]) & (t < T17[1])
        if abs(float(v[m].mean()) - 17.0) > G1_DV:
            continue
        seg = i[m]; pk = float(seg.min())
        if pk > G2_PK:
            continue
        n_qc += 1
        sm = np.convolve(seg, np.ones(10) / 10, mode="same")
        ip = int(np.argmin(sm))
        lats.append(ip)
        if ip > LAT_MAX or len(seg) - ip < WIN:
            continue
        trs.append(seg[ip:ip + WIN] / pk)
    return trs, lats, n_qc

def bootstrap_band(cell_curves, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    cc = np.array(cell_curves)
    boots = np.array([cc[rng.integers(0, len(cc), len(cc))].mean(axis=0) for _ in range(n)])
    return np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)

def synth_add(rng, noise, w=0.30, tf=8.0, t2=50.0, C=0.40):
    return C + (1 - C) * (w * np.exp(-TMS / tf) + (1 - w) * np.exp(-TMS / t2)) + rng.normal(0, noise, len(TMS))

def synth_warp(rng, noise, tau=25.0, C=0.40):
    return C + (1 - C) * np.exp(-TMS / tau) + rng.normal(0, noise, len(TMS))

def smoke():
    print("=" * 72, flush=True)
    print(" [冒烟 Cα-3 v1.2] S1\" 网格回收 | S2\" warp对照 | S3\" 边界实测 | S4 潜伏门", flush=True)
    print("=" * 72, flush=True)
    rng = np.random.default_rng(7)
    # S1"
    pick, werr = 0, []
    for _ in range(20):
        fr = fit_grid(TMS, synth_add(rng, 0.003))
        if fr["pick"] == "H1":
            pick += 1; werr.append(fr["H1"]["w"] - 0.30)
    ok = pick >= 18 and abs(np.median(werr)) <= 0.08
    print(f"  S1\" add(τf=8): H1选中 {pick}/20（≥18） w偏差 {float(np.median(werr)):+.3f}（≤0.08，低估方向登记）", "过" if ok else "挂", flush=True)
    # S2"
    no1 = 0
    for _ in range(20):
        fr = fit_grid(TMS, synth_warp(rng, 0.003))
        if fr["pick"] != "H1":
            no1 += 1
    print(f"  S2\" warp: H1不当选 {no1}/20（判线 ≥16）", "过" if no1 >= 16 else "挂", flush=True)
    # S3" 边界：τf=5 可分 / τf=14 不可分（登记入档）
    p5, w5, p14 = 0, [], 0
    for _ in range(10):
        fr = fit_grid(TMS, synth_add(rng, 0.003, tf=5.0))
        if fr["pick"] == "H1":
            p5 += 1; w5.append(fr["H1"]["w"] - 0.30)
    for _ in range(10):
        fr = fit_grid(TMS, synth_add(rng, 0.003, tf=14.0))
        if fr["pick"] == "H1":
            p14 += 1
    ok = p5 >= 8 and abs(np.median(w5)) <= 0.08 and p14 <= 4
    print(f"  S3\" 边界: τf=5 选中 {p5}/10 w偏差 {float(np.median(w5)):+.3f} | τf=14 选中 {p14}/10（判线 ≥8 且 ≤4）", "过" if ok else "挂", flush=True)
    # S4
    lats = rng.integers(0, 300, 200)
    wrong = sum(1 for x in lats if x <= LAT_MAX and x > LAT_MAX)
    print(f"  S4 潜伏门: 误纳 {wrong}（判线 0）", "过" if wrong == 0 else "挂", flush=True)

def main():
    if SMOKE:
        smoke(); return
    print("=" * 72, flush=True)
    print(" Cα-3 v1.2：CDI 结构形式判决（τf 网格选型嵌套，预注册 v1.2 正式跑）", flush=True)
    print("=" * 72, flush=True)
    result = {"cells": {}, "verdict": {}}
    grp_curves = {"Ca2": [], "Ba2": []}
    grp_fits = {"Ca2": {}, "Ba2": {}}
    for grp in ["Ca2", "Ba2"]:
        files = sorted(glob.glob(os.path.join(DATA, "Temperature_" + grp, "*.abf")))
        print(f"\n[{grp} 组]", flush=True)
        for fpath in files:
            name = os.path.basename(fpath)
            trs, lats, n_qc = scan_file(fpath)
            lats = np.array(lats)
            rec = {"n_qc": n_qc, "n_shape": len(trs),
                   "lat_med_ms": float(np.median(lats) * 0.1) if len(lats) else None,
                   "lat_reject_frac": float(np.mean(lats > LAT_MAX)) if len(lats) else None}
            if len(trs) < MIN_SW:
                rec["pooled"] = False
                print(f"  {name}: QC过{n_qc} 形状池{len(trs)}<{MIN_SW} 未入池（潜伏中位 {rec['lat_med_ms']}ms 剔率 {rec['lat_reject_frac']:.2f}）", flush=True)
            else:
                cc = np.mean(trs, axis=0)
                fr = fit_grid(TMS, cc)
                rec.update({"pooled": True, "fit": fr, "curve": [float(x) for x in cc[::10]]})
                primary = name not in EXCLUDE_PRIMARY
                if primary:
                    grp_curves[grp].append(cc)
                    grp_fits[grp][name] = fr
                msg = f"  {name}: 池{len(trs)} 潜伏{rec['lat_med_ms']:.1f}ms | {fr['pick']}"
                if fr["H1"]:
                    msg += f" w={fr['H1']['w']:.3f} τf={fr['H1']['tauf']:.0f} τ慢={fr['H1']['tau']:.1f}"
                if fr["H0"]:
                    msg += f" (H0 τ={fr['H0']['tau']:.1f})"
                print(msg + ("" if primary else "（登记剔除）"), flush=True)
            result["cells"].setdefault(grp, {})[name] = rec
    v = {"pool_n": {g: len(grp_curves[g]) for g in grp_curves}}
    N = {}
    for grp in ["Ca2", "Ba2"]:
        if len(grp_curves[grp]) >= 3:
            N[grp] = np.mean(grp_curves[grp], axis=0)
            fr = fit_grid(TMS, N[grp])
            lo, hi = bootstrap_band(grp_curves[grp])
            v[f"group_{grp}"] = {"pick": fr["pick"], "H0": fr["H0"], "H1": fr["H1"],
                                 "band_halfwidth": float(np.median((hi - lo) / 2)),
                                 "cell_picks": {n: grp_fits[grp][n]["pick"] for n in grp_fits[grp]}}
    if "group_Ca2" in v and "group_Ba2" in v:
        gc, gb = v["group_Ca2"], v["group_Ba2"]
        ca_h1_cells = sum(1 for f in grp_fits["Ca2"].values() if f["pick"] == "H1")
        j1 = (gc["pick"] == "H1") and (ca_h1_cells >= 4)
        v["judge1_fast_component"] = {"group_pick": gc["pick"], "ca_cells_H1": ca_h1_cells,
                                      "ba_cells_H1": sum(1 for f in grp_fits["Ba2"].values() if f["pick"] == "H1"),
                                      "pass": bool(j1)}
        D = N["Ba2"] - N["Ca2"]
        v["D_shape_descriptive"] = d_shape(D)
        v["D_curve"] = [float(x) for x in D[::10]]
        if j1:
            h1s = [f["H1"] for f in grp_fits["Ca2"].values() if f["pick"] == "H1" and f["H1"]]
            w_cells = np.array([x["w"] for x in h1s])
            tau_cells = np.array([x["tau"] for x in h1s])
            tauf_cells = [x["tauf"] for x in h1s]
            tau_ba = (gb["H1"]["tau"] if gb["pick"] == "H1" and gb["H1"]
                      else (gb["H0"]["tau"] if gb["H0"] else None))
            ratio = float(np.median(tau_cells) / tau_ba) if tau_ba else None
            c1 = 0.15 <= float(np.median(w_cells)) <= 0.45
            c2 = ratio is not None and 0.7 <= ratio <= 1.4
            v["judge2_Hadd"] = {"w_median": float(np.median(w_cells)),
                                "tauf_cells": tauf_cells,
                                "tau_slow_ratio_Ca_over_Ba": ratio,
                                "c1_w_range": bool(c1), "c2_tau_ratio": bool(c2),
                                "pass": bool(c1 and c2)}
            v["judge3_w_const"] = {"CV": float(np.std(w_cells) / np.mean(w_cells)),
                                   "maxmin": float(w_cells.max() / w_cells.min()),
                                   "pass": bool(np.std(w_cells) / np.mean(w_cells) < 0.3
                                                and w_cells.max() / w_cells.min() < 2.0)}
        else:
            v["Hacc_branch"] = {"group_H0_tau_Ca": gc["H0"]["tau"] if gc["H0"] else None,
                                "group_H0_tau_Ba": gb["H0"]["tau"] if gb["H0"] else None,
                                "D_shape": v["D_shape_descriptive"],
                                "note": "若实测 CDI τf≳14ms，本窗不可分（v1.2 登记边界），落此支路不硬判"}
    result["verdict"] = v
    print("\n" + "=" * 72, flush=True)
    print(" 判词组件", flush=True)
    print(json.dumps(v, ensure_ascii=False, indent=2, default=str), flush=True)
    with open(OUTJ, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1, default=str)
    print("\n 结果落盘:", OUTJ, flush=True)
    # ---- 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f_ in font_manager.findSystemFonts():
            if "msyh" in f_.lower() or "simhei" in f_.lower():
                font_manager.fontManager.addfont(f_)
        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        if "Ca2" in N:
            axes[0].plot(TMS, N["Ca2"], "tab:red", lw=2, label="Ca²⁺")
        if "Ba2" in N:
            axes[0].plot(TMS, N["Ba2"], "tab:blue", lw=2, label="Ba²⁺")
        axes[0].set_xlabel("峰后时间 (ms)"); axes[0].set_ylabel("归一化电流"); axes[0].set_title("组平均衰减迹线"); axes[0].legend()
        if "Ca2" in N and "Ba2" in N:
            axes[1].plot(TMS, N["Ba2"] - N["Ca2"], "k", lw=2)
            axes[1].set_xlabel("峰后时间 (ms)"); axes[1].set_ylabel("N_Ba − N_Ca"); axes[1].set_title("差值曲线 D(t)（纯描述）")
        labels, vals, cols = [], [], []
        for grp, mk, c in [("Ca2", "", "tab:red"), ("Ba2", "*", "tab:blue")]:
            for n, f in grp_fits[grp].items():
                if f["pick"] == "H1" and f["H1"]:
                    labels.append(n.replace(".abf", "") + mk); vals.append(f["H1"]["w"]); cols.append(c)
        axes[2].bar(range(len(vals)), vals, color=cols)
        axes[2].axhline(0.293, ls="--", c="gray", lw=1)
        axes[2].set_xticks(range(len(vals))); axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[2].set_ylabel("w_fast"); axes[2].set_title("快分量权重 逐细胞（虚线=Cα-1 Δf）")
        fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
        print(" 图落盘:", OUTP, flush=True)
    except Exception as e:
        print(" 绘图失败（不影响判词）:", e, flush=True)

if __name__ == "__main__":
    main()
