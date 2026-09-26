# -*- coding: utf-8 -*-
"""
Cα-2：钙微域判决——失活的电流幅度相关性（预注册 v1.0，2026-09-15）
数据: 公开数据/Cav12_Ren2022_g3msb/Temperature_{Ca2,Ba2}/*.abf
主统计量: 每细胞 z 标准化二元回归 f_inact = a + b_A·|I_peak| + b_T·T 的 b_A
判线: 预注册 §四（判1 Ca²⁺幅度相关 / 判2 Ba²⁺无 / 判3 微域判决）
用法: 正式跑  %runfile 本文件 --wdir
      冒烟    python 本文件 --smoke
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα2_钙微域判决_电流失活相关性_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα2_钙微域判决_电流失活相关性.png")
SMOKE = "--smoke" in sys.argv

T17 = (0.38, 0.42)
G1_DV = 8.0
G2_PK = -300.0
MIN_SW = 15
COLL_R = 0.9          # 共线性门 |r(A,T)| > 0.9 剔除主判
EXCLUDE_PRIMARY = {"2021_06_30_0016.abf"}   # Cα-1 登记剔除，沿用

def scan_file(path):
    import pyabf
    a = pyabf.ABF(path)
    recs = []
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
        m = (t >= T17[0]) & (t < T17[1])
        vv = float(v[m].mean()); seg = i[m]
        if abs(vv - 17.0) > G1_DV:
            continue
        pk = float(seg.min())
        if pk > G2_PK:
            continue
        i_end = float(np.mean(seg[-100:]))
        recs.append({"f": 1.0 - i_end / pk, "A": -pk / 1000.0, "T": float(bt.mean())})
    return recs

def cell_bA(recs):
    """z 标准化二元回归，返回 b_A / b_T / r(A,T) / n。"""
    if len(recs) < MIN_SW:
        return {"pooled": False, "n": len(recs)}
    f = np.array([r["f"] for r in recs]); A = np.array([r["A"] for r in recs]); T = np.array([r["T"] for r in recs])
    r_at = float(np.corrcoef(A, T)[0, 1])
    def z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x * 0.0
    fz, Az, Tz = z(f), z(A), z(T)
    X = np.column_stack([np.ones_like(fz), Az, Tz])
    beta, *_ = np.linalg.lstsq(X, fz, rcond=None)
    resid = fz - X @ beta
    r2 = 1 - float(np.sum(resid ** 2) / np.sum((fz - fz.mean()) ** 2)) if fz.std() > 0 else 0.0
    return {"pooled": True, "n": len(recs), "b_A": float(beta[1]), "b_T": float(beta[2]),
            "r_AT": r_at, "r2": r2, "A_range": [float(A.min()), float(A.max())],
            "f_med": float(np.median(f))}

def bootstrap_dmedian(ca, ba, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    ca = np.array(ca); ba = np.array(ba)
    ds = []
    for _ in range(n):
        a = rng.choice(ca, size=len(ca), replace=True)
        b = rng.choice(ba, size=len(ba), replace=True)
        ds.append(np.median(a) - np.median(b))
    ds = np.array(ds)
    return float(np.median(ca) - np.median(ba)), float(np.percentile(ds, 5)), float(np.percentile(ds, 95))

def smoke():
    print("=" * 72, flush=True)
    print(" [冒烟 Cα-2] S1 b_A 回收 | S2 零假设 | S3 共线性门", flush=True)
    print("=" * 72, flush=True)
    rng = np.random.default_rng(7)
    # S1（噪声取 σ=√(1−0.3²−0.2²)，使 fz 的 SD=1，真值标准化 b_A 恰为 0.30）
    errs, signs = [], 0
    for _ in range(20):
        n = 200
        A = rng.uniform(1, 3, n); T = rng.uniform(32, 38, n)
        Az = (A - A.mean()) / A.std(); Tz = (T - T.mean()) / T.std()
        fz = 0.30 * Az - 0.20 * Tz + rng.normal(0, np.sqrt(0.87), n)
        recs = [{"f": fz[k], "A": A[k], "T": T[k]} for k in range(n)]
        r = cell_bA(recs)
        errs.append(r["b_A"] / 0.30 - 1); signs += 1 if r["b_A"] > 0 else 0
    med = float(np.median(errs)) * 100
    print(f"  S1 b_A=0.30: 回收中位误差 {med:+.1f}% 符号 {signs}/20（判线 ≤30% 且 20/20）",
          "过" if abs(med) <= 30 and signs == 20 else "挂", flush=True)
    # S2
    bas, fp = [], 0
    for _ in range(20):
        n = 200
        A = rng.uniform(1, 3, n); T = rng.uniform(32, 38, n)
        Tz = (T - T.mean()) / T.std()
        fz = -0.20 * Tz + rng.normal(0, 0.5, n)
        recs = [{"f": fz[k], "A": A[k], "T": T[k]} for k in range(n)]
        r = cell_bA(recs); bas.append(r["b_A"])
        fp += 1 if r["b_A"] > 0.15 else 0
    print(f"  S2 b_A=0: 中位 |b_A| {float(np.median(np.abs(bas))):.3f}（判线 <0.10）假阳性 {fp}/20（判线 ≤1）",
          "过" if np.median(np.abs(bas)) < 0.10 and fp <= 1 else "挂", flush=True)
    # S3
    flagged = 0
    for _ in range(20):
        n = 200
        A = rng.uniform(1, 3, n)
        T = 32 + 6 * (A - 1) / 2 + rng.normal(0, 0.32, n)  # r≈0.95
        fz = rng.normal(0, 1, n)
        recs = [{"f": fz[k], "A": A[k], "T": T[k]} for k in range(n)]
        r = cell_bA(recs)
        flagged += 1 if abs(r["r_AT"]) > COLL_R else 0
    print(f"  S3 r(A,T)≈0.95: 共线性门剔除 {flagged}/20（判线 20/20）", "过" if flagged == 20 else "挂", flush=True)

def main():
    if SMOKE:
        smoke(); return
    print("=" * 72, flush=True)
    print(" Cα-2：钙微域判决——失活的电流幅度相关性（预注册 v1.0 正式跑）", flush=True)
    print("=" * 72, flush=True)
    result = {"cells": {}, "verdict": {}}
    bA = {"Ca2": {}, "Ba2": {}}; bT = {"Ca2": {}, "Ba2": {}}
    excluded = []
    for grp in ["Ca2", "Ba2"]:
        files = sorted(glob.glob(os.path.join(DATA, "Temperature_" + grp, "*.abf")))
        print(f"\n[{grp} 组]", flush=True)
        for fpath in files:
            name = os.path.basename(fpath)
            recs = scan_file(fpath)
            cs = cell_bA(recs)
            primary = name not in EXCLUDE_PRIMARY
            coll = cs.get("pooled") and abs(cs["r_AT"]) > COLL_R
            cs["primary"] = primary; cs["coll_excluded"] = bool(coll)
            result["cells"].setdefault(grp, {})[name] = cs
            if not cs.get("pooled"):
                print(f"  {name}: 有效 {cs['n']} <{MIN_SW} 未入池", flush=True); continue
            tag = ""
            if not primary:
                tag = "（Cα-1 登记剔除）"
            elif coll:
                tag = "（共线性剔除主判）"; excluded.append(name)
            else:
                bA[grp][name] = cs["b_A"]; bT[grp][name] = cs["b_T"]
            print(f"  {name}: n={cs['n']} b_A={cs['b_A']:+.3f} b_T={cs['b_T']:+.3f} r(A,T)={cs['r_AT']:+.2f} R²={cs['r2']:.2f} A={cs['A_range'][0]:.1f}-{cs['A_range'][1]:.1f}nA {tag}", flush=True)
    # ---- 判决
    v = {"coll_excluded": excluded}
    ca = np.array(list(bA["Ca2"].values())); ba = np.array(list(bA["Ba2"].values()))
    v["b_A_cells"] = {g: bA[g] for g in bA}
    v["b_T_median"] = {g: float(np.median(list(bT[g].values()))) if bT[g] else None for g in bT}
    v["n"] = {"Ca2": len(ca), "Ba2": len(ba)}
    if len(ca) >= 3:
        w = stats.wilcoxon(ca, alternative="greater")
        v["judge1_Ca"] = {"median": float(np.median(ca)), "wilcoxon_p": float(w.pvalue),
                          "pass": bool(np.median(ca) > 0.15 and w.pvalue < 0.05)}
    if len(ba) >= 3:
        w = stats.wilcoxon(ba, alternative="greater")
        v["judge2_Ba"] = {"median": float(np.median(ba)), "wilcoxon_p": float(w.pvalue),
                          "pass": bool(np.median(ba) < 0.05 or w.pvalue >= 0.05)}
    if len(ca) >= 3 and len(ba) >= 3:
        dm, lo, hi = bootstrap_dmedian(ca, ba)
        j3 = v.get("judge1_Ca", {}).get("pass") and v.get("judge2_Ba", {}).get("pass") and lo > 0
        v["judge3_microdomain"] = {"dmedian": dm, "CI90": [lo, hi], "pass": bool(j3)}
        # 0016 敏感性
        if "2021_06_30_0016.abf" in result["cells"]["Ba2"]:
            cs16 = result["cells"]["Ba2"]["2021_06_30_0016.abf"]
            if cs16.get("pooled"):
                ba2 = np.append(ba, cs16["b_A"])
                dm2, lo2, hi2 = bootstrap_dmedian(ca, ba2)
                v["sens_0016"] = {"dmedian": dm2, "CI90": [lo2, hi2],
                                  "flip": bool(not (v["judge1_Ca"]["pass"] and lo2 > 0))}
    result["verdict"] = v
    print("\n" + "=" * 72, flush=True)
    print(" 判词组件", flush=True)
    print(json.dumps(v, ensure_ascii=False, indent=2), flush=True)
    with open(OUTJ, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1)
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
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        labels, vals, cols = [], [], []
        for grp, c in [("Ca2", "tab:red"), ("Ba2", "tab:blue")]:
            for name, cs in result["cells"][grp].items():
                if cs.get("pooled"):
                    labels.append(name.replace(".abf", "") + ("*" if grp == "Ba2" else ""))
                    vals.append(cs["b_A"]); cols.append(c if cs["primary"] and not cs["coll_excluded"] else "gray")
        axes[0].bar(range(len(vals)), vals, color=cols)
        axes[0].axhline(0.15, ls="--", c="k", lw=0.8); axes[0].axhline(0.05, ls=":", c="k", lw=0.8)
        axes[0].axhline(0, c="k", lw=0.5)
        axes[0].set_xticks(range(len(vals))); axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("b_A（标准化）"); axes[0].set_title("幅度偏回归系数 逐细胞（灰=剔除主判）")
        data_box = [list(bA["Ca2"].values()), list(bA["Ba2"].values())]
        axes[1].boxplot([d for d in data_box if d], labels=[n for n, d in zip(["Ca²⁺", "Ba²⁺"], data_box) if d])
        axes[1].axhline(0, c="k", lw=0.5)
        axes[1].set_ylabel("b_A"); axes[1].set_title("组间对照（判3 主判）")
        fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
        print(" 图落盘:", OUTP, flush=True)
    except Exception as e:
        print(" 绘图失败（不影响判词）:", e, flush=True)

if __name__ == "__main__":
    main()
