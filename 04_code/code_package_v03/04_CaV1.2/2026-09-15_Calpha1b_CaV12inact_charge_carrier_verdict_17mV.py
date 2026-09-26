# -*- coding: utf-8 -*-
"""
Cα-1b：CaV1.2 失活电荷载子判决 · +17mV 段（预注册 v1.1，2026-09-15）
数据: 公开数据/Cav12_Ren2022_g3msb/Temperature_{Ca2,Ba2}/*.abf （11 细胞，非配对）
主指标: f_inact（40ms 内失活分数，无模型）+ τ_fast（40ms 窗单指数）
判线: 预注册 v1.1 判1b–判4b（跑前钉死）
用法: 正式跑  %runfile 本文件 --wdir
      冒烟    python 本文件 --smoke
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα1b_CaV12失活电荷载子判决_17mV段_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα1b_CaV12失活电荷载子判决_17mV段.png")
SMOKE = "--smoke" in sys.argv

DT = 1e-4
T17 = (0.38, 0.42)   # 主测段 +17 mV（40 ms）

# QC 门（预注册 v1.1）
G1_DV = 8.0          # |V-17| ≤ 8 mV
G2_PK = -300.0       # 峰 ≤ -300 pA
G4_R2 = 0.90         # τ_fast 拟合 R²（f_inact 不受此门限）
MIN_SW_CELL = 15     # 细胞入池最少有效 sweep
MIN_BINS = 4
MIN_TSPAN = 3.0
MIN_PER_BIN = 3
EXCLUDE_PRIMARY = {"2021_06_30_0016.abf"}   # 预注册登记：钳位存疑，主判剔除

def exp1(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def fit_fast(t, y):
    """40ms 窗自峰后单指数+平台。返回 tau/r2 或 None。"""
    i_pk = int(np.argmin(y))
    if i_pk >= len(y) - 50:
        return None
    tt = t[i_pk:] - t[i_pk]; yy = y[i_pk:]
    C0 = float(np.mean(yy[-100:]))
    try:
        p, _ = curve_fit(exp1, tt, yy, p0=[y[i_pk] - C0, 0.012, C0],
                         bounds=([-3e4, 1e-3, -1e4], [0, 0.2, 1e4]), maxfev=20000)
    except Exception:
        return None
    rss = float(np.sum((yy - exp1(tt, *p)) ** 2))
    r2 = 1 - rss / float(np.sum((yy - yy.mean()) ** 2))
    return {"tau": float(p[1]), "r2": float(r2)}

def scan_file(path):
    import pyabf
    a = pyabf.ABF(path)
    recs = []
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
        m = (t >= T17[0]) & (t < T17[1])
        vv = float(v[m].mean()); seg = i[m]; tt = t[m]
        rec = {"sweep": s, "T": float(bt.mean()), "V": vv}
        if abs(vv - 17.0) > G1_DV:
            rec["qc"] = "G1"; recs.append(rec); continue
        pk = float(seg.min())
        if pk > G2_PK:
            rec["qc"] = "G2"; recs.append(rec); continue
        i_end = float(np.mean(seg[-100:]))   # 末 10 ms
        rec["peak"] = pk
        rec["f_inact"] = 1.0 - i_end / pk
        fr = fit_fast(tt, seg)
        if fr and fr["r2"] >= G4_R2:
            rec["tau"] = fr["tau"]; rec["r2"] = fr["r2"]
        rec["qc"] = None
        recs.append(rec)
    return recs

def cell_stats(recs):
    ok = [r for r in recs if r["qc"] is None]
    out = {"n_valid": len(ok)}
    if len(ok) < MIN_SW_CELL:
        out["pooled"] = False; return out
    f = np.array([r["f_inact"] for r in ok])
    out.update({"pooled": True, "f_med": float(np.median(f)),
                "f_p10": float(np.percentile(f, 10)), "f_p90": float(np.percentile(f, 90))})
    # τ_fast 温度律
    tt = [r for r in ok if "tau" in r]
    out["n_tau"] = len(tt)
    if len(tt) >= MIN_SW_CELL:
        Ts = np.array([r["T"] for r in tt]); taus = np.array([r["tau"] for r in tt])
        bins = np.floor(Ts); bx, by = [], []
        for b in sorted(set(bins)):
            mm = bins == b
            if mm.sum() >= MIN_PER_BIN:
                bx.append(b + 0.5); by.append(float(np.median(taus[mm])))
        if len(bx) >= MIN_BINS and (max(bx) - min(bx) + 1) >= MIN_TSPAN:
            bx = np.array(bx); ly = np.log10(np.array(by) * 1e3)
            sl, ic = np.polyfit(bx, ly, 1)
            pred = sl * bx + ic
            r2 = 1 - float(np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2))
            out["Q10"] = float(10 ** (-10 * sl)); out["tau37_ms"] = float(10 ** (ic + sl * 37))
            out["q10_r2"] = r2; out["bin_T"] = [float(x) for x in bx]
            out["bin_tau_ms"] = [float(10 ** y) for y in ly]
    # f_inact 温度趋势（登记）
    Ts = np.array([r["T"] for r in ok])
    if Ts.max() - Ts.min() >= 2:
        sl, ic = np.polyfit(Ts, f, 1)
        out["f_vs_T_slope_perC"] = float(sl)
    return out

def bootstrap_df(ca, ba, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    ca = np.array(ca); ba = np.array(ba)
    ds = []
    for _ in range(n):
        a = rng.choice(ca, size=len(ca), replace=True)
        b = rng.choice(ba, size=len(ba), replace=True)
        ds.append(np.median(a) - np.median(b))
    ds = np.array(ds)
    return float(np.median(ca) - np.median(ba)), float(np.percentile(ds, 5)), float(np.percentile(ds, 95))

def bessel_lowpass(y, fc=2000.0, fs=10000.0, order=4):
    sos = signal.bessel(order, fc / (fs / 2), btype="low", output="sos", norm="mag")
    return signal.sosfilt(sos, y)

def smoke():
    print("=" * 72, flush=True)
    print(" [冒烟 Cα-1b] S1' 40ms窗 τ_fast 回收 | S4 f_inact 回收", flush=True)
    print("=" * 72, flush=True)
    rng = np.random.default_rng(7)
    for tau_true in [0.008, 0.015, 0.025]:
        errs = []
        for _ in range(20):
            t = np.arange(0, 0.04, DT)
            y = -2500 * np.exp(-t / tau_true) - 300 + rng.normal(0, 30, len(t))
            y = bessel_lowpass(y)
            fr = fit_fast(t, y)
            if fr and fr["r2"] >= G4_R2:
                errs.append(fr["tau"] / tau_true - 1)
        med = float(np.median(errs)) * 100 if errs else float("nan")
        print(f"  S1' τ={tau_true*1e3:.0f}ms: 回收中位误差 {med:+.1f}%（n={len(errs)}，判线 ≤15%）",
              "过" if errs and abs(med) <= 15 else "挂", flush=True)
    for f_true in [0.3, 0.6, 0.85]:
        errs = []
        for _ in range(20):
            t = np.arange(0, 0.04, DT)
            pk, base = -2500.0, -300.0
            # 使 1-end/pk = f_true：end = pk*(1-f_true)
            end = pk * (1 - f_true)
            tau = 0.04 / np.log((pk - base) / (end - base))
            # 阶跃前 20 ms 基线，滤波器进稳态后再阶跃（贴近真实协议）
            t_pre = np.arange(0, 0.02, DT); t = np.arange(0, 0.04, DT)
            y = np.concatenate([np.full(len(t_pre), base), (pk - base) * np.exp(-t / tau) + base])
            y = y + rng.normal(0, 30, len(y))
            y = bessel_lowpass(y)
            seg = y[len(t_pre):]
            pk_m = float(seg.min()); end_m = float(np.mean(seg[-100:]))
            errs.append((1 - end_m / pk_m) - f_true)
        bias = float(np.median(errs))
        print(f"  S4 f={f_true}: 回收偏差 {bias:+.3f}（判线 |bias|≤0.03）",
              "过" if abs(bias) <= 0.03 else "挂", flush=True)

def main():
    if SMOKE:
        smoke(); return
    print("=" * 72, flush=True)
    print(" Cα-1b：CaV1.2 失活电荷载子判决 · +17mV 段（预注册 v1.1 正式跑）", flush=True)
    print("=" * 72, flush=True)
    result = {"groups": {}, "verdict": {}}
    f_cells = {"Ca2": {}, "Ba2": {}}
    q10s = {"Ca2": [], "Ba2": []}
    for grp in ["Ca2", "Ba2"]:
        files = sorted(glob.glob(os.path.join(DATA, "Temperature_" + grp, "*.abf")))
        result["groups"][grp] = {}
        print(f"\n[{grp} 组] {len(files)} 文件", flush=True)
        for fpath in files:
            name = os.path.basename(fpath)
            recs = scan_file(fpath)
            cs = cell_stats(recs)
            cs["fail_G1"] = sum(1 for r in recs if r["qc"] == "G1")
            cs["fail_G2"] = sum(1 for r in recs if r["qc"] == "G2")
            cs["n_sweeps"] = len(recs)
            result["groups"][grp][name] = cs
            primary = name not in EXCLUDE_PRIMARY
            if cs.get("pooled"):
                if primary:
                    f_cells[grp][name] = cs["f_med"]
                if "Q10" in cs:
                    q10s[grp].append(cs["Q10"])
                tag = "" if primary else "（登记剔除）"
                q10s_ = f" Q10={cs['Q10']:.2f} τ37={cs['tau37_ms']:.1f}ms R²={cs['q10_r2']:.2f}" if "Q10" in cs else ""
                print(f"  {name}: 有效 {cs['n_valid']}/{cs['n_sweeps']} | f_inact={cs['f_med']:.3f} [{cs['f_p10']:.2f},{cs['f_p90']:.2f}]{q10s_}{tag}", flush=True)
            else:
                print(f"  {name}: 有效 {cs['n_valid']}/{cs['n_sweeps']} | 未入池（G1={cs['fail_G1']} G2={cs['fail_G2']}）", flush=True)
    # ---- 判决
    v = {}
    ca = list(f_cells["Ca2"].values()); ba = list(f_cells["Ba2"].values())
    v["n_cells"] = {"Ca2": len(ca), "Ba2": len(ba)}
    v["f_inact_cells"] = {"Ca2": {k: f_cells["Ca2"][k] for k in f_cells["Ca2"]},
                          "Ba2": {k: f_cells["Ba2"][k] for k in f_cells["Ba2"]}}
    if len(ca) >= 3 and len(ba) >= 3:
        df, lo, hi = bootstrap_df(ca, ba)
        ca_a, ba_a = np.array(ca), np.array(ba)
        v["judge1b_CDI"] = {"delta_f": df, "CI90": [lo, hi],
                            "ratio": float(np.median(ca_a) / np.median(ba_a)),
                            "pass": bool(df > 0.15 and lo > 0.05)}
        v["judge2b_const"] = {
            "Ca2": {"CV": float(np.std(ca_a) / np.mean(ca_a)), "maxmin": float(ca_a.max() / ca_a.min())},
            "Ba2": {"CV": float(np.std(ba_a) / np.mean(ba_a)), "maxmin": float(ba_a.max() / ba_a.min())},
            "pass": bool(np.std(ca_a) / np.mean(ca_a) < 0.3 and ca_a.max() / ca_a.min() < 2.0
                         and np.std(ba_a) / np.mean(ba_a) < 0.3 and ba_a.max() / ba_a.min() < 2.0)}
    allq = np.array(q10s["Ca2"] + q10s["Ba2"])
    if len(allq) >= 3:
        v["judge3b_Q10"] = {"n": len(allq), "per_group_n": {g: len(q10s[g]) for g in q10s},
                            "median": float(np.median(allq)), "CV": float(np.std(allq) / np.mean(allq)),
                            "maxmin": float(allq.max() / allq.min()),
                            "pass": bool(np.std(allq) / np.mean(allq) < 0.3 and allq.max() / allq.min() < 2.0)}
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        # f_inact 逐细胞
        labels, vals, cols = [], [], []
        for grp, c in [("Ca2", "tab:red"), ("Ba2", "tab:blue")]:
            for name, cs in result["groups"][grp].items():
                if cs.get("pooled"):
                    labels.append(name.replace(".abf", "") + ("*" if grp == "Ba2" else ""))
                    vals.append(cs["f_med"]); cols.append(c)
        axes[0].bar(range(len(vals)), vals, color=cols)
        axes[0].set_xticks(range(len(vals))); axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("f_inact"); axes[0].set_title("40ms 失活分数 逐细胞（* = Ba²⁺）")
        # τ(T) 分箱
        for grp, c in [("Ca2", "tab:red"), ("Ba2", "tab:blue")]:
            for name, cs in result["groups"][grp].items():
                if "bin_T" in cs:
                    axes[1].scatter(cs["bin_T"], cs["bin_tau_ms"], c=c, s=18, alpha=0.7)
        axes[1].set_xlabel("浴温 °C"); axes[1].set_ylabel("τ_fast (ms)")
        axes[1].set_title("τ_fast(T) 分箱"); axes[1].set_yscale("log")
        # 组分布箱线
        data_box = [list(f_cells["Ca2"].values()), list(f_cells["Ba2"].values())]
        axes[2].boxplot([d for d in data_box if d], labels=[n for n, d in zip(["Ca²⁺", "Ba²⁺"], data_box) if d])
        axes[2].set_ylabel("f_inact 细胞中位"); axes[2].set_title("组间对照（主判 1b）")
        fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
        print(" 图落盘:", OUTP, flush=True)
    except Exception as e:
        print(" 绘图失败（不影响判词）:", e, flush=True)

if __name__ == "__main__":
    main()
