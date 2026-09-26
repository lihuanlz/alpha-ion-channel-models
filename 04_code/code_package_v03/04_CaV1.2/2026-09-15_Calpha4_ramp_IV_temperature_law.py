# -*- coding: utf-8 -*-
"""
Cα-4：CaV1.2 斜坡 I-V 温度律与反转电位载子判决（预注册 v1.2，2026-09-15）
提取: 逐 sweep 弦电导 G=(I(+45)−I(+30))/15（QC: V_pk≤+25 且 G>0），E_chord=45−I(+45)/G
      Ba²⁺ 零交叉在弦窗内→插值级；Ca²⁺→严格下界（GHK 偏低，冒烟 S4b 标定）
判线: 预注册 §三（判1 Q10(G)恒定 / 判2 E_chord载子差 / 判3 E_chord组内恒定）
用法: 正式跑  %runfile 本文件 --wdir
      冒烟    python 本文件 --smoke
"""
import os, sys, json, glob
import numpy as np

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα4_斜坡IV温度律_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα4_斜坡IV温度律.png")
SMOKE = "--smoke" in sys.argv

TRAMP = (0.605, 0.715)     # 斜坡窗（秒）
VPK_LO, VPK_HI = -10.0, 48.0   # V_pk 搜索范围（文件 mV）
VPK_GATE = 25.0            # QC1：峰须低于弦窗底
CHORD_V = (30.0, 45.0)     # 弦窗（文件 mV）
MIN_SW = 20
MIN_BINS = 4
MIN_TSPAN = 3.0
MIN_PER_BIN = 3
EXCLUDE_PRIMARY = {"2021_06_30_0016.abf"}   # 钳位异常登记剔除（Cα-1 系列沿袭）

def sweep_chord(V0, I0):
    """单 sweep：V_pk 定位 + 弦电导。返回 dict 或 None。"""
    k = np.argsort(V0); Vs, Is = V0[k], I0[k]
    Ism = np.convolve(Is, np.ones(5) / 5, mode="same")
    vpk = float(Vs[2:-2][np.argmin(Ism[2:-2])])
    if vpk > VPK_GATE:
        return None
    def iat(vq):
        mm = (V0 >= vq - 1) & (V0 <= vq + 1)
        return float(np.median(I0[mm])) if mm.sum() >= 10 else None
    i30, i45 = iat(CHORD_V[0]), iat(CHORD_V[1])
    if i30 is None or i45 is None:
        return None
    G = (i45 - i30) / (CHORD_V[1] - CHORD_V[0])
    if G <= 0:
        return None
    return {"G": G, "Ec": CHORD_V[1] - i45 / G, "vpk": vpk}

def scan_file(path):
    import pyabf
    a = pyabf.ABF(path)
    recs = []
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
        m = (t >= TRAMP[0]) & (t < TRAMP[1]) & (v >= VPK_LO) & (v <= VPK_HI)
        if m.sum() < 300:
            continue
        r = sweep_chord(v[m], i[m])
        if r is None:
            continue
        r["T"] = float(bt.mean()); r["sweep"] = s
        recs.append(r)
    return recs

def cell_q10(recs):
    out = {"n_valid": len(recs)}
    if len(recs) < MIN_SW:
        out["pooled"] = False; out["why"] = "n<%d" % MIN_SW; return out
    Ts = np.array([r["T"] for r in recs]); Gs = np.array([r["G"] for r in recs])
    Es = np.array([r["Ec"] for r in recs])
    out["E_chord_med"] = float(np.median(Es))
    out["E_chord_iqr"] = float(np.percentile(Es, 75) - np.percentile(Es, 25))
    out["G_med"] = float(np.median(Gs))
    bins = np.floor(Ts); bx, by = [], []
    for b in sorted(set(bins)):
        mm = bins == b
        if mm.sum() >= MIN_PER_BIN:
            bx.append(b + 0.5); by.append(float(np.median(Gs[mm])))
    out["bins"] = len(bx); out["Tspan"] = float(max(bx) - min(bx) + 1) if bx else 0.0
    if len(bx) < MIN_BINS or out["Tspan"] < MIN_TSPAN:
        out["pooled"] = False; out["why"] = "bins/span"; return out
    bx = np.array(bx); ly = np.log10(np.array(by))
    sl, ic = np.polyfit(bx, ly, 1)
    pred = sl * bx + ic
    r2 = 1 - float(np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2))
    out.update({"pooled": True, "Q10": float(10 ** (10 * sl)), "q10_r2": r2,
                "bin_T": [float(x) for x in bx], "bin_G": [float(y) for y in by]})
    return out

def bootstrap_dE(ca, ba, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    ca = np.array(ca); ba = np.array(ba)
    ds = []
    for _ in range(n):
        a = rng.choice(ca, size=len(ca), replace=True)
        b = rng.choice(ba, size=len(ba), replace=True)
        ds.append(np.median(a) - np.median(b))
    ds = np.array(ds)
    return float(np.median(ca) - np.median(ba)), float(np.percentile(ds, 5)), float(np.percentile(ds, 95))

# ---------------------------------------------------------------- 冒烟
def synth_iv(V, Gmax, E_rev, vhalf, k, ghk_s=None):
    """m∞(V)·G·(V−E_rev)，可选近 E_rev GHK 平坦化（ghk_s=平坦化尺度 mV）。"""
    m = 1.0 / (1.0 + np.exp(-(V - vhalf) / k))
    g = Gmax * m
    if ghk_s is not None:
        g = g * (1.0 - np.exp(-(E_rev - V) / ghk_s))
    return g * (V - E_rev)

def smoke():
    print("=" * 72, flush=True)
    print(" [冒烟 Cα-4 v1.2] S1 弦回收 | S2 V_pk 门 | S3 零Q10 | S4 E_chord 边界性", flush=True)
    rng = np.random.default_rng(7)
    ok_all = True

    # S1：线性 I-V（E_rev=+42）弦 G 与 E_chord 回收
    recs = []
    for T in np.linspace(30, 39, 60):
        V = np.linspace(-10, 48, 1160)
        G_true = 40 * 1.4 ** ((T - 37) / 10)
        I = G_true * (V - 42.0) + rng.normal(0, 8, len(V))
        # 峰压到 +3 以下模拟 Ba 形
        m = 1.0 / (1.0 + np.exp(-(V - (-15)) / 8))
        r = sweep_chord(V, I * m)
        if r: r["T"] = T; recs.append(r)
    q = cell_q10(recs)
    e_bias = np.median([r["Ec"] for r in recs]) - 42.0
    g_rec = np.median([r["G"] for r in recs if abs(r["T"] - 37) < 0.6]) / 40.0 - 1
    ok = q.get("pooled") and abs(q["Q10"] / 1.4 - 1) <= 0.15 and abs(e_bias) <= 2 and abs(g_rec) <= 0.10
    ok_all &= ok
    print(f"  S1: Q10回收 {q.get('Q10', float('nan')):.2f}（真1.4，±15%） E_chord偏差 {e_bias:+.2f}mV（≤2） G误差 {g_rec*100:+.1f}%（≤10%）", "过" if ok else "挂", flush=True)

    # S2：V_pk 门——峰+19 通过、峰+30 剔除
    V = np.linspace(-10, 48, 1160)
    n_pass, n_block = 0, 0
    for _ in range(20):
        I1 = synth_iv(V, 40, 63, -5, 6) + rng.normal(0, 8, len(V))   # 峰≈+19
        I2 = synth_iv(V, 40, 63, 22, 6) + rng.normal(0, 8, len(V))   # 峰≈+30
        if sweep_chord(V, I1): n_pass += 1
        if sweep_chord(V, I2) is None: n_block += 1
    ok = n_pass >= 19 and n_block >= 19
    ok_all &= ok
    print(f"  S2: 峰+19 通过 {n_pass}/20（≥19） 峰+30 剔除 {n_block}/20（≥19）", "过" if ok else "挂", flush=True)

    # S3：零 Q10
    recs = []
    for T in np.linspace(30, 39, 60):
        V = np.linspace(-10, 48, 1160)
        I = 40.0 * (V - 42.0) / (1.0 + np.exp(-(V - (-15)) / 8)) + rng.normal(0, 8, len(V))
        r = sweep_chord(V, I)
        if r: r["T"] = T; recs.append(r)
    q3 = cell_q10(recs)
    ok = q3.get("pooled") and 0.9 <= q3["Q10"] <= 1.1
    ok_all &= ok
    print(f"  S3: 零Q10 回收 {q3.get('Q10', float('nan')):.2f}（∈[0.9,1.1]）", "过" if ok else "挂", flush=True)

    # S4a：Ba-like（E_rev=+43，激活半压 −8 → 峰≈+3，弦窗区 m∞ 饱和近线性）E_chord vs 上肢体线拟合截距
    diffs = []
    for _ in range(20):
        V = np.linspace(-10, 48, 1160)
        I = synth_iv(V, 40, 43, -8, 7) + rng.normal(0, 8, len(V))
        r = sweep_chord(V, I)
        if r is None: continue
        mm = (V >= r["vpk"] + 5) & (V <= 45)
        A = np.column_stack([np.ones(mm.sum()), V[mm]])
        (a0, b0), *_ = np.linalg.lstsq(A, I[mm], rcond=None)
        e_limb = -a0 / b0
        diffs.append(abs(r["Ec"] - e_limb))
    ok_a = len(diffs) >= 19 and float(np.median(diffs)) <= 3.0
    # S4b：Ca-like（E_rev=+63、弦窗内激活仍升 vhalf=+10 + 窗顶以上 GHK 平坦化）E_chord 经验下界 + 偏低幅度
    unders = []
    for _ in range(20):
        V = np.linspace(-10, 48, 1160)
        I = synth_iv(V, 40, 63, 10, 8, ghk_s=15.0) + rng.normal(0, 8, len(V))
        r = sweep_chord(V, I)
        if r: unders.append(63.0 - r["Ec"])
    unders = np.array(unders)
    ok_b = len(unders) >= 19 and float(np.median(unders)) > 2 and float(np.median(unders)) < 25
    ok = ok_a and ok_b
    ok_all &= ok
    print(f"  S4a: E_chord vs 体拟合差 中位 {np.median(diffs):.2f}mV（≤3）", "过" if ok_a else "挂", flush=True)
    print(f"  S4b: Ca-like E_chord 偏低幅度 中位 {np.median(unders):.1f}mV（2<x<25，经验下界成立）", "过" if ok_b else "挂", flush=True)

    print(" 冒烟总判:", "全过 -> 可正式跑" if ok_all else "未全过 -> 不开正式跑", flush=True)
    return ok_all

# ---------------------------------------------------------------- 正式
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {"Ca2": sorted(glob.glob(os.path.join(DATA, "Temperature_Ca2", "*.abf"))),
              "Ba2": sorted(glob.glob(os.path.join(DATA, "Temperature_Ba2", "*.abf")))}
    result = {"groups": {}, "verdict": {}}
    q10s, erevs, r2s = {"Ca2": [], "Ba2": []}, {"Ca2": [], "Ba2": []}, {"Ca2": [], "Ba2": []}
    print("=" * 72, flush=True)
    print(" Cα-4：CaV1.2 斜坡弦电导温度律与 E_chord 载子判决（预注册 v1.2 正式跑）", flush=True)
    print("=" * 72, flush=True)
    for grp, files in groups.items():
        print(f"\n[{grp} 组] {len(files)} 文件", flush=True)
        result["groups"][grp] = {}
        for p in files:
            name = os.path.basename(p)
            recs = scan_file(p)
            q = cell_q10(recs)
            result["groups"][grp][name] = q
            primary = name not in EXCLUDE_PRIMARY
            if q.get("pooled"):
                if primary:
                    q10s[grp].append(q["Q10"]); erevs[grp].append(q["E_chord_med"]); r2s[grp].append(q["q10_r2"])
                print(f"  {name}: 有效{q['n_valid']} | Q10(G)={q['Q10']:.2f} R²={q['q10_r2']:.2f} E_chord={q['E_chord_med']:+.1f}mV"
                      + ("" if primary else "（登记剔除）"), flush=True)
            else:
                print(f"  {name}: 有效{q['n_valid']} | 未入池（{q.get('why','?')}）", flush=True)

    vd = {}
    for grp in ("Ca2", "Ba2"):
        arr = np.array(q10s[grp]); er = np.array(erevs[grp])
        vd[grp] = {"n_pooled": len(arr),
                   "Q10_med": float(np.median(arr)) if len(arr) else None,
                   "Q10_cv": float(np.std(arr) / np.mean(arr)) if len(arr) > 1 else None,
                   "Q10_maxmin": float(np.max(arr) / np.min(arr)) if len(arr) > 1 else None,
                   "Q10_r2_ge05": float(np.mean([x >= 0.5 for x in r2s[grp]])) if r2s[grp] else None,
                   "E_chord_med": float(np.median(er)) if len(er) else None,
                   "E_chord_sd": float(np.std(er)) if len(er) > 1 else None,
                   "E_chord_range": float(np.max(er) - np.min(er)) if len(er) > 1 else None}
    # 判1
    j1 = {}
    for grp in ("Ca2", "Ba2"):
        v = vd[grp]
        j1[grp] = bool(v["n_pooled"] >= 3 and v["Q10_cv"] is not None and v["Q10_cv"] < 0.3
                       and v["Q10_maxmin"] < 2.0 and v["Q10_r2_ge05"] == 1.0)
    # 判2
    if len(erevs["Ca2"]) >= 2 and len(erevs["Ba2"]) >= 2:
        d, lo, hi = bootstrap_dE(erevs["Ca2"], erevs["Ba2"])
        vd["判2"] = {"dE_med": d, "ci90": [lo, hi], "pass": bool(d > 0 and lo > 0)}
    else:
        vd["判2"] = {"pass": False, "why": "细胞数不足"}
    # 判3
    vd["判3"] = {grp: bool(vd[grp]["E_chord_sd"] is not None and vd[grp]["E_chord_sd"] < 5
                           and vd[grp]["E_chord_range"] < 12) for grp in ("Ca2", "Ba2")}
    vd["判1"] = j1
    result["verdict"] = vd

    print("\n" + "=" * 72, flush=True)
    print(" 判词组件", flush=True)
    print(json.dumps(vd, ensure_ascii=False, indent=2), flush=True)

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for grp, c in (("Ca2", "tab:red"), ("Ba2", "tab:blue")):
        for name, q in result["groups"][grp].items():
            if q.get("pooled"):
                axes[0].plot(q["bin_T"], q["bin_G"], "o-", alpha=0.7, color=c,
                             label=f"{grp} {name}" if grp == "Ca2" else None)
    axes[0].set_xlabel("bath T (°C)"); axes[0].set_ylabel("chord G (pA/mV)")
    axes[0].set_title("弦电导 × 浴温（红=Ca²⁺ 蓝=Ba²⁺）")
    for i, grp in enumerate(("Ca2", "Ba2")):
        axes[1].plot(np.full(len(q10s[grp]), i) + np.linspace(-0.15, 0.15, max(1, len(q10s[grp]))),
                     q10s[grp], "o", color=("tab:red", "tab:blue")[i])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Ca²⁺", "Ba²⁺"])
    axes[1].set_ylabel("Q10(G)"); axes[1].set_title("电导温度律逐细胞")
    for i, grp in enumerate(("Ca2", "Ba2")):
        axes[2].plot(np.full(len(erevs[grp]), i) + np.linspace(-0.15, 0.15, max(1, len(erevs[grp]))),
                     erevs[grp], "o", color=("tab:red", "tab:blue")[i])
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["Ca²⁺（下界）", "Ba²⁺（插值级）"])
    axes[2].set_ylabel("E_chord (mV, 文件坐标)"); axes[2].set_title("反转电位逐细胞")
    fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
    print(f"\n 结果落盘: {OUTJ}\n 图落盘: {OUTP}", flush=True)

if __name__ == "__main__":
    if SMOKE:
        smoke()
    else:
        main()
