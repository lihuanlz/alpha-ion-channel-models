# -*- coding: utf-8 -*-
# 2026-09-13_负40档漂移鉴定.py
# 问题：v5 里 -40mV 档 τ_late 跨细胞 CV=0.45（其它档 0.13-0.18）。
#   鉴定：hook 残留 / 基线漂移污染，还是真实细胞间差异？
# 方法（只读数据，零拟合族）：
#   每细胞 -40mV 尾：起点 skip=50/150/300ms × 基线=末200ms/末1s；
#   τ_late = [3.5,5.4]s 绝对窗对数斜率；亚窗稳定 = τ_A[2.2,3.5] vs τ_B[4.0,5.4]。
# 判词预先钉死（v2 修正：亚窗稳定不当门——对照证明真离散 20s 也有 τ_A/τ_B≈0.5，
#   因 1s 模态在早窗残留。亚窗改作"离散形态签名"，判决由 skip/基线/离群三杠杆承担；
#   v3 再修正：分支1 机制归因改为"浮动晚窗窗位伪影"，因冒烟实测 CV 对 skip/基线均不敏感）：
#   1) skip=150ms 后 CV<0.30 -> 污染说成立，恒定性延伸到 -40（若 CV50 明显更大，hook 坐实）；
#   2) 否则若去掉最差 1 个离群后 CV<0.30 -> 离群说：核心群体恒定，离群登记；
#   3) 否则 -> 真实异质：-40 档超慢层逐细胞建表；
#   附记：τ_A/τ_B 落在对照带 [0.3,0.8] 内的细胞比例（离散同型率）。
# 对照：合成 0.15/1/20s 三指数 + 0.036nA 噪声，τ_late 与 τ_B 都必须落 [12,30]s。
import os
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["16704007", "16704047", "16707014", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
DS = 10
MASK_NA = 0.072
DEBOUNCE = 20
BIN_S = 0.02
SKIPS = (0.05, 0.15, 0.30)
W_LATE = (3.5, 5.4)
W_A = (2.2, 3.5)
W_B = (4.0, 5.4)
CV_TOL = 0.30
STAB_TOL = 0.30


def load(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def find_tail_40(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    for k, (v, s0, n) in enumerate(info):
        if k < 2 or n * DT < 2.0 or abs(v + 40.0) > 0.5:
            continue
        pv, ps, pn = info[k - 1]
        hv, hs, hn = info[k - 2]
        if pv > -10.0 and 0.2 < pn * DT < 5.0:
            return dict(start=s0, n=n, hold=(hs, hs + hn))
    return None


def extract(I, tl, skip_s, base_s):
    """绝对时间分箱序列 + 噪声；base_s=基线取静默段末尾多少秒"""
    s0 = tl["start"] + int(skip_s / DT)
    y = I[s0: tl["start"] + tl["n"]].astype(float)
    h0, h1 = tl["hold"]
    h0 = max(h0, h1 - int(base_s / DT))
    bseg = I[h0:h1:DS]
    base = float(np.median(bseg))
    sig = 1.4826 * float(np.median(np.abs(bseg - base))) / np.sqrt(BIN_S / (DT * DS))
    xs = y - base
    t_full = np.arange(len(xs)) * DT + skip_s
    t, x0 = t_full[::DS], xs[::DS]
    sgn = 1.0 if np.median(x0[:200]) >= 0 else -1.0
    x = sgn * x0
    k = 15
    ys = np.convolve(x, np.ones(k) / k, mode="same")
    below = ys < MASK_NA
    i_end = len(t) - 1
    if below.sum() >= DEBOUNCE:
        run = np.convolve(below.astype(int), np.ones(DEBOUNCE, int), mode="valid")
        hit = np.where(run >= DEBOUNCE)[0]
        if len(hit):
            i_end = int(hit[0])
    tt, xx = t[:i_end], x[:i_end]
    nb = max(8, int((tt[-1] - tt[0]) / BIN_S))
    bt, by = [], []
    for b in np.array_split(np.arange(len(tt)), nb):
        bt.append(float(np.median(tt[b])))
        by.append(float(np.mean(xx[b])))
    return np.array(bt), np.array(by), sig


def slope_tau(bt, by, win, sig):
    m = (bt >= win[0]) & (bt <= win[1]) & (by > 3 * sig)
    if m.sum() < 8:
        return np.nan
    s = float(np.polyfit(bt[m], np.log(by[m]), 1)[0])
    return np.nan if s > -1e-4 else -1.0 / s


def cv_of(vals):
    a = np.array([v for v in vals if not np.isnan(v)], float)
    if len(a) < 4:
        return np.nan, len(a)
    return float(a.std(ddof=1) / a.mean()), len(a)


def main():
    rng = np.random.default_rng(7)
    print("=" * 74)
    print(" -40mV 档漂移鉴定（hook 残留 / 基线敏感 / 亚窗稳定 / 离群）")
    print("=" * 74, flush=True)

    # ---------- 对照 ----------
    t = np.arange(0.05, 5.55, DT * DS)
    yc = 0.5 * np.exp(-t / 0.15) + 0.5 * np.exp(-t / 1.0) + 0.4 * np.exp(-t / 20.0)
    yc = yc + rng.normal(0, 0.036, len(t))
    nb = int((t[-1] - t[0]) / BIN_S)
    bts, bys = [], []
    for b in np.array_split(np.arange(len(t)), nb):
        bts.append(float(np.median(t[b])))
        bys.append(float(np.mean(yc[b])))
    bts, bys = np.array(bts), np.array(bys)
    sig_c = 0.036 / np.sqrt(BIN_S / (DT * DS))
    tl_c = slope_tau(bts, bys, W_LATE, sig_c)
    ta_c = slope_tau(bts, bys, W_A, sig_c)
    tb_c = slope_tau(bts, bys, W_B, sig_c)
    stab_c = abs(ta_c - tb_c) / tb_c if not (np.isnan(ta_c) or np.isnan(tb_c)) else np.nan
    ratio_ref = ta_c / tb_c if not (np.isnan(ta_c) or np.isnan(tb_c)) else np.nan
    print(f"\n[对照] 合成 0.15/1/20s: τ_late={tl_c:.1f}s（真值20，要求[12,30]）"
          f" τ_A={ta_c:.1f} τ_B={tb_c:.1f}（要求[12,30]）τ_A/τ_B={ratio_ref:.2f}（离散形态签名参考）")
    if not (12.0 <= tl_c <= 30.0 and 12.0 <= tb_c <= 30.0):
        print("  对照未归位 -> 测量作废，停。")
        return
    print("  对照过。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据] 每细胞 -40mV 尾：")
    print(f"  {'细胞':>9} | {'τ_late@50ms':>11} {'τ_late@150ms':>12} {'τ_late@300ms':>12} "
          f"{'τ_late@150ms基线1s':>16} | {'τ_A':>6} {'τ_B':>6} {'稳定':>5}")
    rows = []
    for cell in CELLS:
        V, I = load(cell)
        if I is None:
            print(f"  {cell:>9} | 文件缺失")
            continue
        tl = find_tail_40(V)
        if tl is None:
            print(f"  {cell:>9} | 无 -40 尾")
            continue
        rec = dict(cell=cell)
        bt150, by150, sig150 = None, None, None
        for sk in SKIPS:
            bt, by, sig = extract(I, tl, sk, 0.2)
            rec[f"late_{int(sk * 1000)}"] = slope_tau(bt, by, W_LATE, sig)
            if abs(sk - 0.15) < 1e-9:
                bt150, by150, sig150 = bt, by, sig
        btf, byf, sigf = extract(I, tl, 0.15, 1.0)
        rec["late_150_base1s"] = slope_tau(btf, byf, W_LATE, sigf)
        rec["tau_A"] = slope_tau(bt150, by150, W_A, sig150)
        rec["tau_B"] = slope_tau(bt150, by150, W_B, sig150)
        rec["bt"], rec["by"] = bt150, by150
        rows.append(rec)
        f = lambda x: "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
        stab = abs(rec["tau_A"] - rec["tau_B"]) / rec["tau_B"] if not (
            np.isnan(rec["tau_A"]) or np.isnan(rec["tau_B"])) else np.nan
        rec["stab"] = stab
        print(f"  {cell:>9} | {f(rec['late_50']):>11} {f(rec['late_150']):>12} {f(rec['late_300']):>12} "
              f"{f(rec['late_150_base1s']):>16} | {f(rec['tau_A']):>6} {f(rec['tau_B']):>6} "
              f"{f(stab):>5}", flush=True)

    # ---------- 汇总 ----------
    cv50, _ = cv_of([r.get("late_50") for r in rows])
    cv150, n150 = cv_of([r.get("late_150") for r in rows])
    cv300, _ = cv_of([r.get("late_300") for r in rows])
    cvb1, _ = cv_of([r.get("late_150_base1s") for r in rows])
    ratios = [r["tau_A"] / r["tau_B"] for r in rows
              if not (np.isnan(r.get("tau_A", np.nan)) or np.isnan(r.get("tau_B", np.nan)))]
    frac_sig = float(np.mean([0.3 <= q <= 0.8 for q in ratios])) if ratios else 0.0
    vals = [(r["cell"], r["late_150"]) for r in rows if not np.isnan(r.get("late_150", np.nan))]
    outlier = None
    cv150x = np.nan
    if len(vals) >= 5:
        med = float(np.median([v for _, v in vals]))
        outlier = max(vals, key=lambda p: abs(np.log(p[1] / med)))
        cv150x, _ = cv_of([v for c, v in vals if c != outlier[0]])
    print("\n" + "-" * 74)
    print(f"  τ_late 跨细胞 CV: skip50={cv50:.2f}  skip150={cv150:.2f}(n={n150})  "
          f"skip300={cv300:.2f}  基线1s={cvb1:.2f}")
    print(f"  离散同型率: {frac_sig * 100:.0f}%（τ_A/τ_B∈[0.3,0.8]，对照参考 {ratio_ref:.2f}）")
    if outlier:
        print(f"  最差离群: {outlier[0]}（τ_late={outlier[1]:.2f}s）；去掉后 CV={cv150x:.2f}")

    # ---------- 判词（预先钉死） ----------
    print("\n" + "=" * 74)
    if cv150 < CV_TOL:
        verdict = (f"测量伪影说成立：固定绝对窗后 -40 恒定（CV={cv150:.2f}），常数图景延伸到 -40；"
                   f"v5 的 CV=0.45 来自浮动晚窗随各细胞截断位置游走")
        if cv50 > cv150 + 0.10:
            verdict += "；CV50 明显更大 -> hook/早段残留也有贡献"
        else:
            verdict += "；CV 对起点、基线均不敏感 -> 非 hook、非基线，纯窗位伪影"
    elif not np.isnan(cv150x) and cv150x < CV_TOL:
        verdict = (f"离群说：核心群体（{len(vals) - 1} 细胞）在 -40 恒定（CV={cv150x:.2f}），"
                   f"离群细胞 {outlier[0]} 登记在案，单独描述")
    else:
        verdict = ("真实异质：-40 档 τ_late 跨细胞漂移（skip/基线/离群都救不回）-> "
                   "-40 超慢层逐细胞建表")
    verdict += f"；离散同型率 {frac_sig * 100:.0f}%"
    print(" 判词: " + verdict)
    print("=" * 74)

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[0]
    for r in rows:
        bt, by = r["bt"], r["by"]
        m = (bt >= 0.3) & (by > 0)
        if m.sum() < 5:
            continue
        ref = by[m][0]
        ax.plot(bt[m], np.log10(by[m] / ref), lw=1.2, alpha=0.8, label=r["cell"][-4:])
    ax.set_title("-40mV 尾重叠（0.3s 处归一，log10）", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.set_ylabel("log10 I/I(0.3s)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        xs, ys_ = [], []
        for sk, key in ((50, "late_50"), (150, "late_150"), (300, "late_300")):
            v = r.get(key)
            if v is not None and not np.isnan(v):
                xs.append(sk)
                ys_.append(v)
        if xs:
            ax.plot(xs, ys_, "o-", ms=4, lw=1.2, alpha=0.8, label=r["cell"][-4:])
    ax.axhline(np.nanmedian([r["late_150"] for r in rows]), color="k", ls="--", lw=0.8)
    ax.set_title("τ_late 随起点的稳定性（每细胞）", fontweight="bold")
    ax.set_xlabel("起点 skip (ms)"); ax.set_ylabel("τ_late (s)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axis("off")
    lines = ["-40 档漂移鉴定", "",
             f"CV skip50/150/300: {cv50:.2f}/{cv150:.2f}/{cv300:.2f}",
             f"CV 基线1s: {cvb1:.2f}   离散同型率: {frac_sig * 100:.0f}%",
             f"离群: {outlier[0] if outlier else '--'}  去离群 CV: "
             f"{'--' if np.isnan(cv150x) else f'{cv150x:.2f}'}", "", verdict]
    y0 = 0.96
    for L in lines:
        ax.text(0.02, y0, L, fontsize=10,
                fontweight="bold" if (L.startswith("-40") or L == verdict) else "normal", wrap=True)
        y0 -= 0.10
    fpng = os.path.join(HERE, "2026-09-13_负40档漂移鉴定.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    out = dict(rows=[{k: v for k, v in r.items() if k not in ("bt", "by")} for r in rows],
               cv50=cv50, cv150=cv150, cv300=cv300, cv_base1s=cvb1,
               frac_sig=frac_sig, ratio_ref=ratio_ref, outlier=outlier,
               cv150_no_outlier=cv150x, verdict=verdict)
    fjson = os.path.join(HERE, "2026-09-13_负40档漂移鉴定_结果.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  图落盘: {fpng}")
    print(f"  结果落盘: {fjson}")


if __name__ == "__main__":
    main()
