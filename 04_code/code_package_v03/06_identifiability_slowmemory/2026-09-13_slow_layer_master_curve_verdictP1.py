# -*- coding: utf-8 -*-
# 2026-09-13_慢层主曲线判决P1.py   (P1.2：形状比 ρ*=t50/t25，免对齐免拟合)
# P1.0/P1.1 时间缩放对齐的结构性死亡：优化器钻重叠窗空子（r 越大窗越窄 rms 越小），
#   异形对照反而对齐更好。放弃对齐，换标度天然统计量：
#   ρ* = t50/t25（走完自身窗内落差 50% / 25% 的时刻比）
#   单指数恒=2（恒等式）；拉伸指数=2^(1/β)（β=0.5 -> 4.0）；时间缩放由构造消掉。
# 判线对照锚定：对照N 双指数对 ρ*≈2 vs 2；对照B 指数 vs 拉伸 = 2 vs 4。
#   可分性 ratioB > 1.3*ratioN；判线 TOL = sqrt(ratioN*ratioB)；对照挂则停。
# 诚实条款：窗内落差 D<0.35 log 的尾巴标"不可测"（超慢尾近似常数，不携带形状信息）。
# 检验A：细胞内跨电压 ρ* 极差比 <=TOL；检验B：同电压跨细胞同理。
# 运行：python 本文件
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
REF_CELL = "16704007"
DT = 1e-4
DS = 10
SKIP_MS = 5.0
NOISE_NA = 0.036
MASK_NA = 2 * NOISE_NA
DEBOUNCE = 20
BASE_MS = 200.0
SLOW_START_S = 0.5
WIN_MIN_S = 1.0
AMP0_MIN = 0.10
D_MIN = 0.35          # 窗内落差下限（log 单位）


def load(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None, "文件缺失"
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    diag = f"lenI={len(I)} lenV={len(V)} NaN={int(np.isnan(I).sum())}"
    n = min(len(I), len(V))
    return V[:n], I[:n], diag


def find_tails(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    tails = []
    for k, (v, s0, n) in enumerate(info):
        dur = n * DT
        if dur < 2.0 or k < 2:
            continue
        pv, ps, pn = info[k - 1]
        hv, hs, hn = info[k - 2]
        if pv > -10.0 and 0.2 < pn * DT < 5.0:
            b0 = hs + hn - int(BASE_MS / 1000.0 / DT)
            tails.append(dict(v=v, start=s0, n=n, base_idx=(b0, hs + hn), pre_v=pv))
    return tails


def slow_curve(I, tl):
    """分箱中位 log 曲线（P1.1 同款，已验证件）"""
    s0 = tl["start"] + int(SKIP_MS / 1000.0 / DT)
    y = I[s0: s0 + tl["n"]].astype(float)
    b0, b1 = tl["base_idx"]
    base = float(np.median(I[b0:b1]))
    xs0 = y - base
    t_full = np.arange(len(xs0)) * DT
    t, x0 = t_full[::DS], xs0[::DS]
    if len(t) < int(1.6 / (t[1] - t[0])):
        return None
    early = x0[:int(0.2 / (t[1] - t[0]))]
    sgn = 1.0 if np.median(early) >= 0 else -1.0
    x = sgn * x0
    dt_s = t[1] - t[0]
    i0 = int(SLOW_START_S / dt_s)
    k = max(3, int(15.0 / 1000.0 / dt_s) | 1)
    ys = np.convolve(x, np.ones(k) / k, mode="same")
    if ys[i0] < AMP0_MIN:
        return None
    below = ys[i0:] < MASK_NA
    i1 = len(t) - 1
    if below.sum() >= DEBOUNCE:
        run = np.convolve(below.astype(int), np.ones(DEBOUNCE, int), mode="valid")
        hit = np.where(run >= DEBOUNCE)[0]
        if len(hit):
            i1 = i0 + int(hit[0])
    win = (i1 - i0) * dt_s
    if win < WIN_MIN_S:
        return None
    tt, xx = t[i0:i1], x[i0:i1]
    nbin = int(np.clip(win / 0.05, 16, 100))
    sig_bin = 3 * NOISE_NA / np.sqrt(max(1, len(tt) // nbin))
    bt, bl = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        med = float(np.median(xx[b]))
        if med > max(sig_bin, 0.01):
            bt.append(float(np.median(tt[b])))
            bl.append(float(np.log(med)))
    if len(bt) < 10:
        return None
    return dict(t=np.array(bt), l=np.array(bl), v=tl["v"], base=float(base),
                amp=float(np.max(np.abs(x0))), win=float(win))


def shape_ratio(bt, bl):
    """ρ* = t50/t25（自身落差分数）。D 太浅返回 None。另返回 ρ38=t75/t25 作旁证"""
    D = bl[0] - bl[-1]
    if D < D_MIN:
        return None
    def tcross(frac):
        target = bl[0] - frac * D
        return float(np.interp(target, bl[::-1], bt[::-1]))
    t25, t50, t75 = tcross(0.25), tcross(0.50), tcross(0.75)
    if t25 <= 0:
        return None
    return dict(rho=t50 / t25, rho3=t75 / t25, D=float(D), t25=t25, t50=t50, t75=t75)


def synth_curve(t, y):
    i0 = int(SLOW_START_S / (t[1] - t[0]))
    tt, xx = t[i0:], y[i0:]
    nbin = int(np.clip((tt[-1] - tt[0]) / 0.05, 16, 100))
    bt, bl = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        med = float(np.median(xx[b]))
        if med > 0.01:
            bt.append(float(np.median(tt[b])))
            bl.append(float(np.log(med)))
    return np.array(bt), np.array(bl)


def controls(rng):
    t = np.arange(0.01, 5.5, DT * DS)
    def sr(y):
        return shape_ratio(*synth_curve(t, y))
    out = {}
    out["N1_exp_tau3"] = sr(0.6 * np.exp(-t / 3.0) + rng.normal(0, NOISE_NA, len(t)))
    out["N2_exp_tau8"] = sr(0.8 * np.exp(-t / 8.0) + rng.normal(0, NOISE_NA, len(t)))
    out["B1_exp_tau8"] = sr(0.8 * np.exp(-t / 8.0) + rng.normal(0, NOISE_NA, len(t)))
    out["B2_stretch_b0.5"] = sr(0.6 * np.exp(-(t / 3.0) ** 0.5) + rng.normal(0, NOISE_NA, len(t)))
    return out


def spread_of(rhos):
    r = [x for x in rhos if x is not None]
    if len(r) < 2:
        return None, r
    return max(r) / min(r), r


def main():
    print("=" * 66)
    print(" 慢层主曲线判决 P1.2 · 形状比 ρ*=t50/t25（免对齐免拟合）")
    print(" 单指数恒=2；拉伸β=0.5 -> 4.0；判线由对照锚定")
    print("=" * 66, flush=True)

    rng = np.random.default_rng(11)
    c = controls(rng)
    print("\n[对照]")
    for k, v in c.items():
        if v is None:
            print(f"  {k}: 窗内太浅，对照设计错误 -> 停")
            return
        print(f"  {k}: ρ*={v['rho']:.3f} (ρ38={v['rho3']:.3f}, D={v['D']:.2f})")
    ratioN = c["N1_exp_tau3"]["rho"] / c["N2_exp_tau8"]["rho"]
    ratioN = max(ratioN, 1 / ratioN)
    ratioB = c["B2_stretch_b0.5"]["rho"] / c["B1_exp_tau8"]["rho"]
    ratioB = max(ratioB, 1 / ratioB)
    print(f"  ratioN(同形)={ratioN:.3f}  ratioB(异形)={ratioB:.3f}")
    if not (ratioB > 1.3 * ratioN):
        print("\n  对照不可分（需要 ratioB > 1.3×ratioN）-> 统计量作废，停。")
        return
    TOL = float(np.sqrt(ratioN * ratioB))
    print(f"  对照过。判线 TOL = sqrt(ratioN×ratioB) = {TOL:.3f}")

    # ---------- 真实数据 ----------
    per_cell = {}
    for cell in CELLS:
        V, I, diag = load(cell)
        if I is None:
            print(f"\n  细胞 {cell}: {diag}")
            continue
        rows = []
        for tl in find_tails(V):
            sc = slow_curve(I, tl)
            if sc is None:
                continue
            sr = shape_ratio(sc["t"], sc["l"])
            if sr is None:
                rows.append(dict(v=sc["v"], shallow=True))
                continue
            rows.append(dict(v=sc["v"], shallow=False, win=sc["win"], **sr))
        per_cell[cell] = rows
        txt = " ".join(f"{r['v']:+.0f}:" + ("浅" if r["shallow"] else "%.2f" % r["rho"]) for r in rows)
        print(f"\n  细胞 {cell}: 曲线 {len(rows)} 条 [{txt}]  [{diag}]")

    # ---------- 检验 A ----------
    print("\n" + "-" * 66)
    print(f"[检验 A] 细胞内跨电压 ρ* 一致性（极差比 ≤ {TOL:.3f}）")
    resA = {}
    for cell, rows in per_cell.items():
        good = [r for r in rows if not r["shallow"]]
        if len(good) < 3:
            resA[cell] = dict(verdict="不足", n=len(good))
            print(f"  {cell}: 可测曲线不足（{len(good)} 条）")
            continue
        sp, rs = spread_of([r["rho"] for r in good])
        verdict = "过" if sp <= TOL else "不过"
        resA[cell] = dict(verdict=verdict, spread=float(sp),
                          rhos={f"{r['v']:+.0f}": round(r["rho"], 3) for r in good})
        txt = " ".join(f"{r['v']:+.0f}:{r['rho']:.2f}" for r in good)
        print(f"  {cell}: {verdict}  极差比={sp:.2f}  [{txt}]")

    # ---------- 检验 B ----------
    print("-" * 66)
    print(f"[检验 B] 同电压跨细胞 ρ* 一致性（极差比 ≤ {TOL:.3f}）")
    voltages = sorted({r["v"] for rows in per_cell.values() for r in rows if not r["shallow"]})
    resB = {}
    for vv in voltages:
        rs = {}
        for cell, rows in per_cell.items():
            r = next((x for x in rows if not x["shallow"] and abs(x["v"] - vv) < 0.5), None)
            if r:
                rs[cell] = r["rho"]
        if len(rs) < 3:
            continue
        sp = max(rs.values()) / min(rs.values())
        verdict = "过" if sp <= TOL else "不过"
        resB[vv] = dict(verdict=verdict, spread=float(sp), rhos={k: round(v, 3) for k, v in rs.items()})
        txt = " ".join(f"{k[-4:]}:{v:.2f}" for k, v in rs.items())
        print(f"  {vv:+.0f} mV: {verdict}  极差比={sp:.2f}  [{txt}]")

    va = [r for r in resA.values() if r["verdict"] in ("过", "不过")]
    vb = list(resB.values())
    nAp = sum(1 for r in va if r["verdict"] == "过")
    nBp = sum(1 for r in vb if r["verdict"] == "过")
    print("\n" + "=" * 66)
    print(" 总判词：")
    print(f"  检验A（跨电压）: {nAp}/{len(va)} 细胞过")
    print(f"  检验B（跨细胞）: {nBp}/{len(vb)} 电压档过")
    a_good = len(va) > 0 and nAp >= len(va) * 2 / 3
    b_good = len(vb) > 0 and nBp >= len(vb) * 2 / 3
    if a_good and b_good:
        final = "主曲线存在（ρ* 跨电压跨细胞一致）-> 慢层 = g(t/τ(V))，建模可行"
    elif nAp == 0 and nBp == 0:
        final = "ρ* 不一致 -> 无主曲线，无通用核，此方向封卷"
    else:
        final = "部分一致 -> 慢层有受限结构，按明细界定边界"
    print("  " + final)
    print("=" * 66)

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axes[0]
    names = list(c.keys())
    ax.bar([n[:10] for n in names], [c[n]["rho"] for n in names],
           color=["steelblue", "steelblue", "darkorange", "darkorange"])
    ax.axhline(2.0, color="gray", ls="--", label="单指数=2")
    ax.axhline(4.0, color="red", ls="--", label="拉伸β0.5=4")
    ax.set_title(f"对照（TOL={TOL:.2f}）", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.tick_params(axis='x', rotation=20)

    ax = axes[1]
    for cell, rows in per_cell.items():
        good = [r for r in rows if not r["shallow"]]
        if good:
            ax.plot([r["v"] for r in good], [r["rho"] for r in good], "o-", ms=4, lw=1,
                    alpha=0.7, label=cell[-4:])
    ax.axhline(TOL, color="red", ls="--", label=f"TOL={TOL:.2f}")
    ax.set_title("ρ*(电压) 每细胞", fontweight="bold")
    ax.set_xlabel("尾电压 mV"); ax.set_ylabel("ρ*")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axis("off")
    lines = ["P1.2 总判词", "",
             f"检验A 跨电压: {nAp}/{len(va)} 细胞过",
             f"检验B 跨细胞: {nBp}/{len(vb)} 电压档过", "", final]
    y0 = 0.92
    for L in lines:
        ax.text(0.03, y0, L, fontsize=11,
                fontweight="bold" if (L.startswith("P1") or L == final) else "normal", wrap=True)
        y0 -= 0.11
    fpng = os.path.join(HERE, "2026-09-13_慢层主曲线判决P1.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    out = dict(TOL=TOL, ratioN=float(ratioN), ratioB=float(ratioB),
               controls={k: v for k, v in c.items()},
               testA=resA, testB={str(k): v for k, v in resB.items()}, final=final)
    fjson = os.path.join(HERE, "2026-09-13_慢层主曲线判决P1_结果.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n  图落盘: {fpng}")
    print(f"  结果落盘: {fjson}")


if __name__ == "__main__":
    main()
