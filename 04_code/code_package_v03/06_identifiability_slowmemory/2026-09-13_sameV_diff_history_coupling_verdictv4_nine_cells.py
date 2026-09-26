# -*- coding: utf-8 -*-
# 2026-09-13_同压异史_耦合判决v4_九细胞.py
# v3 的教训：7/9 细胞尾巴振幅小，基线高估 + 噪声使 xs 反复过零，log 深刺把 A/B 拉爆，数字不可引用。
# v4 测量纪律（判线与 v2/v3 完全相同：A<30% 且 B<15%）：
#   1. 信号掩码：基线扣除后 xs < 2*噪声地板(0.072 nA) 的点不取对数
#   2. 窗口斜率 = Theil-Sen 稳健直线拟合 log(xs)~t（抗离群）；每窗有效点 <15 的尾巴弃用
# 只读数据，窗内除稳健直线外无模型拟合。
# 运行：python 本文件
import os
import json
import numpy as np
import scipy.io as sio
from scipy.stats import theilslopes
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
SKIP_MS = 5.0
WIN_S = 0.9
DS = 10
NOISE_NA = 0.036          # v1 实测噪声地板
MASK_NA = 2 * NOISE_NA    # 掩码阈值 2σ
LATE = (0.6, 0.9)
MID = (0.4, 0.6)
MIN_PTS = 15              # 每窗最少有效点数
AMP_MIN_NA = 0.15
CONV_TOL = 0.30
DRIFT_TOL = 0.15


def load(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/steady_activation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/steady_activation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    return V, I


def find_tails(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(V[s[0]], s[0], len(s)) for s in segs]
    tails = []
    for k, (v, s0, n) in enumerate(info):
        dur = n * DT
        if abs(v + 40.0) < 0.5 and dur < 2.0 and k > 0:
            hist_v = None
            for j in range(k - 1, -1, -1):
                if info[j][2] * DT > 2.0:
                    hist_v = float(info[j][0])
                    break
            tails.append(dict(start=s0, n=n, hist_v=hist_v))
    return tails


def win_slope(t, lx, win):
    """窗口内有效点的 Theil-Sen 斜率；点不够返回 None"""
    m = (t >= win[0]) & (t <= win[1]) & np.isfinite(lx)
    if m.sum() < MIN_PTS:
        return None, int(m.sum())
    s = theilslopes(lx[m], t[m])[0]
    return float(s), int(m.sum())


def judge_cell(cell, tails, I):
    rows = []
    for tl in tails:
        s0 = tl["start"] + int(SKIP_MS / 1000.0 / DT)
        n = int(WIN_S / DT)
        y = I[s0: s0 + n].astype(float)
        t = np.arange(n) * DT
        t, y = t[::DS], y[::DS]
        amp = float(np.max(np.abs(y)))
        if amp < AMP_MIN_NA or not np.all(np.isfinite(y)):
            continue
        base = float(np.median(y[int(0.94 * len(y)):]))   # 基线：最后 50ms 中位
        xs = y - base
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.where(xs >= MASK_NA, np.log(xs), np.nan)
        lam_mid, n_mid = win_slope(t, lx, MID)
        lam_late, n_late = win_slope(t, lx, LATE)
        usable = lam_mid is not None and lam_late is not None
        drift = abs(lam_mid - lam_late) / (abs(lam_late) + 1e-300) if usable else float("nan")
        rows.append(dict(hist_v=tl["hist_v"], amp=amp, t=t, lx=lx, base=base,
                         lam_mid=lam_mid, lam_late=lam_late,
                         n_mid=n_mid, n_late=n_late, usable=usable, drift=drift))
    good = [r for r in rows if r["usable"]]
    if len(good) < 3:
        return dict(cell=cell, rows=rows, conv=float("nan"), drift_med=float("nan"),
                    A=None, B=None, verdict=f"数据不足（可用尾巴 {len(good)} 条）")
    lams = np.array([r["lam_late"] for r in good])
    drifts = np.array([r["drift"] for r in good])
    conv = float((lams.max() - lams.min()) / abs(np.median(lams)))
    drift_med = float(np.median(drifts))
    A = bool(conv < CONV_TOL)
    B = bool(drift_med < DRIFT_TOL)
    verdict = "平庸（初态/乘性）" if (A and B) else "率耦合/记忆"
    return dict(cell=cell, rows=rows, conv=conv, drift_med=drift_med, A=A, B=B, verdict=verdict)


def main():
    print("=" * 64)
    print(" 同压异史耦合判决 v4 · 九细胞（掩码+稳健斜率，判线不变）")
    print(f" 判线: A 跨历史收敛 <{CONV_TOL:.0%} 且 B 时间漂移 <{DRIFT_TOL:.0%}   掩码 xs<{MASK_NA:.3f} nA")
    print("=" * 64, flush=True)

    allres = []
    for cell in CELLS:
        V, I = load(cell)
        if I is None:
            print(f"\n  细胞 {cell}: 数据文件缺失，跳过")
            allres.append(dict(cell=cell, rows=[], conv=float("nan"), drift_med=float("nan"),
                               A=None, B=None, verdict="数据缺失"))
            continue
        res = judge_cell(cell, find_tails(V), I)
        allres.append(res)
        good = [r for r in res["rows"] if r["usable"]]
        if res["A"] is None:
            print(f"\n  细胞 {cell}: {res['verdict']}")
        else:
            print(f"\n  细胞 {cell}: 可用尾巴 {len(good)} 条")
            for r in good:
                print(f"    历史 {r['hist_v']:+6.0f} mV: 中段 {r['lam_mid']:7.2f}/s  晚窗 {r['lam_late']:7.2f}/s  漂移 {r['drift']:5.1%}  (点 {r['n_mid']}/{r['n_late']})")
            print(f"    A 跨历史收敛 {res['conv']:7.1%} -> {'过' if res['A'] else '不过'}   "
                  f"B 时间平稳 {res['drift_med']:6.1%} -> {'过' if res['B'] else '不过'}   => {res['verdict']}")

    valid = [r for r in allres if r["A"] is not None]
    n_mem = sum(1 for r in valid if r["verdict"] == "率耦合/记忆")
    n_flat = sum(1 for r in valid if r["verdict"].startswith("平庸"))
    print("\n" + "=" * 64)
    print(" 九细胞汇总（v4 掩码+稳健斜率）")
    print("-" * 64)
    print(f"  {'细胞':<10} {'可用尾':>4} {'A收敛':>8} {'B漂移':>8}   判词")
    for r in allres:
        if r["A"] is None:
            print(f"  {r['cell']:<10} {sum(1 for x in r['rows'] if x['usable']):>4} {'--':>8} {'--':>8}   {r['verdict']}")
        else:
            print(f"  {r['cell']:<10} {sum(1 for x in r['rows'] if x['usable']):>4} {r['conv']:>7.0%} {r['drift_med']:>7.0%}   {r['verdict']}")
    print("-" * 64)
    print(f"  有效判决 {len(valid)} 细胞：率耦合/记忆 {n_mem} 个，平庸 {n_flat} 个")
    print("=" * 64)

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for ax, r in zip(axes.flat, allres):
        good = [x for x in r["rows"] if x["usable"]]
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(good), 2)))
        for k, row in enumerate(good):
            ax.plot(row["t"], row["lx"], ".", color=cmap[k], ms=2.5, label=f"{row['hist_v']:+.0f}")
            for win, lam in ((MID, row["lam_mid"]), (LATE, row["lam_late"])):
                m = (row["t"] >= win[0]) & (row["t"] <= win[1])
                tt = row["t"][m]
                if len(tt) and lam is not None:
                    # 用窗内中位高度画拟合直线，便于肉眼核对
                    mm = np.isfinite(row["lx"][m])
                    if mm.sum():
                        c0 = np.median(row["lx"][m][mm]) - lam * np.median(tt)
                        ax.plot(tt, lam * tt + c0, color=cmap[k], lw=1.2)
        ax.axvspan(*LATE, color="gray", alpha=0.15)
        color = "green" if r["verdict"].startswith("平庸") else ("red" if "率耦合" in r["verdict"] else "gray")
        ax.set_title(f"{r['cell']} · {r['verdict']}", fontsize=10, fontweight="bold", color=color)
        ax.set_ylim(-8, 0)
        ax.grid(alpha=0.3)
        if good:
            ax.legend(fontsize=7, loc="lower left")
    for ax in axes.flat[len(allres):]:
        ax.axis("off")
    fig.suptitle(f"同压异史 v4 · 九细胞（掩码 xs<{MASK_NA:.2f} nA + Theil-Sen）率耦合/记忆 {n_mem}/{len(valid)}",
                 fontsize=14, fontweight="bold")
    fpng = os.path.join(HERE, "2026-09-13_同压异史_耦合判决v4_九细胞.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    out = {
        "conv_tol": CONV_TOL, "drift_tol": DRIFT_TOL, "mask_nA": MASK_NA, "min_pts": MIN_PTS,
        "cells": [{
            "cell": r["cell"],
            "tails": [{"hist_mV": x["hist_v"], "amp_nA": x["amp"], "baseline_nA": x["base"],
                       "lam_mid_per_s": x["lam_mid"], "lam_late_per_s": x["lam_late"],
                       "n_pts_mid": x["n_mid"], "n_pts_late": x["n_late"],
                       "usable": x["usable"], "drift": x["drift"]} for x in r["rows"]],
            "conv_spread": r["conv"], "drift_median": r["drift_med"],
            "A_pass": r["A"], "B_pass": r["B"], "verdict": r["verdict"],
        } for r in allres],
        "n_memory": n_mem, "n_flat": n_flat, "n_valid": len(valid),
    }
    fjson = os.path.join(HERE, "2026-09-13_同压异史_耦合判决v4_九细胞_结果.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n  图落盘: {fpng}")
    print(f"  结果落盘: {fjson}")


if __name__ == "__main__":
    main()
