# -*- coding: utf-8 -*-
"""
2026-09-14 · α模型 · B6-H1 高压激活判决
============================================
B6 悬案：+40 envelope 说激活慢（m@100ms≈8%m∞，τ_act 131–290ms，延迟 50–150ms），
AP 短峰期要求 m≈29%m∞——3× 矛盾。H1 调和候选：激活在 +50/+60/+70 比 +40 快得多
（envelope 只有 0/+40 档，>+40 无 rebound 数据，envelope 路判不了）。

本卡换路：直接拟合去极化长阶跃起始段的慢上升电流——h 在 V≥+40 数 ms 内钉死在
h_ss（τ_h≈1.5ms），起始 10ms 后 I(t) ≈ G·h_ss·(V−E)·m(t)，是 m(t) 的直接画像。
  数据源：steady_activation 协议 +40/+60 档（5s 阶跃，每细胞一次）；
          deactivation 协议 +50 档（2s 阶跃 ×9 重复，逐细胞平均）。
  模型：I(t) = a·(1 − exp(−max(0, t−d)/τ)) + b0 + b1·t，前 10ms 弃（沿口尖刺+h 瞬态）。

判线（跑前钉死）：
  H1 支持：τ(+50)、τ(+60) 群体中位均 ≤ τ(+40) 中位 × 1/3，且逐细胞方向一致率 ≥ 7/9；
  H1 否决：任一档比值 > 1/2，或一致率 < 6/9；
  中间地带：登记"弱证据"，不动模型。
  对照（不过则全卡统计量作废，照实登记"数据不足"）：
  C1 合成 τ=150ms + 实测噪声 -> 恢复 τ 相对误差 <30%；
  C2 合成 τ=50ms vs 150ms 在本噪声地板下可区分（两样本 t 检验 p<0.05）。

纪律：本侧仅 ast.parse + SMOKE=1（16713003）；正式跑（九细胞）用户 Spyder：
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_B6_H1_高压激活判决.py' --wdir
输出：本脚本同目录 _结果.json/.png（冒烟带 _冒烟 后缀）。
"""

import os
import json
import time

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
DT = 1e-4
SMOKE = os.environ.get("SMOKE", "0") == "1"

CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
D11_DEAD = {"16704007", "16704047"}   # 激活协议记录失败（D11 在案）——结果逐细胞登记，不剔除

SKIP_S = 0.010          # 前 10ms 弃（沿口尖刺 + h 瞬态）
WIN_S = 0.500           # 拟合窗 500ms
T_GRID = np.arange(5.0, 401.0, 5.0) * 1e-3      # τ: 5–400ms
D_GRID = np.arange(0.0, 201.0, 5.0) * 1e-3      # 延迟 d: 0–200ms


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def fit_onset(t, y):
    """网格 (τ, d)，线性解 (a, b0, b1)。返回 dict 或 None。"""
    best = None
    for tau in T_GRID:
        for d in D_GRID:
            x = 1.0 - np.exp(-np.clip(t - d, 0.0, None) / tau)
            X = np.column_stack([x, np.ones(len(t)), t])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tau, d, sol)
    sse, tau, d, sol = best
    a, b0, b1 = float(sol[0]), float(sol[1]), float(sol[2])
    edge = bool(tau <= T_GRID[0] * 1.02 or tau >= T_GRID[-1] * 0.98 or d >= D_GRID[-1] * 0.98)
    return dict(tau_ms=tau * 1e3, d_ms=d * 1e3, a=a, b0=b0, b1=b1, edge=edge,
                sse=sse, n=len(y))


def onset_curve(I, s0, n):
    """从阶跃段提取拟合窗（弃前10ms，取500ms），去段内前10ms均值作基线粗校正。"""
    w0 = s0 + int(SKIP_S / DT)
    w1 = min(s0 + n, w0 + int(WIN_S / DT))
    if (w1 - w0) * DT < 0.3:
        return None, None
    t = np.arange(w1 - w0) * DT
    y = I[w0:w1].astype(float)
    return t, y


def noise_sigma(I, info):
    """逐细胞噪声地板：最长的 -80mV 基线段（≥0.5s）去线性趋势后的 std。"""
    best = None
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT >= 0.5:
            if best is None or n > best[2]:
                best = (s0, n, n)
    if best is None:
        return None
    s0, n, _ = best
    seg = I[s0 + int(0.05 / DT): s0 + n].astype(float)
    t = np.arange(len(seg)) * DT
    A = np.vstack([t, np.ones(len(t))]).T
    sol, *_ = np.linalg.lstsq(A, seg, rcond=None)
    return float(np.std(seg - A @ sol))


def synth_cell_curve(sigma, n_rep=9, tau_true=0.150, d_true=0.080, a_true=0.03, rng=None):
    """合成 +50 档等价物：9 条重复平均后的 onset 曲线（含漂移项零均值）。"""
    rng = rng or np.random.default_rng(7)
    t = np.arange(int(WIN_S / DT)) * DT
    x = a_true * (1.0 - np.exp(-np.clip(t - d_true, 0.0, None) / tau_true))
    acc = np.zeros(len(t))
    for _ in range(n_rep):
        acc += x + rng.normal(0, sigma, len(t)) + rng.normal(0, 0.002) * t
    return t, acc / n_rep


def main():
    t_start = time.time()
    print("=" * 74, flush=True)
    print(" α模型 · B6-H1 高压激活判决" + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"), flush=True)
    print(" H1：激活在 +50/+60 远快于 +40 —— steady_act(+40/+60) + deact(+50×9) 起始段直拟合", flush=True)
    print("=" * 74, flush=True)

    rng = np.random.default_rng(20260914)

    # ---------- 每细胞三档拟合 ----------
    cells_out = {}
    sigmas = {}
    for cell in CELLS:
        rec = {}
        # steady_activation: +40 / +60
        V, I = load_mat("steady_activation_protocol.mat", cell, "steady_activation")
        if I is not None:
            info = segments(V)
            sig = noise_sigma(I, info)
            sigmas[cell] = sig
            for vt in (40.0, 60.0):
                for v, s0, n in info:
                    if abs(v - vt) < 2 and n * DT > 2.0:
                        t, y = onset_curve(I, s0, n)
                        if t is not None:
                            r = fit_onset(t, y)
                            r["sigma"] = sig
                            r["snr_a_over_sigma"] = (abs(r["a"]) / sig) if sig else None
                            rec[f"steady_{vt:+.0f}"] = r
                        break
        # deactivation: +50 ×9 平均
        V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
        if I is not None:
            info = segments(V)
            if cell not in sigmas:
                sigmas[cell] = noise_sigma(I, info)
            reps = []
            for v, s0, n in info:
                if abs(v - 50.0) < 2 and n * DT > 1.5:
                    t, y = onset_curve(I, s0, n)
                    if t is not None and len(t) == int(WIN_S / DT):
                        reps.append(y)
            if reps:
                t = np.arange(int(WIN_S / DT)) * DT
                y = np.mean(reps, axis=0)
                r = fit_onset(t, y)
                r["n_rep"] = len(reps)
                r["sigma"] = sigmas.get(cell)
                rec["deact_+50"] = r
        cells_out[cell] = rec
        s40 = rec.get("steady_+40"); s50 = rec.get("deact_+50"); s60 = rec.get("steady_+60")
        line = f"  {cell}: "
        for tag, r in [("+40", s40), ("+50", s50), ("+60", s60)]:
            if r:
                line += (f"{tag}: τ={r['tau_ms']:7.1f}ms d={r['d_ms']:6.1f}ms"
                         f"{'[边]' if r['edge'] else ''}  ")
            else:
                line += f"{tag}: 无数据  "
        print(line, flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    sig_vals = [s for s in sigmas.values() if s]
    sig_ref = float(np.median(sig_vals)) if sig_vals else float("nan")
    if not np.isfinite(sig_ref):
        print("  [警告] 无有效噪声地板段，对照无法构建 -> 全卡登记数据不足", flush=True)
        ctrl_ok = False
        c1 = c2 = False
        c1_err, c2_p = [float("nan")], float("nan")
    else:
        sig_avg = sig_ref / 3.0                   # 9 重复平均后的等效噪声
        c1_err = []
        t_syn, y_syn = synth_cell_curve(sig_avg, tau_true=0.150, d_true=0.080)
        for rep in range(20):
            _, y1 = synth_cell_curve(sig_avg, tau_true=0.150, d_true=0.080,
                                     rng=np.random.default_rng(1000 + rep))
            r1 = fit_onset(t_syn, y1)
            c1_err.append(abs(r1["tau_ms"] - 150.0) / 150.0)
        c1 = float(np.median(c1_err)) < 0.30
        print(f"  C1 合成 τ=150ms ×20: 恢复误差中位 {np.median(c1_err) * 100:.1f}%（<30%）-> "
              f"{'过' if c1 else '不过'}", flush=True)
        from scipy.stats import ttest_ind
        taus_fast, taus_slow = [], []
        for rep in range(12):
            _, yf = synth_cell_curve(sig_avg, tau_true=0.050, d_true=0.020,
                                     rng=np.random.default_rng(2000 + rep))
            _, ys2 = synth_cell_curve(sig_avg, tau_true=0.150, d_true=0.080,
                                      rng=np.random.default_rng(3000 + rep))
            taus_fast.append(fit_onset(t_syn, yf)["tau_ms"])
            taus_slow.append(fit_onset(t_syn, ys2)["tau_ms"])
        c2_p = float(ttest_ind(taus_fast, taus_slow).pvalue)
        c2 = c2_p < 0.05
        print(f"  C2 τ=50ms vs 150ms 可区分: 恢复中位 {np.median(taus_fast):.0f} vs "
              f"{np.median(taus_slow):.0f}ms，t 检验 p={c2_p:.2e}（<0.05）-> "
              f"{'过' if c2 else '不过'}", flush=True)
        ctrl_ok = c1 and c2

    # ---------- 判词 ----------
    print("\n" + "=" * 74, flush=True)
    verdict = None
    if not ctrl_ok:
        print(" 对照未归位 -> 统计量作废，照实登记：本噪声地板下 H1 不可判（数据不足）。", flush=True)
    else:
        rows = []
        for cell, rec in cells_out.items():
            r40, r50, r60 = rec.get("steady_+40"), rec.get("deact_+50"), rec.get("steady_+60")
            if r40 and r50 and r60 and not (r40["edge"] or r50["edge"] or r60["edge"]):
                rows.append(dict(cell=cell,
                                 r50=r50["tau_ms"] / r40["tau_ms"],
                                 r60=r60["tau_ms"] / r40["tau_ms"]))
        n = len(rows)
        agree = sum(1 for r in rows if r["r50"] <= 1 / 3 and r["r60"] <= 1 / 3)
        med50 = float(np.median([r["r50"] for r in rows])) if rows else float("nan")
        med60 = float(np.median([r["r60"] for r in rows])) if rows else float("nan")
        print(f" 有效细胞 {n}/9（顶边剔除）；τ(+50)/τ(+40) 中位={med50:.3f}  "
              f"τ(+60)/τ(+40) 中位={med60:.3f}；双档 ≤1/3 一致率 {agree}/{n}", flush=True)
        if n >= 6:
            if med50 <= 1 / 3 and med60 <= 1 / 3 and agree >= 7 * n / 9:
                verdict = "H1 支持：+50/+60 激活显著快于 +40（≥3×），B6 矛盾向 H1 收敛"
            elif med50 > 1 / 2 or med60 > 1 / 2 or agree < 6 * n / 9:
                verdict = "H1 否决：+50/+60 激活不比 +40 快多少，3× 矛盾不随电压解开"
            else:
                verdict = "弱证据：介于两判线之间，登记不动模型"
        else:
            verdict = f"有效细胞不足（{n}<6）-> 数据不足，H1 登记未决"
        print(f" 判词：{verdict}", flush=True)
        if not SMOKE:
            print(" （正式口径：判词生效）", flush=True)
        else:
            print(" （冒烟：判线路径演练，非判词）", flush=True)
    print("=" * 74, flush=True)

    # ---------- 落盘 ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(HERE, f"2026-09-14_α模型_B6_H1_高压激活判决{tag}.json")
    out = {"meta": {"smoke": SMOKE, "cells": CELLS, "sigma_ref": sig_ref,
                    "runtime_s": time.time() - t_start},
           "cells": cells_out,
           "controls": {"C1_tau_recovery_median_err": float(np.median(c1_err)),
                        "C1_pass": c1, "C2_p": c2_p, "C2_pass": c2},
           "verdict": verdict}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\n 结果落盘: {fjson}", flush=True)

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    ncell = len(CELLS)
    fig, axes = plt.subplots(ncell, 1, figsize=(11, 2.6 * ncell), squeeze=False)
    for i, cell in enumerate(CELLS):
        ax = axes[i, 0]
        rec = cells_out.get(cell, {})
        for key, col, lab in [("steady_+40", "tab:blue", "+40"), ("deact_+50", "tab:orange", "+50(×9均)"),
                              ("steady_+60", "tab:red", "+60")]:
            # 重画原始曲线+拟合
            pass
        # 直接重载数据画三档
        V, I = load_mat("steady_activation_protocol.mat", cell, "steady_activation")
        drawn = set()
        if I is not None:
            for v, s0, n in segments(V):
                for vt, col in ((40.0, "tab:blue"), (60.0, "tab:red")):
                    if abs(v - vt) < 2 and n * DT > 2.0 and vt not in drawn:
                        t, y = onset_curve(I, s0, n)
                        if t is not None:
                            ax.plot(t * 1e3, y, color=col, lw=0.8, alpha=0.85,
                                    label=f"{vt:+.0f}mV 实测")
                            key = f"steady_{vt:+.0f}"
                            if key in rec:
                                r = rec[key]
                                tt = np.arange(len(y)) * DT
                                xf = r["a"] * (1 - np.exp(-np.clip(tt - r["d_ms"] * 1e-3, 0, None)
                                                          / (r["tau_ms"] * 1e-3))) + r["b0"] + r["b1"] * tt
                                ax.plot(tt * 1e3, xf, color=col, lw=1.6, ls="--",
                                        label=f"{vt:+.0f} 拟合 τ={r['tau_ms']:.0f}ms d={r['d_ms']:.0f}ms")
                            drawn.add(vt)
        V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
        if I is not None:
            reps = []
            for v, s0, n in segments(V):
                if abs(v - 50.0) < 2 and n * DT > 1.5:
                    t, y = onset_curve(I, s0, n)
                    if t is not None and len(t) == int(WIN_S / DT):
                        reps.append(y)
            if reps:
                t = np.arange(int(WIN_S / DT)) * DT
                y = np.mean(reps, axis=0)
                ax.plot(t * 1e3, y, color="tab:orange", lw=0.8, alpha=0.85, label="+50mV 实测(×9均)")
                if "deact_+50" in rec:
                    r = rec["deact_+50"]
                    xf = r["a"] * (1 - np.exp(-np.clip(t - r["d_ms"] * 1e-3, 0, None)
                                              / (r["tau_ms"] * 1e-3))) + r["b0"] + r["b1"] * t
                    ax.plot(t * 1e3, xf, color="tab:orange", lw=1.6, ls="--",
                            label=f"+50 拟合 τ={r['tau_ms']:.0f}ms d={r['d_ms']:.0f}ms")
        ax.set_title(f"细胞 {cell}" + ("（D11 记录失败，登记）" if cell in D11_DEAD else ""),
                     fontsize=9)
        ax.set_xlabel("阶跃后 t (ms)"); ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("B6-H1 高压激活判决：+40/+50/+60 起始段 m(t) 直拟合"
                 + ("（冒烟）" if SMOKE else ""), fontsize=12)
    fpng = os.path.join(HERE, f"2026-09-14_α模型_B6_H1_高压激活判决{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" 图落盘: {fpng}", flush=True)

    if SMOKE:
        print("\n[冒烟完] 正式跑指令（Spyder）：\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_B6_H1_高压激活判决.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
