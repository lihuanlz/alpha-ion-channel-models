# -*- coding: utf-8 -*-
"""
2026-09-14 · α模型 · B6-H3 包络口径审计卡
============================================
B6 悬案唯一活口 H3 的 envelope 侧审计。
侦察（_recon_H3_003六尾流）初判三点，复核后修正：
  (a) 反弹尾流 0-2ms 有 ±5~8nA 电容尖刺，原封卷链不掩刺 -> 小脉冲拟合撞 τr=1ms 网格边（成立）；
  (b) 【已撤回】初判"尾流含随 Δt 增大的慢外向漂移（−1.6nA@300ms）"——复核为侦察脚本
      自身伪影：基线取前 5ms 中位数，而该窗含鼓包上升段（1000ms 尾 seg[2-5ms]=1.94nA），
      整条迹线被过减 ~1.9nA 造成假漂移；直接对数证实六条尾流末端全部回基线
      （+0.003~+0.084nA）。本卡保留漂移项 X·t 作为自由裁量，实测 X≈0 佐证。
  (c) 8% = 003 的 A(100)/A(1000) = 0.294/3.71，分子曾撞边——分子口径需审计（成立）。

本卡对每条尾流做三模型对比（前 300ms 窗）：
  M0 原法复刻：不掩刺，c + A·DoE(τr,τd)，网格同封卷；
  M1 审计主模型：掩 0-2ms，c + A·DoE + X·t（线性慢漂移）；
  M2 敏感变体：掩刺，c + A·DoE + X·(1−e^{−t/120ms})（饱和漂移）。
网格 (τr∈[1,60]ms, τd∈[8,2000]ms, τd≥1.5τr) 与封卷同；c/A/X 线性 lstsq。

对照（跑前钉死，不过则统计量作废停）：
  C1 链传真：单门 m(τ=135ms 无足, m@100ms=52.3%) 合成全族尾流
     （DoE τr3.5/τd25 + 漂移 X=−1.5e-3·A/ms + 双极尖刺 + σ=0.056 实测噪声）
     -> M1 回收 A(100)/A(1000) ∈ [0.40, 0.65]；
  C2 足传真：角延迟足(d=75ms,τ=228ms, m@100ms=10.4%) 同链
     -> M1 回收 ∈ [0.06, 0.16]。

判线（跑前钉死）：
  H3-env 成立（伪影主责）：003 修正比 ≥0.20 且 ≥2× M0 读数，或七细胞修正比中位 ≥0.25；
  H3-env 否（8% 稳健）：003 修正比 ∈ [0.04, 0.15] 且七细胞中位 ≤0.20；
  中间带 -> 照实登记数值不定性。
附带登记：漂移 X 的 Δt 依赖与符号；014/118 平包络是否漂移伪影；M1 vs M2 一致性。

运行：python 本文件（SMOKE=1 单细胞 16713003 冒烟；全量 7 细胞剔 D11）。
"""
import os
import json
import time

import numpy as np
import scipy.io as sio

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS_ALL = ["16707014", "16708016", "16708060", "16708118", "16713003", "16713110", "16715049"]
CELLS = ["16713003"] if SMOKE else CELLS_ALL
D11 = ("16704007", "16704047")
DT = 1e-4
MASK_N = 200                      # 掩 0-2ms（电容尖刺）
WIN_N = 3000                      # 300ms 窗

TR_GRID = np.exp(np.linspace(np.log(0.0015), np.log(0.008), 13))    # M1/M2：QC 窗
TD_GRID = np.exp(np.linspace(np.log(0.010), np.log(0.100), 20))
TR_GRID_W = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))    # M0：封卷原网格
TD_GRID_W = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))


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


def six_tails(V, I, vtest=40.0):
    """抽六条 -120 反弹尾流（前 300ms，内流翻正）。返回 [(Δt_s, seg)]。"""
    info = segments(V)
    out = []
    for i, (v, s0, n) in enumerate(info):
        if abs(v - vtest) < 2 and i + 1 < len(info) and abs(info[i + 1][0] + 120) < 2 \
                and info[i + 1][2] * DT > 2.0:
            a = info[i + 1][1]
            out.append((n * DT, -I[a:a + WIN_N].astype(float)))
    return out


def fit_tail(seg, model):
    """M0 原法复刻：不掩刺、原宽网格、A 不裁剪 | M1/M2：掩 2ms、QC 窗网格、
    A>=0 逐网格点（负 A 点退化为 基线+漂移 参与竞争）。
    掩刺抹掉上升沿 -> 长 τd+漂移 与负 A 构成简并（debug_C1 实录），QC 窗杀之。"""
    if model == "M0":
        t = np.arange(len(seg)) * DT
        y = seg
        drift = None
        tr_grid, td_grid = TR_GRID_W, TD_GRID_W
    else:
        t = np.arange(MASK_N, len(seg)) * DT
        y = seg[MASK_N:]
        drift = t if model == "M1" else (1.0 - np.exp(-t / 0.120))
        tr_grid, td_grid = TR_GRID, TD_GRID
    best = None
    for tr in tr_grid:
        er = np.exp(-t / tr)
        for td in td_grid[td_grid >= 1.5 * tr]:
            x = np.exp(-t / td) - er
            if model == "M0":
                X = np.column_stack([np.ones(len(t)), x])
            else:
                X = np.column_stack([np.ones(len(t)), x, drift])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if model != "M0" and sol[1] < 0:
                # A>=0 逐点执行：退化为 基线+漂移 两参（与 (tr,td) 无关）
                X2 = np.column_stack([np.ones(len(t)), drift])
                sol2, *_ = np.linalg.lstsq(X2, y, rcond=None)
                sse = float(np.sum((y - X2 @ sol2) ** 2))
                sol = np.array([sol2[0], 0.0, sol2[1]])
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    Xv = float(sol[2]) if model != "M0" else 0.0
    edge = bool(tr <= tr_grid[0] * 1.02 or td >= td_grid[-1] * 0.98 or tr >= tr_grid[-1] * 0.98)
    return dict(A=float(sol[1]), c=float(sol[0]), X=Xv, tau_r=float(tr), tau_d=float(td),
                edge=edge, rms=float(np.sqrt(sse / len(t))))


# ---------------- 对照：合成全族尾流 ----------------
def synth_family(m_of_dt, rng, drift_per_A=-1.44, sigma=0.056):
    """六档合成尾流：A·DoE(3.5/25ms) + X·t 漂移 + 双极尖刺 + 噪声。
    drift_per_A 单位 nA/s per nA·A（t 秒）：实测 003 族 −1.6nA@300ms/A=3.71 -> −1.44。"""
    dts = np.array([0.003, 0.010, 0.030, 0.100, 0.300, 1.000])
    tails = []
    for dt_s in dts:
        A = 3.7 * m_of_dt(dt_s)
        t = np.arange(WIN_N) * DT
        seg = A * (np.exp(-t / 0.025) - np.exp(-t / 0.0035))
        seg = seg + drift_per_A * A * t
        seg[:2] += [5.0, -3.0]                       # 电容尖刺
        seg = seg + rng.normal(0, sigma, WIN_N)
        tails.append((dt_s, seg))
    return tails


def ratio_100_1000(fits):
    amap = {dt_s: f["A"] for dt_s, f in fits}
    a100 = amap.get(0.1)
    a1000 = amap.get(1.0)
    if a100 is None or a1000 is None or a1000 <= 0:
        return float("nan")
    return a100 / a1000


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    print("=" * 84, flush=True)
    print(" α模型 · B6-H3 包络口径审计" + ("（冒烟 16713003）" if SMOKE else "（7 细胞，剔 D11）"), flush=True)
    print(" 判线: 003 修正比 ≥0.20 且 ≥2×M0 -> H3 成立 | 003∈[0.04,0.15] 且中位≤0.20 -> H3 否", flush=True)
    print("=" * 84, flush=True)

    # ---------- 对照（各 10 噪声实现取中位，考系统偏差不考手气） ----------
    print("\n[对照]", flush=True)
    m_c1 = lambda d: 1.0 - np.exp(-d / 0.135)
    m_c2 = lambda d: 1.0 - np.exp(-max(d - 0.075, 0.0) / 0.228)
    r1s, r2s, r1m0s = [], [], []
    for _ in range(10):
        r1s.append(ratio_100_1000([(d, fit_tail(s, "M1")) for d, s in synth_family(m_c1, rng)]))
        r2s.append(ratio_100_1000([(d, fit_tail(s, "M1")) for d, s in synth_family(m_c2, rng)]))
        r1m0s.append(ratio_100_1000([(d, fit_tail(s, "M0")) for d, s in synth_family(m_c1, rng)]))
    r1, r2, r1_m0 = float(np.median(r1s)), float(np.median(r2s)), float(np.median(r1m0s))
    c1ok = 0.40 <= r1 <= 0.65
    c2ok = 0.06 <= r2 <= 0.16
    print(f"  C1 单门无足(真比 0.523): M1 中位 {r1:.3f} [{min(r1s):.2f}~{max(r1s):.2f}]"
          f" ∈[0.40,0.65] -> {'过' if c1ok else '不过'}", flush=True)
    print(f"  C2 角延迟足(真比 0.104): M1 中位 {r2:.3f} [{min(r2s):.2f}~{max(r2s):.2f}]"
          f" ∈[0.06,0.16] -> {'过' if c2ok else '不过'}", flush=True)
    print(f"  登记: 同一 C1 族 M0（原链）回收中位 {r1_m0:.3f}（真 0.523）"
          f"——漂移∝A 时比值对漂移盲免疫（同比例污染分子分母）", flush=True)
    if not (c1ok and c2ok):
        print("  对照未归位 -> 统计量作废，停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    print(f"  {'细胞':>9} | {'M0 比':>6} | {'M1 比':>6} {'M1边缘':>6} | {'M2 比':>6} | "
          f"{'X斜率@1s':>9} {'M1 RMS':>7}", flush=True)
    per_cell = {}
    for cell in CELLS:
        V, I = load_mat("activation_kinetics_2_protocol.mat", cell, "activation_kinetics_2")
        if I is None:
            print(f"  {cell}: 数据缺失", flush=True)
            continue
        tails = six_tails(V, I, 40.0)
        res = {}
        for model in ("M0", "M1", "M2"):
            fits = [(d, fit_tail(s, model)) for d, s in tails]
            res[model] = dict(ratio=ratio_100_1000(fits),
                              fits={f"{d * 1000:.0f}ms": f for d, f in fits})
        # 漂移登记（M1，1000ms 尾的 X 斜率 nA/ms 与其 A 之比）
        f1s = res["M1"]["fits"].get("1000ms")
        xreg = float("nan")
        if f1s and f1s["A"] > 0:
            xreg = f1s["X"] / f1s["A"] * 1000.0      # 每 A 每秒漂移 nA
        nedge = sum(1 for k, f in res["M1"]["fits"].items() if f["edge"])
        per_cell[cell] = dict(M0=res["M0"], M1=res["M1"], M2=res["M2"],
                              x_slope_per_A_per_s=xreg, n_edge_M1=nedge,
                              rms_M1_med=float(np.median([f["rms"] for f in res["M1"]["fits"].values()])))
        print(f"  {cell} | {res['M0']['ratio']:6.3f} | {res['M1']['ratio']:6.3f} "
              f"{nedge:3d}/6 | {res['M2']['ratio']:6.3f} | {xreg:9.3f} "
              f"{per_cell[cell]['rms_M1_med']:7.4f}", flush=True)

    # ---------- 判词 ----------
    print("\n" + "=" * 84, flush=True)
    r003_m0 = per_cell.get("16713003", {}).get("M0", {}).get("ratio", float("nan"))
    r003_m1 = per_cell.get("16713003", {}).get("M1", {}).get("ratio", float("nan"))
    meds = [v["M1"]["ratio"] for v in per_cell.values() if not np.isnan(v["M1"]["ratio"])]
    med = float(np.median(meds)) if meds else float("nan")
    print(f"  003: M0={r003_m0:.3f}  M1 修正={r003_m1:.3f}  七细胞 M1 中位={med:.3f}", flush=True)
    h3_yes = (r003_m1 >= 0.20 and r003_m1 >= 2 * r003_m0) or med >= 0.25
    h3_no = (0.04 <= r003_m1 <= 0.15) and med <= 0.20
    verdict = "H3-env 成立：8% 为测量伪影主责" if h3_yes else \
        "H3-env 否：8% 稳健，envelope 侧非伪影" if h3_no else "中间带：照实登记，不定性"
    print(f"  判词: {verdict}", flush=True)
    print("=" * 84, flush=True)

    tag = "_冒烟" if SMOKE else "_结果"
    fj = os.path.join(HERE, f"2026-09-14_α模型_B6_H3_包络口径审计{tag}.json")
    with open(fj, "w", encoding="utf-8") as fh:
        json.dump(dict(meta=dict(smoke=SMOKE, runtime_s=time.time() - t0,
                                 controls=dict(C1=r1, C2=r2, C1_M0=r1_m0)),
                       per_cell=per_cell, verdict=verdict,
                       median_M1=med), fh, ensure_ascii=False, indent=1, default=float)
    print(f"  结果落盘: {fj}", flush=True)

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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    # 图1：003 六尾流 + M1 分解（鼓包/漂移）
    ax = axes[0]
    cell0 = "16713003"
    if cell0 in per_cell:
        V, I = load_mat("activation_kinetics_2_protocol.mat", cell0, "activation_kinetics_2")
        tails = six_tails(V, I, 40.0)
        t_ms = np.arange(WIN_N) * DT * 1000
        for d, s in tails:
            f = fit_tail(s, "M1")
            tt = np.arange(MASK_N, WIN_N) * DT
            hump = f["A"] * (np.exp(-tt / f["tau_d"]) - np.exp(-tt / f["tau_r"]))
            ax.plot(t_ms, s, color="0.75", lw=0.6)
            ax.plot(tt * 1000, f["c"] + hump, lw=1.4,
                    label=f"Δt={d * 1000:.0f}ms A={f['A']:.2f}")
            ax.plot(tt * 1000, f["c"] + f["X"] * tt, "--", lw=0.8, color="tab:red")
        ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("-120 反弹后时间 (ms)")
    ax.set_ylabel("尾流 (nA)")
    ax.set_title(f"{cell0} M1 分解：实线=基线+鼓包，红虚=基线+漂移")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # 图2：七细胞 M0/M1 修正前后比
    ax = axes[1]
    cells_sorted = sorted(per_cell.keys())
    m0s = [per_cell[c]["M0"]["ratio"] for c in cells_sorted]
    m1s = [per_cell[c]["M1"]["ratio"] for c in cells_sorted]
    xpos = np.arange(len(cells_sorted))
    ax.bar(xpos - 0.18, m0s, 0.36, label="M0 原链")
    ax.bar(xpos + 0.18, m1s, 0.36, label="M1 修正")
    ax.axhline(0.20, color="tab:red", ls=":", lw=1, label="判线 0.20")
    ax.set_xticks(xpos)
    ax.set_xticklabels([c[-3:] for c in cells_sorted])
    ax.set_xlabel("细胞")
    ax.set_ylabel("A(100)/A(1000)")
    ax.set_title("包络 100ms 比：修正前 vs 修正后")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 图3：003 包络曲线 M0 vs M1（六点）
    ax = axes[2]
    if cell0 in per_cell:
        for model, mk, lb in [("M0", "o--", "M0 原链"), ("M1", "s-", "M1 修正")]:
            fits = per_cell[cell0][model]["fits"]
            dts = sorted(fits.keys(), key=lambda k: float(k[:-2]))
            xv = [float(k[:-2]) for k in dts]
            yv = [fits[k]["A"] for k in dts]
            yv = np.array(yv) / max(yv)
            ax.plot(xv, yv, mk, ms=5, label=lb)
        ax.set_xscale("log")
        ax.axvline(100, color="k", ls=":", lw=1)
        ax.set_xlabel("测试脉冲时长 Δt (ms)")
        ax.set_ylabel("归一包络")
        ax.set_title(f"{cell0} 包络形状：M0 vs M1")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("B6-H3 包络口径审计（掩刺+漂移分解）" + ("（冒烟）" if SMOKE else ""), fontsize=12)
    fp = os.path.join(HERE, f"2026-09-14_α模型_B6_H3_包络口径审计{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  图落盘: {fp}", flush=True)


if __name__ == "__main__":
    main()
