# 2026-09-13_α模型_激活envelope_τact封卷判决.py
# 目的（缺口 3 主攻：τ_act(V) 就地封卷，D7 绕开方案）：
#   activation_kinetics_1/2 是尾流包络协议：-120x2.5s 全恢复+全去激活复位
#   （m≈0,h=1 干净初态）-> -120x0.05 -> -80x0.2 -> 测试脉冲（k1=0mV / k2=+40mV，
#   Δt = 3/10/30/100/300/1000ms 六档递增）-> -120x2.5s（反弹 DoE：τr 3ms 恢复、
#   τd 30ms 去激活，已封卷口径）-> 反弹幅度 A ∝ m(Δt)。包络 A(Δt) 即激活时程。
#   优势：绕开 D7（不在 +40 稳态小电流里找 m 骑乘），反弹是大信号快结构，
#   DoE 管道已封卷验证（16713003 冒烟：k2 单调包络，300/1000ms 点回收
#   τr/τd=3.9/20.7、3.3/30.3 正中封卷值 3.04/30.3，内证自洽）。
#
# 【16713003 设计冒烟事实（判线设计依据）】
#   k2(+40) 包络比 0.012/0.011/0.015/0.079/0.637/1.0，主上升 100-300ms，
#     带 sigmoid 足（延迟）；A_max=3.71 vs G·DF 预测 4.15（89%）。
#   k1(0) 前 5 档全在噪声地板（<=0.13nA，DoE 撞边作废），仅 1000ms 点 0.776 扎实
#     -> 0mV 激活秒级慢，与登记梯子（0mV:2.0s）方向一致；A(V) 曲线浅系
#     -90x60ms 复位残留 m≈0.43 地板效应（纠偏在案），两口径不矛盾。
#
# 判线（跑前声明）：
#   QC1（DoE 门）：进入包络拟合的点须 τr∈[1.5,8]ms 且 τd∈[12,80]ms（封卷值
#     3.04/30.3 留裕量）且 A>=4σ_cell（σ_cell 取该细胞 -80x2.4s 基线段 std）；
#     不满足的点剔除并计数。
#   拟合形：A(Δt)=A_inf·(1-exp(-max(Δt-d,0)/τ))，网格 d∈{0,25,...,150}ms，
#     τ 对数网格（k2: [0.02,1.5]s；k1: [0.2,8]s），A_inf 由线性最小二乘。
#   QC2（A_inf 门）：A_inf_fit / (G·31.67) ∈ [0.5,2.0]（G 声明 ±30% + m_ss
#     声明 + h 因子），越界细胞记旗 posthoc_ainf_out。
#   k1 另做锚定拟合：A_inf 固定 = G·31.67（m_ss(0)=1 声明），单参数 τ_anch。
#   单点锚定回退级（2026-09-13 跑前增补）：n_qc=1 时 τ_single = -Δt/ln(1-A/(G·DF))，
#     仅当 0<A/(G·DF)<1；d=0 声明、延迟简并旗 single_pt_d0 在案；A/(G·DF)>=1
#     说明 m_ss(0)<1 或 G 偏高，该点记旗 mss_lt1_candidate 不给出 τ。
#   封卷判线（三档，跑前钉死）：
#     τ_act(+40)：QC 全过细胞 >=7 且 τ 九细胞 CV<0.3 -> 【封卷】；
#       QC<7 或 CV>=0.3 -> 【登记】（照实写数值不封）。
#     τ_act(0)：每细胞最佳估计 = τ_anch（>=2点）优先、τ_single（1点）兜底；
#       有效细胞 >=7 且 CV<0.3 -> 【半封卷】（锚定值入表，声明外推与单点简并）；
#       否则 -> 【登记区间】（给 τ_free/τ_anch/τ_single 三列）。
# 对照（跑前声明）：
#   C1 DoE 反演 τr 3.5ms 误差<15%（管道）；
#   C2 合成包络（d=50ms,τ=150ms,A_inf=3.7 + 16713003 实测基线噪声）回收
#     τ 误差<20%、d 误差<50ms -> 不过则统计量作废停。
# 运行：python 本文件（九细胞全量）；SMOKE=1 单细胞 16713003 冒烟。
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
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)

F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
SEP_MIN = 1.5
D_GRID = np.arange(0, 0.1501, 0.025)          # 延迟网格 s
TAU_GRID_K2 = np.exp(np.linspace(np.log(0.02), np.log(1.5), 40))
TAU_GRID_K1 = np.exp(np.linspace(np.log(0.2), np.log(8.0), 40))
TR_OK = (0.0015, 0.008)
TD_OK = (0.012, 0.080)


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def doe_fit(t, y):
    if len(t) < 50:
        return None
    best = None
    for tr in TR_GRID:
        for td in TD_GRID[TD_GRID >= SEP_MIN * tr]:
            x = np.exp(-t / td) - np.exp(-t / tr)
            X = np.column_stack([np.ones(len(t)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    return dict(tau_r=float(tr), tau_d=float(td), A=float(sol[1]),
                edge=bool(tr <= TR_GRID[0] * 1.02 or td >= TD_GRID[-1] * 0.98
                          or tr >= TR_GRID[-1] * 0.98))


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def envelope(cell, proto, vtest):
    """抽包络：每周期 测试脉冲(vtest) -> -120 反弹 DoE。返回 (dt列表, 拟合点列表, σ)。"""
    V, I = load_mat(f"{proto}_protocol.mat", cell, proto)
    if I is None:
        return None
    info = segments(V)
    # σ_cell：首个 -80x2.4s 长基线段
    sigma = None
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT > 2.0:
            sigma = float(np.std(I[s0 + 2000: s0 + n]))
            break
    pts = []
    for i, (v, s0, n) in enumerate(info):
        if abs(v - vtest) < 2 and i + 1 < len(info) and abs(info[i + 1][0] + 120) < 2 \
                and info[i + 1][2] * DT > 2.0:
            a = info[i + 1][1]
            w = a + 3000                          # 反弹前 300ms（DoE 快结构）
            t = np.arange(w - a) * DT
            r = doe_fit(t, -I[a:w])
            if r is None:
                continue
            ok = (not r["edge"]) and TR_OK[0] <= r["tau_r"] <= TR_OK[1] \
                and TD_OK[0] <= r["tau_d"] <= TD_OK[1] and r["A"] >= 4 * (sigma or 1)
            pts.append(dict(dt=n * DT, A=r["A"], tau_r=r["tau_r"], tau_d=r["tau_d"],
                            qc=bool(ok)))
    return dict(pts=pts, sigma=sigma)


def fit_env(dts, As, tau_grid, d_grid=D_GRID, ainf_fix=None):
    """网格 (d,τ)，A_inf 线性 lstsq（或固定）。返回 best dict。"""
    dts = np.asarray(dts, float)
    As = np.asarray(As, float)
    best = None
    for d in d_grid:
        rise = 1.0 - np.exp(-np.maximum(dts - d, 0.0)[:, None] / tau_grid[None, :])
        for j, tau in enumerate(tau_grid):
            x = rise[:, j]
            if ainf_fix is not None:
                ainf = ainf_fix
            else:
                ainf = float(As @ x / (x @ x)) if x @ x > 1e-12 else 0.0
            sse = float(np.sum((As - ainf * x) ** 2))
            if best is None or sse < best[0]:
                best = (sse, d, float(tau), ainf)
    sse, d, tau, ainf = best
    return dict(d=d, tau=tau, A_inf=ainf, sse=sse)


def g_anchor(cell, hook, inact):
    """G 回退链（与 sine 冒烟 R1 同口径）。"""
    for r in hook["A_rows"]:
        if r["cell"] == cell and r["v"] == -120 and r["valid"] and r["A"] > 0:
            return r["A"] / DF_M120, None
    cands = [r["A"] / DF_M120 for r in hook["A_rows"]
             if r["cell"] == cell and r["valid"] and r["A"] > 0]
    if cands:
        return float(np.median(cands)), None
    ci = inact["cells"].get(cell, {})
    if ci.get("g_hat") and ci["g_hat"] > 0:
        return float(ci["g_hat"]), "posthoc_G_inact"
    return None, "posthoc_G_fail"


def main():
    rng = np.random.default_rng(7)
    print("=" * 86)
    print(" α模型 激活 envelope τ_act 封卷判决" + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"))
    print(" 判线: +40 QC>=7且CV<0.3 封卷 | 0mV free/anch 一致且 CV<0.3 半封卷，否则登记")
    print("=" * 86, flush=True)

    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    t_c = np.arange(int(0.3 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE 反演 τr: {rc['tau_r'] * 1e3:.2f}ms（真值3.5）-> {'过' if c1 else '不过'}",
          flush=True)
    # C2 合成包络回收
    dts_syn = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    A_syn = 3.7 * (1 - np.exp(-np.maximum(dts_syn - 0.05, 0) / 0.15)) \
        + rng.normal(0, 0.04, len(dts_syn))
    f_syn = fit_env(dts_syn, A_syn, TAU_GRID_K2)
    c2 = bool(abs(f_syn["tau"] - 0.15) / 0.15 < 0.20 and abs(f_syn["d"] - 0.05) < 0.05)
    print(f"  C2 合成包络回收: d={f_syn['d'] * 1e3:.0f}ms（真50） τ={f_syn['tau'] * 1e3:.0f}ms"
          f"（真150） A_inf={f_syn['A_inf']:.2f}（真3.7）-> {'过' if c2 else '不过'}", flush=True)
    if not (c1 and c2):
        print("  对照未归位 -> 统计量作废，停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    res = {}
    for c in CELLS:
        G, gflag = g_anchor(c, hook, inact)
        e1 = envelope(c, "activation_kinetics_1", 0.0)
        e2 = envelope(c, "activation_kinetics_2", 40.0)
        out = dict(G=G, gflag=gflag)
        for tag, e, tg in (("k1_0mV", e1, TAU_GRID_K1), ("k2_+40", e2, TAU_GRID_K2)):
            if e is None or not e["pts"]:
                out[tag] = dict(n=0)
                continue
            good = [p for p in e["pts"] if p["qc"]]
            rec = dict(n=len(e["pts"]), n_qc=len(good), sigma=e["sigma"],
                       pts=[(round(p["dt"] * 1e3), round(p["A"], 4)) for p in e["pts"]])
            if len(good) >= 2:
                f = fit_env([p["dt"] for p in good], [p["A"] for p in good], tg)
                rec.update(tau=f["tau"], d=f["d"], A_inf=f["A_inf"])
                if G:
                    pred = G * DF_M120
                    rec["ainf_ratio"] = f["A_inf"] / pred
                    rec["qc2"] = bool(0.5 <= rec["ainf_ratio"] <= 2.0)
                    fa = fit_env([p["dt"] for p in good], [p["A"] for p in good], tg,
                                 ainf_fix=pred)
                    rec["tau_anch"] = fa["tau"]
                    rec["d_anch"] = fa["d"]
            elif len(good) == 1 and G:
                p = good[0]                               # 单点锚定回退级（跑前增补）
                r_ = p["A"] / (G * DF_M120)
                if 0 < r_ < 1:
                    rec["tau_single"] = float(-p["dt"] / np.log(1 - r_))
                    rec["sflag"] = "single_pt_d0"
                else:
                    rec["sflag"] = "mss_lt1_candidate"
            out[tag] = rec
        res[c] = out
        k1, k2 = out["k1_0mV"], out["k2_+40"]
        print(f"  {c}: G={G:.4f}" +
              (f" | +40: QC {k2.get('n_qc', 0)}/{k2.get('n', 0)} "
               f"τ={k2.get('tau', np.nan) * 1e3:.0f}ms d={k2.get('d', np.nan) * 1e3:.0f}ms "
               f"A_inf={k2.get('A_inf', np.nan):.2f}(比{k2.get('ainf_ratio', np.nan):.2f})"
               if k2.get("n") else " | +40: 无数据") +
              (f" | 0mV: QC {k1.get('n_qc', 0)}/{k1.get('n', 0)} "
               f"τf={k1.get('tau', np.nan):.2f}s τa={k1.get('tau_anch', np.nan):.2f}s "
               f"τs={k1.get('tau_single', np.nan):.2f}s"
               if k1.get("n") else " | 0mV: 无数据"), flush=True)

    # ---------- 封卷判决 ----------
    print("\n" + "-" * 86, flush=True)
    taus40 = [res[c]["k2_+40"]["tau"] for c in res
              if res[c]["k2_+40"].get("qc2") and res[c]["k2_+40"].get("n_qc", 0) >= 2]
    n40 = len(taus40)
    if n40 >= 2:
        cv40 = float(np.std(taus40) / np.mean(taus40))
    else:
        cv40 = np.nan
    seal40 = bool(n40 >= (1 if SMOKE else 7) and np.isfinite(cv40) and cv40 < 0.3)
    print(f"  τ_act(+40): QC2 过 {n40}/{len(res)}，τ 中位 "
          f"{np.median(taus40) * 1e3 if taus40 else np.nan:.0f}ms，CV={cv40:.3f} -> "
          f"{'【封卷】' if seal40 else '【登记】'}", flush=True)

    taf, taa = [], []
    best0 = {}
    for c in res:
        k1 = res[c]["k1_0mV"]
        if k1.get("tau") and k1.get("tau_anch"):
            taf.append(k1["tau"])
            taa.append(k1["tau_anch"])
        est = k1.get("tau_anch", k1.get("tau_single"))   # τ_anch 优先，τ_single 兜底
        if est:
            best0[c] = est
    n0 = len(best0)
    v0 = list(best0.values())
    if n0 >= 2:
        cv0 = float(np.std(v0) / np.mean(v0))
    else:
        cv0 = np.nan
    ratio01 = float(np.median(np.array(taf) / np.array(taa))) if len(taf) >= 2 else np.nan
    seal0 = bool(n0 >= (1 if SMOKE else 7) and np.isfinite(cv0) and cv0 < 0.3)
    print(f"  τ_act(0mV): 有效估计 {n0}/{len(res)}（τ_anch 优先 τ_single 兜底），"
          f"中位 {np.median(v0) if v0 else np.nan:.2f}s，CV={cv0:.3f}"
          f"（free/anch 比 {ratio01:.2f} 参考）-> "
          f"{'【半封卷】' if seal0 else '【登记区间】'}", flush=True)

    verdict = []
    verdict.append("τ_act(+40) " + ("封卷" if seal40 else "登记"))
    verdict.append("τ_act(0mV) " + ("半封卷" if seal0 else "登记区间"))
    print(" 总判词：", "；".join(verdict), flush=True)

    # ---------- 图 ----------
    nfig = len(res)
    fig, axes = plt.subplots(nfig, 2, figsize=(13, 2.6 * nfig), squeeze=False)
    for row, c in enumerate(res):
        for col, (proto, vt, tag) in enumerate(
                [("activation_kinetics_1", 0.0, "k1_0mV"),
                 ("activation_kinetics_2", 40.0, "k2_+40")]):
            ax = axes[row][col]
            e = envelope(c, proto, vt)
            if e is None:
                continue
            dts = [p["dt"] * 1e3 for p in e["pts"]]
            As = [p["A"] for p in e["pts"]]
            qc = [p["qc"] for p in e["pts"]]
            ax.scatter([d for d, q in zip(dts, qc) if q],
                       [a for a, q in zip(As, qc) if q], c="tab:blue", s=40, label="QC过")
            ax.scatter([d for d, q in zip(dts, qc) if not q],
                       [a for a, q in zip(As, qc) if not q], c="0.7", s=30, marker="x",
                       label="QC剔")
            rec = res[c][tag]
            if rec.get("tau"):
                dd = np.linspace(0, 1.0, 200)
                ax.plot(dd * 1e3, rec["A_inf"] * (1 - np.exp(
                    -np.maximum(dd - rec["d"], 0) / rec["tau"])), "tab:red",
                    lw=1.2, label=f"拟合 τ={rec['tau'] * 1e3:.0f}ms d={rec['d'] * 1e3:.0f}ms")
            ax.set_xscale("log")
            ax.set_title(f"{c} {tag}", fontsize=9)
            ax.set_xlabel("Δt (ms)")
            ax.set_ylabel("反弹 A (nA)")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_激活envelope_τact封卷判决.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    fjson = os.path.join(HERE, "2026-09-13_α模型_激活envelope_τact封卷判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(dict(cells=res, seal40=seal40, cv40=cv40, tau40_med=float(
            np.median(taus40)) if taus40 else None, seal0=seal0, cv0=cv0,
            tau0_best=best0, tau0_med=float(np.median(v0)) if v0 else None,
            ratio_free_anch=ratio01), f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  图落盘: {fpng}", flush=True)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
