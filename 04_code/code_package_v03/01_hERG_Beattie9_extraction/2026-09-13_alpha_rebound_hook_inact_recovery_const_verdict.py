# -*- coding: utf-8 -*-
# 2026-09-13_α模型_反弹hook_失活恢复常数判决.py
# 目的：回答扩展 α 模型的前提问题——失活恢复动力学是不是也是离散常数？
#   方法（与 α 模型同方法论：不假设框架、数据直提、看跨细胞恒定性）：
#     hook/反弹电流 = 先升（失活解除 τ_rec）后降（去激活 τ_deact）
#     拟合形：I(t) = c + A·(e^(−t/τ_d) − e^(−t/τ_r))，τ_d≥1.5τ_r 网格+线性 LS，确定性。
#   样本：
#     A 路 deactivation 协议：+50×2s 后 −120/−110/−100 三条内向尾的 hook
#        （−90/−80 跨反转电位 −88.33mV，信号≈0，不用）；
#     B 路 sine 协议：+40 段后 −120 反弹（单样本下跳>50mV 检测；chirp 前后各一，
#        两次对比 = 历史是否改变时间常数的顺带探针，仅登记不进判词）。
#   判词（预注册，跑前钉死）：
#     每电压每 τ：跨细胞 CV<0.3 判恒定；τ_d/τ_r 中位>2 判可分离；
#     B 路 vs A 路 τ_rec(−120) 每细胞相对差中位<0.3 判跨协议一致；
#     三条全过 => "失活恢复=离散常数，表格式扩展前提成立"；
#     CV 0.3–0.5 => 弱恒定（登记）；>0.5 或不可分离 => 扩展前提不成立，另开新卡。
#   QC：|A|>5σ 且 RMS<max(2σ, 2%|A|) 且 τ 不顶网格边（ms 级上升沿采样错位
#       会注入超噪声残差，绝对 σ 门不适用——此为判决 τ 稳定性非白化）。
#   对照：C1 已知 DoE（τ_r=8ms,τ_d=80ms）恢复误差>15% 作废；
#         C2 纯衰减（无恢复）须把 τ_r 钉在下网格边（不误报有限恢复）。
# 运行：python 本文件（全量九细胞）；SMOKE=1 单细胞 16713003 冒烟。
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
HOOK_GEARS = {-120: 0.30, -110: 0.50, -100: 0.80}   # 电压 -> 拟合窗长(s)
SKIP_S = 0.0005                                      # 台阶后让位 0.5ms（电容尖峰）
SEP_MIN = 1.5                                        # 网格约束 τ_d ≥ 1.5 τ_r
SEP_PASS = 2.0                                       # 可分离判线（中位比值）
CV_PASS = 0.3
AMP_QC = 5.0                                         # 峰幅 > 5σ

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))     # 1–60 ms
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))     # 8ms–2s


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def find_tails(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    tails = []
    for k, (v, s0, n) in enumerate(info):
        if n * DT < 2.0 or k < 1:
            continue
        pv, ps, pn = info[k - 1]
        if pv > -10.0 and 0.2 < pn * DT < 5.0:
            tails.append(dict(v=int(round(v)), start=s0, n=n))
    return tails


def noise_sigma(I_segs):
    """静默段（-80，全长）去趋势 std 中位 -> 每细胞 σ（10kHz 全带宽）"""
    sigs = []
    for seg in I_segs:
        if len(seg) < 2000:
            continue
        t = np.arange(len(seg))
        tr = np.polyfit(t, seg, 1)
        sigs.append(float(np.std(seg - np.polyval(tr, t))))
    return float(np.median(sigs)) if sigs else np.nan


def quiet_segs(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    out = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 0.15 and abs(float(V[s[0]]) + 80.0) < 2.0:
            out.append(I[s[0]:s[-1] + 1].astype(float))
    return out


def doe_fit(t, y, tr_grid=TR_GRID, td_grid=TD_GRID):
    """确定性网格 DoE 拟合 + 一次局部精修。返回 dict 或 None（数据不足）"""
    if len(t) < 50:
        return None

    def scan(trg, tdg):
        best = None
        for tr in trg:
            td_ok = tdg[tdg >= SEP_MIN * tr]
            if not len(td_ok):
                continue
            er = np.exp(-t / tr)
            for td in td_ok:
                x = np.exp(-t / td) - er
                X = np.column_stack([np.ones(len(t)), x])
                sol, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = float(np.sum((y - X @ sol) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, tr, td, sol)
        return best

    b = scan(tr_grid, td_grid)
    if b is None:
        return None
    _, tr0, td0, _ = b
    trg = tr0 * np.exp(np.linspace(-0.35, 0.35, 9))
    tdg = td0 * np.exp(np.linspace(-0.35, 0.35, 9))
    b = scan(trg, tdg)
    sse, tr, td, sol = b
    res = y - (sol[0] + sol[1] * (np.exp(-t / td) - np.exp(-t / tr)))
    return dict(tau_r=float(tr), tau_d=float(td), c=float(sol[0]), A=float(sol[1]),
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(tr <= TR_GRID[0] * 1.02 or tr >= TR_GRID[-1] * 0.98))


def qc_ok(r, sigma):
    return bool(abs(r["A"]) > AMP_QC * sigma
                and r["rms"] < max(2.0 * sigma, 0.02 * abs(r["A"]))
                and not r["edge"])


def fit_hook(I, s0, win_s, sigma):
    n = int(win_s / DT)
    y = I[s0 + int(SKIP_S / DT): s0 + n].astype(float)
    t = np.arange(len(y)) * DT
    half = y[:len(y) // 2]
    sgn = -1.0 if abs(half.min()) > abs(half.max()) else 1.0
    yy = sgn * y
    r = doe_fit(t, yy)
    if r is None:
        return None
    r["valid"] = qc_ok(r, sigma)
    r["sgn"] = sgn
    r["t"] = t
    r["y"] = yy
    return r


def sine_rebounds(V, I):
    """单样本下跳 >50mV 且落到 <-100mV 的沿（chirp 前/后各一；连续 chirp 不触发）"""
    dV = np.diff(V)
    idx = np.where((V[1:] < -100.0) & (V[:-1] > -40.0) & (dV < -50.0))[0]
    out = []
    for k, i in enumerate(idx):
        s0 = i + 1
        n = int(0.15 / DT)
        if s0 + n > len(I):
            continue
        out.append((k, s0, I[s0:s0 + n]))
    return out


def main():
    rng = np.random.default_rng(7)
    print("=" * 78)
    print(" 反弹 hook 失活恢复常数判决" + ("（冒烟）" if SMOKE else "（九细胞全量）"))
    print(f" 判线: CV<{CV_PASS} 恒定 | τ_d/τ_r 中位>{SEP_PASS} 可分离 | 跨协议相对差<0.3")
    print("=" * 78, flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    t_c = np.arange(int(0.3 / DT)) * DT
    y1 = 1.0 * (np.exp(-t_c / 0.08) - np.exp(-t_c / 0.008)) + 0.02 + rng.normal(0, 0.05, len(t_c))
    r1 = doe_fit(t_c, y1)
    e_r = abs(r1["tau_r"] - 0.008) / 0.008
    e_d = abs(r1["tau_d"] - 0.08) / 0.08
    print(f"  C1 DoE(8ms,80ms): τ_r={r1['tau_r']*1000:.1f}ms τ_d={r1['tau_d']*1000:.1f}ms "
          f"误差 {e_r*100:.1f}%/{e_d*100:.1f}%", flush=True)
    y2 = 1.0 * np.exp(-t_c / 0.08) + 0.02 + rng.normal(0, 0.05, len(t_c))
    r2 = doe_fit(t_c, y2)
    c2_ok = r2["tau_r"] <= 0.002 and abs(r2["tau_d"] - 0.08) / 0.08 < 0.15
    print(f"  C2 纯衰减(80ms): τ_r={r2['tau_r']*1000:.2f}ms（应钉下边≤2ms）τ_d={r2['tau_d']*1000:.1f}ms", flush=True)
    if max(e_r, e_d) > 0.15 or not c2_ok:
        print("  对照未归位 -> 统计量作废，停。")
        return
    print("  对照过。", flush=True)

    # ---------- A 路：deactivation hook ----------
    print("\n[A 路] deactivation +50 后 −120/−110/−100 尾 hook", flush=True)
    rows = []
    curves = {}
    for cell in CELLS:
        V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
        if I is None:
            print(f"  细胞 {cell}: 文件缺失")
            continue
        sigma = noise_sigma(quiet_segs(V, I))
        for tl in find_tails(V):
            if tl["v"] not in HOOK_GEARS:
                continue
            r = fit_hook(I, tl["start"], HOOK_GEARS[tl["v"]], sigma)
            if r is None:
                continue
            rows.append(dict(cell=cell, v=tl["v"], tau_r=r["tau_r"], tau_d=r["tau_d"],
                             A=r["A"], c=r["c"], rms=r["rms"], valid=r["valid"], sigma=sigma))
            curves[(cell, tl["v"])] = r
        nv = sum(x["valid"] for x in rows if x["cell"] == cell)
        print(f"  细胞 {cell}: σ={sigma*1000:.0f}pA  有效 hook {nv}/{len(HOOK_GEARS)}", flush=True)

    # ---------- B 路：sine 反弹 ----------
    print("\n[B 路] sine −120 反弹（chirp 前/后各一）", flush=True)
    srows = []
    scurves = {}
    for cell in CELLS:
        V, I = load_mat("sine_wave_protocol.mat", cell, "sine_wave")
        if I is None:
            print(f"  细胞 {cell}: sine 缺失")
            continue
        sigma = noise_sigma(quiet_segs(V, I))
        for k, s0, seg in sine_rebounds(V, I):
            t = np.arange(len(seg)) * DT
            yy = -seg.astype(float)                     # 内向 -> 翻正
            r = doe_fit(t[int(SKIP_S / DT):], yy[int(SKIP_S / DT):])
            if r is None:
                continue
            valid = qc_ok(r, sigma)
            srows.append(dict(cell=cell, which=k, tau_r=r["tau_r"], tau_d=r["tau_d"],
                              A=r["A"], rms=r["rms"], valid=valid, sigma=sigma))
            scurves[(cell, k)] = (t, yy, r)
            tag = "chirp前" if k == 0 else f"chirp后{k}"
            print(f"  细胞 {cell} {tag}: τ_r={r['tau_r']*1000:.2f}ms τ_d={r['tau_d']*1000:.1f}ms "
                  f"A={r['A']:.2f}nA {'过' if valid else '弃'}", flush=True)

    # ---------- 恒定性统计 ----------
    print("\n" + "-" * 78)
    print("[恒定性] A 路跨细胞（仅 QC 有效；n≥3 才判）")
    print(f"  {'V':>6} {'n':>3} | {'τ_rec 中位':>10} {'CV':>6} | {'τ_deact 中位':>10} {'CV':>6} | {'τ_d/τ_r':>8} 判")
    summ = {}
    verdicts = []
    for vv in sorted(HOOK_GEARS):
        rs = [r for r in rows if r["v"] == vv and r["valid"]]
        if len(rs) < 3:
            print(f"  {vv:>6} {len(rs):>3} | 有效不足")
            continue
        tr = np.array([r["tau_r"] for r in rs])
        td = np.array([r["tau_d"] for r in rs])
        ratio = float(np.median(td / tr))
        cv_r = float(tr.std(ddof=1) / tr.mean())
        cv_d = float(td.std(ddof=1) / td.mean())
        ok = cv_r < CV_PASS and cv_d < CV_PASS and ratio > SEP_PASS
        verdicts.append(ok)
        summ[vv] = dict(n=len(rs), tr_med=float(np.median(tr)), cv_r=cv_r,
                        td_med=float(np.median(td)), cv_d=cv_d, ratio=ratio, ok=ok)
        print(f"  {vv:>6} {len(rs):>3} | {np.median(tr)*1000:>8.1f}ms {cv_r:>6.2f} | "
              f"{np.median(td)*1000:>8.1f}ms {cv_d:>6.2f} | {ratio:>8.1f} {'恒定' if ok else '不恒定'}")

    # 跨协议一致（−120：A 路 vs B 路 chirp 前）
    cross = []
    for cell in CELLS:
        a = [r for r in rows if r["cell"] == cell and r["v"] == -120 and r["valid"]]
        b = [r for r in srows if r["cell"] == cell and r["which"] == 0 and r["valid"]]
        if a and b:
            cross.append(abs(a[0]["tau_r"] - b[0]["tau_r"]) / (0.5 * (a[0]["tau_r"] + b[0]["tau_r"])))
    med_cross = float(np.median(cross)) if cross else np.nan
    if cross:
        print(f"\n[跨协议] τ_rec(−120) A路vsB路 每细胞相对差中位: {med_cross:.2f}（n={len(cross)}，判线 <0.3）")
    else:
        print("\n[跨协议] 可比对细胞不足")

    # chirp 前 vs 后登记（不进判词）
    pre = [r for r in srows if r["which"] == 0 and r["valid"]]
    post = [r for r in srows if r["which"] >= 1 and r["valid"]]
    if pre and post:
        print(f"[登记] chirp后 vs 前: τ_r 中位 {np.median([r['tau_r'] for r in pre])*1000:.2f}"
              f"->{np.median([r['tau_r'] for r in post])*1000:.2f}ms，"
              f"A 中位 {np.median([r['A'] for r in pre]):.2f}->{np.median([r['A'] for r in post]):.2f}nA")

    # ---------- 总判词 ----------
    print("\n" + "=" * 78)
    print(" 总判词：")
    n_ok = sum(verdicts)
    print(f"  A 路恒定电压档 {n_ok}/{len(summ)}；跨协议相对差中位 {med_cross:.2f}")
    if n_ok == len(summ) and len(summ) >= 2 and (not np.isnan(med_cross)) and med_cross < 0.3:
        final = "失活恢复=离散常数且跨协议一致 -> 表格式扩展前提成立，可进下一步（扩展模型覆盖 sine/AP）"
    elif n_ok >= 1:
        final = "部分档位恒定：恒定档可入表，其余档位另查（扩展模型分段可行）"
    else:
        final = "失活恢复非离散常数（或不可分离）-> 表格式扩展前提不成立，另开新卡"
    print("  " + final)
    print("=" * 78)

    # ---------- 图 ----------
    # 页1：A 路 hook 叠图 9细胞×3档
    fig, axes = plt.subplots(len(CELLS), 3, figsize=(15, 1.9 * len(CELLS)), squeeze=False)
    for i, cell in enumerate(CELLS):
        for j, vv in enumerate(sorted(HOOK_GEARS)):
            ax = axes[i][j]
            r = curves.get((cell, vv))
            if r is None:
                ax.axis("off"); continue
            dn = max(1, int(0.0005 / DT))
            ax.plot(r["t"][::dn] * 1000, r["y"][::dn], ".", ms=1.5, color="black", alpha=0.5)
            yf = r["c"] + r["A"] * (np.exp(-r["t"] / r["tau_d"]) - np.exp(-r["t"] / r["tau_r"]))
            ax.plot(r["t"] * 1000, yf, "-", lw=1.3, color="crimson")
            ax.set_title(f"{cell[-4:]} @{vv}mV τ_r={r['tau_r']*1000:.1f}ms τ_d={r['tau_d']*1000:.0f}ms "
                         f"{'过' if r['valid'] else '弃'}", fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("A 路 deactivation hook：原始 vs DoE 拟合（升=失活解除，降=去激活）", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fp1 = os.path.join(HERE, f"2026-09-13_α模型_反弹hook_A路{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 页2：B 路反弹叠图 + τ(V) 汇总
    fig, axes = plt.subplots(2, max(3, int(np.ceil(len(CELLS) / 2)) + 1), figsize=(17, 7))
    axs = axes.ravel()
    ax = axs[0]
    plotted = False
    for vv in sorted(HOOK_GEARS):
        if vv in summ:
            s = summ[vv]
            ax.errorbar([vv], [s["tr_med"] * 1000], yerr=s["tr_med"] * 1000 * s["cv_r"],
                        fmt="o", ms=7, capsize=4, color="crimson",
                        label="τ_rec" if not plotted else None)
            ax.errorbar([vv], [s["td_med"] * 1000], yerr=s["td_med"] * 1000 * s["cv_d"],
                        fmt="s", ms=7, capsize=4, color="navy",
                        label="τ_deact" if not plotted else None)
            plotted = True
    ax.set_yscale("log"); ax.set_xlabel("mV"); ax.set_ylabel("τ (ms)")
    ax.set_title("τ_rec / τ_deact(V)（中位 ± CV·中位）", fontweight="bold")
    if plotted:
        ax.legend()
    ax.grid(alpha=0.3, which="both")
    for i, cell in enumerate(CELLS):
        ax = axs[i + 1]
        got = False
        for k, lbl, clr in ((0, "chirp前", "crimson"), (1, "chirp后", "darkorange")):
            key = (cell, k)
            if key not in scurves:
                continue
            t, yy, r = scurves[key]
            i0 = int(SKIP_S / DT)
            dn = 5
            ax.plot(t[i0::dn] * 1000, yy[i0::dn], ".", ms=1.5, color="black", alpha=0.4)
            yf = r["c"] + r["A"] * (np.exp(-t / r["tau_d"]) - np.exp(-t / r["tau_r"]))
            ax.plot(t[i0:] * 1000, yf[i0:], "-", lw=1.2, color=clr, label=lbl)
            got = True
        if got:
            ax.legend(fontsize=7)
        ax.set_title(f"{cell[-4:]} sine −120 反弹", fontsize=8)
        ax.grid(alpha=0.3)
    for k in range(len(CELLS) + 1, len(axs)):
        axs[k].axis("off")
    fig.suptitle("B 路 sine −120 反弹：chirp 前 vs chirp 后（翻正显示）", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp2 = os.path.join(HERE, f"2026-09-13_α模型_反弹hook_B路{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    out = dict(summary=summ, med_cross=med_cross, final=final,
               A_rows=[{k: v for k, v in r.items() if k not in ("t", "y")} for r in rows],
               B_rows=srows, figA=fp1, figB=fp2)
    fj = os.path.join(HERE, f"2026-09-13_α模型_反弹hook{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  图落盘: {fp1}")
    print(f"           {fp2}")
    print(f"  结果落盘: {fj}")


if __name__ == "__main__":
    main()
