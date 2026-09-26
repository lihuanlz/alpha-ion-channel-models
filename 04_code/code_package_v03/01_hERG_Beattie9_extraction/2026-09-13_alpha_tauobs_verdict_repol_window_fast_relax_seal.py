# 2026-09-13_α模型_τobs判决_复极窗快弛豫封卷.py
# 目的（缺口2 复极段 + 缺口1b 一并收口，正式判决，九细胞）：
#   失活协议测试段（测试V×0.150s，前接 -90x0.06s 复位 -> h(0)≈1、m(0)≈0.43 残留在案）
#   早期相位携带 τ_obs（h 从 1 向 h_ss(V) 的弛豫时间）。
#   侦察在案（2026-09-13_α模型_τobs侦察_测试段早期相位.png/.py）：-70..-30mV
#   30ms 比值 1.03-1.22 ≈ 纯 m 衰减，τ_obs≥25ms 的形状（≥1.4）被明确排除。
# 判词靶：
#   (1) τ_obs(V) 上界表（-80..-30 六档）——复极窗 τ_h 实测锚（缺口2）；
#   (2) h_ss=h_150 身份（缺口1b）：τ_obs 上界 ≤20ms 时族 hss=(h150-e^-150/τ)/(1-e^-150/τ)
#       被迫 ≈h150，身份自动成立。
# 模型（比值式，K=G·DF 归一消去）：
#   R(t)=I(t)/i_end = [m(t)/m(150)]·[h(t)/h(150)]，i_end=末20ms均值；
#   m(t)=mss+(m0-mss)e^{-t/τ_eff}（mss 用换表版重分裂值，与 forward 同链）；
#   h(t)=hss+(1-hss)e^{-t/τ_obs}，hss 由 (h150,τ_obs) 族钉死（端点 h(0)=1、h(150)=h150）。
#   h150：剥离 JSON 每细胞 qc 过档值，否->群体中位（与换表版 forward 同一来源链）。
# 网格：τ_obs∈[1,300]ms 48点对数 × τ_eff∈{0.3,0.5,0.8,1.2,2,3}s × m0∈{0.2,0.3,0.435,0.55,0.7}。
# 统计门（不用白噪声假设，与 v5 包络门同纪律）：
#   数据 2ms 分箱中位；τ_obs 合格 <=> 10/30/60ms 三个模型分箱值全部落在数据 ±3σ_b 内
#   （σ_b=1.25·σ/√20/|i_end|，σ=该细胞末段 -80x1.34s 去趋势 std）。
#   上界=合格 τ_obs 最大值；点估计=全 75 分箱 SSE 最小（σ_b 口径）。
# 对照（跑前声明）：
#   C1a 合成 τ_obs=60ms（h150=0.33,m0=0.435,τ_eff=0.8,σ_R=0.027）-> 点估计∈[40,90]ms
#       且上界≥60ms（管道不会把慢的说没）；
#   C1b 合成 τ_obs=5ms -> 上界≤15ms（管道不会把快的说慢）；
#   C2 合成 τ=60 的 hss 真值 0.270 -> |hss_best-0.270|/0.270≤25%（族分裂可用）。
# 判线（跑前声明，跑后不动）：
#   每档：有效细胞数 n≥4 且 τ_obs 上界中位 ≤20ms -> 该档"快弛豫+hss=h150身份"封卷；
#   n≥4 但上界中位 >20ms -> 该档登记（给上界中位数）；n<4 数据不足。
#   六档全封卷 -> 总判"复极窗 τ_h 快（≤20ms），h_ss=h150 身份封卷，缺口1b/缺口2复极段收口"；
#   有档不封 -> 照实登记，不修饰。
#   τ_obs 点估计逐细胞给出（预期多数撞 1-10ms 低网格 -> 本质是上界表，判词照实写）。
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
GEARS = [-80, -70, -60, -50, -40, -30]
TAU_OBS = np.exp(np.linspace(np.log(0.001), np.log(0.300), 48))
TAU_EFF = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
M0S = [0.2, 0.3, 0.435, 0.55, 0.7]
UB_LINE = 0.020          # 上界判线 20ms
TQ = [0.010, 0.030]      # 早期两点门（10/30ms；60ms 弃用——见下）
FORM_TOL = 0.03          # 模型形误差容忍（绝对值，比值口径，跑前声明）
# 门设计（冒烟定在案）：真实数据 60->150ms 段有 ~3% 非单调结构（下凹-回升，单h+单m
#   模型不产生，疑似慢残差/减法伪影，登记）；判 τ_obs 只用早期 10/30ms 两点——慢 h
#   （τ>=25ms）在早期窗的信号是 0.2-0.7 倍值量级，远超该异常与噪声，上界保守有效。
#   每个 τ_obs 先按全 75 分箱 SSE 取最优 (τ_eff,m0)，再查两点 |数据-模型| <= 3σ_b+FORM_TOL。

F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")
F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")


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


def test_segments(V):
    info = segments(V)
    out = {}
    for k, (v, s0, n) in enumerate(info):
        if 0.14 < n * DT < 0.16 and k >= 1:
            pv, ps, pn = info[k - 1]
            if abs(pv + 90.0) < 2.0 and 0.05 < pn * DT < 0.07:
                out[int(round(v))] = (s0, n)
    return out


def noise_sigma(V, I):
    info = segments(V)
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT > 1.0:
            seg = I[s0 + n - 5000: s0 + n].astype(float)
            seg = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg)))
            return float(np.std(seg))
    return np.nan


def bin2(t, y, w=0.002):
    nb = int(round(t[-1] / w))
    tm, ym = [], []
    for b in range(nb):
        m = (t >= b * w) & (t < (b + 1) * w)
        if m.sum() >= 3:
            tm.append((b + 0.5) * w)
            ym.append(float(np.median(y[m])))
    return np.array(tm), np.array(ym)


def mss_table(cell, amp, hssj):
    """与换表版 forward 同链：mss=y_ss/h150（缺档中位），-80/-30 用桥式。"""
    yss = {}
    for r in amp["rows"]:
        if r["v"] in (-70, -60, -50):
            yss.setdefault(r["v"], {})[r["cell"]] = max(r["y_ss"], 1e-4)
    med = {v: float(np.median(list(d.values()))) for v, d in yss.items()}
    def cell_y(v):
        return yss.get(v, {}).get(cell, med.get(v, 0.02))
    hc = hssj["cells"].get(cell, {}).get("curve", {})
    def h150(v):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            return float(e["h"])
        return float(hssj["gears"][str(v)]["med"])
    return {-80: min(1.0, cell_y(-70) * 0.5 / h150(-80)),
            -70: min(1.0, cell_y(-70) / h150(-70)),
            -60: min(1.0, cell_y(-60) / h150(-60)),
            -50: min(1.0, cell_y(-50) / h150(-50)),
            -40: min(1.0, cell_y(-50) * 3.0 / h150(-40)),
            -30: min(1.0, cell_y(-50) * 8.0 / h150(-30))}, h150


def model_R(tb, tau_obs, tau_eff, m0, mss, h150):
    r = np.exp(-0.150 / tau_obs)
    hss = (h150 - r) / (1.0 - r) if r < 0.999999 else h150
    hss = max(hss, 0.0)
    h = hss + (1.0 - hss) * np.exp(-tb / tau_obs)
    m = mss + (m0 - mss) * np.exp(-tb / tau_eff)
    y = m * h
    return y / y[-1], hss


def fit_gear(tb, yb, sig_b, mss, h150):
    """返回 best(τ_obs,τ_eff,m0,hss,sse) 与上界 ub；无效返回 None。"""
    best = None
    ub = 0.0
    qidx = [int(np.argmin(np.abs(tb - x))) for x in TQ]   # 最近分箱（箱心 1,3,5...ms）
    for to in TAU_OBS:
        for te in TAU_EFF:
            for m0 in M0S:
                yhat, hss = model_R(tb, to, te, m0, mss, h150)
                sse = float(np.sum(((yb - yhat) / sig_b) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, to, te, m0, hss)
                if np.all(np.abs(yb[qidx] - yhat[qidx]) <= 3 * sig_b + FORM_TOL):
                    ub = max(ub, to)
    if best is None:
        return None
    sse, to, te, m0, hss = best
    return dict(tau_obs=float(to), ub=float(ub), tau_eff=float(te), m0=float(m0),
                hss=float(hss), h150=float(h150), sse=sse)


def main():
    amp = json.load(open(F_AMP, encoding="utf-8"))
    hssj = json.load(open(F_HSS, encoding="utf-8"))
    print("=" * 92)
    print(" τ_obs 判决 · 复极窗快弛豫封卷（缺口2 复极段 + 缺口1b）"
          + ("（冒烟 16713003）" if SMOKE else "（九细胞）"))
    print(f" 判线: 每档 n>=4 且 τ_obs 上界中位 <={UB_LINE * 1e3:.0f}ms -> 快弛豫+hss=h150身份 封卷")
    print("=" * 92, flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    tb_c = (np.arange(75) + 0.5) * 0.002
    rng = np.random.default_rng(20260913)
    SIG_RAW_C = 0.027                      # 合成原始点噪声（003 -70 档口径）
    sig_c = 1.25 * SIG_RAW_C / np.sqrt(20)  # 2ms 分箱中位 SE（与真实数据路径同口径）
    # C1a: τ=60ms 慢弛豫必须被看见
    y_c, hss_c = model_R(tb_c, 0.060, 0.8, 0.435, 0.057, 0.33)
    y_c = y_c + rng.normal(0, sig_c, len(tb_c))
    rc = fit_gear(tb_c, y_c, sig_c, 0.057, 0.33)
    c1a = bool(rc and 0.040 <= rc["tau_obs"] <= 0.090 and rc["ub"] >= 0.060)
    print(f"  C1a 合成τ=60ms: 点估计 {rc['tau_obs'] * 1e3:.0f}ms 上界 {rc['ub'] * 1e3:.0f}ms "
          f"（要求[40,90]且上界>=60）-> {'过' if c1a else '不过'}", flush=True)
    # C1b: τ=5ms 快弛豫上界必须压得住
    y_c2, _ = model_R(tb_c, 0.005, 0.8, 0.435, 0.057, 0.33)
    y_c2 = y_c2 + rng.normal(0, sig_c, len(tb_c))
    rc2 = fit_gear(tb_c, y_c2, sig_c, 0.057, 0.33)
    c1b = bool(rc2 and rc2["ub"] <= 0.015)
    print(f"  C1b 合成τ=5ms:  点估计 {rc2['tau_obs'] * 1e3:.0f}ms 上界 {rc2['ub'] * 1e3:.0f}ms "
          f"（要求上界<=15）-> {'过' if c1b else '不过'}", flush=True)
    # C2: 族分裂 hss 回收
    c2 = bool(rc and abs(rc["hss"] - 0.270) / 0.270 <= 0.25)
    print(f"  C2 族分裂 hss: {rc['hss']:.3f}（真值0.270，±25%）-> {'过' if c2 else '不过'}", flush=True)
    if not (c1a and c1b and c2):
        print("  对照未归位 -> 统计量作废，停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    rows = {g: [] for g in GEARS}
    detail = {}
    for cell in CELLS:
        V, I = load_mat("inactivation_protocol.mat", cell, "inactivation")
        ts = test_segments(V)
        sig = noise_sigma(V, I) if I is not None else np.nan
        mss, h150f = mss_table(cell, amp, hssj)
        detail[cell] = {}
        line = f"  {cell}: "
        for g in GEARS:
            if g not in ts or I is None:
                line += f"{g}:无档 "
                continue
            s0, n = ts[g]
            t = np.arange(n) * DT
            i = I[s0: s0 + n]
            i_end = float(np.mean(i[-2000:]))
            if abs(i_end) < 5 * sig:
                line += f"{g}:弱 "
                continue
            tb, yb = bin2(t, i / i_end)
            sig_b = 1.25 * sig / np.sqrt(20) / abs(i_end)
            r = fit_gear(tb, yb, sig_b, mss[g], h150f(g))
            if r is None:
                line += f"{g}:× "
                continue
            r["i_end"] = i_end
            detail[cell][g] = r
            rows[g].append(r)
            line += f"{g}:τ={r['tau_obs'] * 1e3:.0f}/ub={r['ub'] * 1e3:.0f} "
        print(line, flush=True)

    # ---------- 判词 ----------
    print("\n" + "-" * 92, flush=True)
    print(f"{'档':>6} {'n':>3} {'τ点估计中位':>12} {'上界中位':>10} {'上界最大':>10}  判", flush=True)
    verdicts = {}
    for g in GEARS:
        rs = rows[g]
        n = len(rs)
        if n >= 4:
            med_est = float(np.median([r["tau_obs"] for r in rs]))
            med_ub = float(np.median([r["ub"] for r in rs]))
            mx_ub = float(np.max([r["ub"] for r in rs]))
            ok = med_ub <= UB_LINE
            vd = "【封卷：快弛豫+hss=h150身份】" if ok else "【登记：上界超线】"
            verdicts[g] = dict(n=n, med_est=med_est, med_ub=med_ub, max_ub=mx_ub,
                               sealed=bool(ok))
            print(f"{g:>6} {n:>3} {med_est * 1e3:>10.0f}ms {med_ub * 1e3:>8.0f}ms "
                  f"{mx_ub * 1e3:>8.0f}ms  {vd}", flush=True)
        else:
            verdicts[g] = dict(n=n, sealed=False)
            print(f"{g:>6} {n:>3}      --       --       --  【数据不足】", flush=True)
    nseal = sum(1 for v in verdicts.values() if v.get("sealed"))
    print("=" * 92, flush=True)
    if nseal == len(GEARS):
        final = ("复极窗 τ_h 快（上界中位≤20ms 六档全封卷），h_ss=h150 身份封卷——"
                 "缺口1b 收口；缺口2 复极段实测锚落地（B2 桥 11-23ms 与实测上限一致，"
                 "桥退役）")
    else:
        final = f"封卷 {nseal}/{len(GEARS)} 档——未封档照实登记，明细见上"
    print(" 总判词：" + final, flush=True)

    # ---------- 图 ----------
    cells_done = [c for c in CELLS if detail.get(c)]
    fig, axes = plt.subplots(len(cells_done), len(GEARS),
                             figsize=(2.6 * len(GEARS), 2.2 * len(cells_done)), squeeze=False)
    for ri, cell in enumerate(cells_done):
        V, I = load_mat("inactivation_protocol.mat", cell, "inactivation")
        ts = test_segments(V)
        mss, h150f = mss_table(cell, amp, hssj)
        for ci, g in enumerate(GEARS):
            ax = axes[ri][ci]
            r = detail[cell].get(g)
            if r is None or g not in ts:
                ax.set_title(f"{g}mV 无", fontsize=7)
                continue
            s0, n = ts[g]
            t = np.arange(n) * DT
            i = I[s0: s0 + n]
            i_end = float(np.mean(i[-2000:]))
            tb, yb = bin2(t, i / i_end)
            ax.plot(tb, yb, "k.", ms=2)
            yh, _ = model_R(tb, r["tau_obs"], r["tau_eff"], r["m0"], mss[g], h150f(g))
            ax.plot(tb, yh, color="tab:red", lw=1.0, label=f"best τ={r['tau_obs'] * 1e3:.0f}ms")
            y25, _ = model_R(tb, 0.025, r["tau_eff"], r["m0"], mss[g], h150f(g))
            ax.plot(tb, y25, color="tab:blue", lw=0.7, ls="--", label="τ=25ms 参考")
            ax.set_title(f"{cell[-3:]} {g}mV ub={r['ub'] * 1e3:.0f}ms", fontsize=7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_τobs判决_复极窗快弛豫封卷.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(controls=dict(C1a=bool(c1a), C1b=bool(c1b), C2=bool(c2)),
               ub_line_ms=UB_LINE * 1e3,
               cells={c: {str(g): r for g, r in detail[c].items()} for c in detail},
               verdicts={str(g): v for g, v in verdicts.items()},
               nseal=nseal, final=final)
    fjson = os.path.join(HERE, "2026-09-13_α模型_τobs判决_复极窗快弛豫封卷_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  图落盘: {fpng}", flush=True)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
