# 2026-09-13_α模型_纯hssV剥离判决.py
# 目的（缺口 1 主攻：纯 h_ss(V) 曲线，剥离 m 污染）：
#   失活协议每周期：+50x0.6s 全激活全失活 -> -90x0.06s 复位（h 全恢复、m 残留
#   ~0.43）-> 测试档 Vx0.150s（-100..+50 家族）-> -120x0.5s 反弹 DoE。
#   剥离推导（零 τ 依赖、G 精确抵消）：
#     测试段末电流 I_test = G·m_T·h_T·(V-E_rev)（τ_h(V)<<150ms 档 h_T≈h_ss(V)，
#       τ_h 较慢档 h_T 为 h_ss 上界，声明）；
#     反弹 DoE 幅度 A = G·DF120·m_T（I_reb=G·m_T·DF·[e^{-t/τd}-(1-h_T)e^{-t/τr}]，
#       DoE 拟合 A 对任意 h_T 均 ≈G·DF·m_T，已推导）；
#     => h_ss(V) ≈ I_test·DF120 / (A·(V-E_rev))，DF120=31.67mV，G 抵消。
#   -90 档 DF=-1.67mV 除零弃用（协议该档为 -90x0.21 连写，无 0.15s 测试段）。
#
# 【协议结构（2026-09-13 程序化解析在案）】
#   全长 46.41s；周期 = [-120x0.05, -80x0.2, +50x0.6, -90x0.06, 测试Vx0.150,
#   -120x0.5, -80x1.34]；测试档家族 -100..+50（acurve 16 档，-100/-90/-80 n 少）。
#
# 判线（跑前声明）：
#   QC1（物理门）：h_ss(V) ∈ [-0.1, 1.3] 且 |I_test|>=4σ（σ=该细胞 -80x1.34s
#     基线 std），否则该细胞该档剔除记旗（D1/D5 过减漏家族预期在正档中旗）。
#   QC2（B1 生死判）：-100..-50 各有效档 h_ss 中位 ∈[0.7,1.3] -> B1（h_ss=1
#     物理声明）支持；任一档中位 <0.7 -> B1 该档死亡，实测曲线替代声明。
#   QC3（正档一致性）：+40 档剥离 h_ss 与 §3.2 h40 九细胞值（0.0023-0.0296）
#     中位比 ∈ [0.3,3]。
#   封卷判线（每档独立）：QC1 过细胞数 >=5 且 CV<0.3 -> 该档【封卷】；
#     >=5 但 CV>=0.3 ->【登记】；<5 ->【数据不足】。
# 对照（跑前声明）：
#   C1 DoE 反演 τr 3.5ms 误差<15%；
#   C2 合成周期（m_T=0.5, h_T=0.3, G=0.1, V=-20, 噪声=16713003 实测 σ）
#     剥离公式回收 h_T 误差<10% -> 不过则统计量作废停。
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
DF120 = abs(-120.0 - E_REV)

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
SEP_MIN = 1.5


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


def strip_cell(V, I):
    """每细胞：逐测试档剥离 h_ss。返回 {gear: dict(I_test, A, h, qc, flag)}, σ。"""
    info = segments(V)
    sig = []
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT > 1.0:
            sig.append(np.std(I[s0 + 2000: s0 + n]))
    sigma = float(np.median(sig)) if sig else np.nan
    out = {}
    for i, (v, s0, n) in enumerate(info):
        # 测试档：0.14-0.16s，后接 -120x0.5s，前两段内有 -90 复位
        if not (0.14 < n * DT < 0.16):
            continue
        if i + 1 >= len(info) or not (abs(info[i + 1][0] + 120) < 2
                                      and info[i + 1][2] * DT > 0.4):
            continue
        prev90 = any(abs(info[j][0] + 90) < 2 for j in range(max(0, i - 2), i))
        if not prev90 or abs(v + 90) < 2:                    # -90 档 DF≈0 弃用
            continue
        i_test = float(np.mean(I[s0 + n - 2000: s0 + n]))  # 测试段末 20ms
        a, b = info[i + 1][1], info[i + 1][1] + 3000       # 反弹前 300ms
        r = doe_fit(np.arange(3000) * DT, -I[a:b])
        if r is None:
            continue
        df = v - E_REV
        h = i_test * DF120 / (r["A"] * df) if r["A"] != 0 and abs(df) > 1e-6 else np.nan
        qc = bool(np.isfinite(h) and -0.1 <= h <= 1.3
                  and abs(i_test) >= 4 * (sigma or 0) and not r["edge"])
        out[round(v)] = dict(I_test=i_test, A=r["A"], h=float(h), qc=qc,
                             tau_r=r["tau_r"], tau_d=r["tau_d"],
                             flag=None if qc else "qc1_fail")
    return out, sigma


def main():
    rng = np.random.default_rng(7)
    print("=" * 88)
    print(" α模型 纯 h_ss(V) 剥离判决" + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"))
    print(" 判线: 每档 QC1>=5 且 CV<0.3 封卷 | QC2 B1 生死（-100..-50 中位∈[0.7,1.3]）| QC3 +40 一致")
    print("=" * 88, flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    t_c = np.arange(int(0.3 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE 反演 τr: {rc['tau_r'] * 1e3:.2f}ms（真值3.5）-> {'过' if c1 else '不过'}",
          flush=True)
    # C2 合成周期：m_T=0.5 h_T=0.3 G=0.1 V=-20
    G0, mT, hT, vt = 0.1, 0.5, 0.3, -20.0
    i_syn = G0 * mT * hT * (vt - E_REV) + rng.normal(0, 0.02)
    reb = G0 * DF120 * mT * (np.exp(-t_c / 0.030) - (1 - hT) * np.exp(-t_c / 0.003)) \
        + rng.normal(0, 0.02, len(t_c))
    rs = doe_fit(t_c, reb)
    h_rec = i_syn * DF120 / (rs["A"] * (vt - E_REV))
    c2 = bool(rs and abs(h_rec - hT) / hT < 0.10)
    print(f"  C2 合成剥离: h_rec={h_rec:.3f}（真值0.300，A={rs['A']:.3f}）-> "
          f"{'过' if c2 else '不过'}", flush=True)
    if not (c1 and c2):
        print("  对照未归位 -> 统计量作废，停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    res = {}
    all_gears = set()
    for c in CELLS:
        V, I = load_mat("inactivation_protocol.mat", c, "inactivation")
        if I is None:
            print(f"  {c}: 无数据", flush=True)
            continue
        cur, sigma = strip_cell(V, I)
        res[c] = dict(curve=cur, sigma=sigma)
        all_gears.update(cur.keys())
        line = " ".join(f"{g}:{cur[g]['h']:.2f}{'' if cur[g]['qc'] else '×'}"
                        for g in sorted(cur))
        print(f"  {c}: σ={sigma:.4f} | {line}", flush=True)

    gears = sorted(all_gears)
    print("\n" + "-" * 88, flush=True)
    print("[逐档判决]", flush=True)
    gear_verdict = {}
    for g in gears:
        vals = [res[c]["curve"][g]["h"] for c in res
                if g in res[c]["curve"] and res[c]["curve"][g]["qc"]]
        n = len(vals)
        if n >= 2:
            med = float(np.median(vals))
            cv = float(np.std(vals) / abs(np.mean(vals))) if abs(np.mean(vals)) > 1e-9 \
                else np.inf
        else:
            med, cv = (float(vals[0]), np.inf) if n == 1 else (np.nan, np.inf)
        if n >= 5 and cv < 0.3:
            v = "【封卷】"
        elif n >= 5:
            v = "【登记】"
        else:
            v = "【数据不足】"
        gear_verdict[g] = dict(n=n, med=med, cv=cv, verdict=v)
        print(f"  {g:+5d}mV: n={n} 中位={med:.3f} CV={cv:.3f} {v}", flush=True)

    # ---------- QC2 B1 生死 ----------
    b1_gears = [g for g in gears if -100 <= g <= -50 and gear_verdict[g]["n"] >= 5]
    b1_dead = []
    if not b1_gears:
        b1_ok = None
        print("\n[QC2] B1: 有效档不足（n<5），不判", flush=True)
    else:
        b1_dead = [g for g in b1_gears if gear_verdict[g]["med"] < 0.7]
        b1_ok = not b1_dead
        print(f"\n[QC2] B1（h_ss(-100..-50)=1 物理声明）: "
              f"{'支持（各档中位均∈[0.7,1.3]）' if b1_ok else f'死亡档 {b1_dead}'}", flush=True)

    # ---------- QC3 +40 一致性 ----------
    h40_ref = {c: v for c, v in
               [("16704007", 0.0216), ("16704047", 0.0197), ("16707014", 0.0143),
                ("16708016", 0.0042), ("16708060", 0.0022), ("16708118", 0.0296),
                ("16713003", 0.0023), ("16713110", 0.0164), ("16715049", 0.0148)]}
    qc3 = None
    if 40 in gear_verdict and gear_verdict[40]["n"] >= 3:
        ref = float(np.median(list(h40_ref.values())))
        ratio = gear_verdict[40]["med"] / ref if ref else np.nan
        qc3 = bool(0.3 <= ratio <= 3.0)
        print(f"[QC3] +40 剥离中位={gear_verdict[40]['med']:.4f} vs h40 参考中位"
              f"={ref:.4f} 比={ratio:.2f} -> {'一致' if qc3 else '不一致'}", flush=True)
    else:
        print("[QC3] +40 档有效细胞不足，不判", flush=True)

    n_seal = sum(1 for d in gear_verdict.values() if d["verdict"] == "【封卷】")
    n_reg = sum(1 for d in gear_verdict.values() if d["verdict"] == "【登记】")
    b1_txt = "支持" if b1_ok else ("死亡（见上）" if b1_ok is False else "未判")
    print(f"\n 总判词：封卷 {n_seal} 档，登记 {n_reg} 档，"
          f"数据不足 {sum(1 for d in gear_verdict.values() if d['verdict'] == '【数据不足】')} 档；"
          f"B1 {b1_txt}；QC3 "
          f"{'一致' if qc3 else ('不一致' if qc3 is False else '未判')}", flush=True)

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in res:
        gs = sorted(res[c]["curve"])
        hs = [res[c]["curve"][g]["h"] for g in gs]
        qc = [res[c]["curve"][g]["qc"] for g in gs]
        ax.plot([g for g, q in zip(gs, qc) if q], [h for h, q in zip(hs, qc) if q],
                "o-", ms=4, lw=0.8, alpha=0.55, label=c)
        ax.plot([g for g, q in zip(gs, qc) if not q],
                [h for h, q in zip(hs, qc) if not q], "x", ms=5, alpha=0.4)
    meds = [gear_verdict[g]["med"] for g in gears]
    ax.plot(gears, meds, "k-s", lw=2.2, ms=7, label="跨细胞中位")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axhspan(0.7, 1.3, color="tab:green", alpha=0.06, label="B1 判带")
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("h_ss(V) 剥离值")
    ax.set_title("纯 h_ss(V) 剥离（×=QC1 剔除点）")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    fjson = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(dict(cells=res, gears=gear_verdict, b1_ok=b1_ok, b1_dead=b1_dead,
                       qc3=qc3), f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  图落盘: {fpng}", flush=True)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
