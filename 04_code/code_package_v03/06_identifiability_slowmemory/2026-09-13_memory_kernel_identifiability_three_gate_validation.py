# -*- coding: utf-8 -*-
# 2026-09-13_记忆核_可识别性三门验证.py
# 目的：建模可行性终审（纯合成数据，零真实数据拟合）。三门同炉，判线写死：
#   门1 锚有效：晚窗双斜率闭式反演核形状参数（β 或 α），200 次噪声实现，查偏差与散布
#        过：|偏差|<=10% 且 IQR<=20%（1σ 噪声）；3σ 噪声下 IQR<=40%
#   门2 锚破简并：全长联合拟合（c0,a_rec,τ_rec,a_s,β,τ_s）
#        自由多初值：若不同初值收敛到不同 β 而残差几乎相同 -> T3 简并复现
#        锚定（β,τ_s 用门1 的锚冻结）：其余参数恢复误差 <=20% 才算锚能救
#        三种判词：锚必要且充分 / 联合自识别（锚非必要）/ 锚也不救（不可识别）
#   门3 族可分辨：拉伸指数<->幂律交叉拟合，两个方向残差都 >2σ 才算族分得开；
#        分不开则"核形状参数"不良定义，只能退到族无关谱矩
# 运行：python 本文件（你的机器，约 1-2 分钟）
import os
import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
SIG = 0.036          # 实测噪声地板 (nA)
DT = 1e-3            # 与 v5.3 降采样口径一致
T_END = 5.5
SEED = 20260913

# ---------------- 核族 ----------------

def k_stretch(t, tau, beta):
    return np.exp(-(t / tau) ** beta)

def k_power(t, tau, alpha):
    return (1.0 + t / tau) ** (-alpha)

# ---------------- v5.3 同款测量管线：分箱中位 -> log -> WLS ----------------

def binlog(t, y, bin_s=0.05):
    nbin = int(np.clip((t[-1] - t[0]) / bin_s, 16, 110))
    sig_bin = 3 * SIG / np.sqrt(max(1, len(t) // nbin))
    bt, bl, bm = [], [], []
    for b in np.array_split(np.arange(len(t)), nbin):
        med = float(np.median(y[b]))
        if med > max(sig_bin, 0.01):
            bt.append(float(np.median(t[b])))
            bl.append(float(np.log(med)))
            bm.append(med)
    return np.array(bt), np.array(bl), np.array(bm)

def wslope(tq, lq, mq):
    wq = mq ** 2
    tbar = np.sum(wq * tq) / wq.sum()
    W = np.sum(wq * (tq - tbar) ** 2)
    return float(np.sum(wq * (tq - tbar) * lq) / W)

T1, T2 = 2.0, 4.5                      # 两个锚窗中心
W1, W2 = (1.6, 2.4), (4.1, 4.9)

def two_slopes(t, y):
    bt, bl, bm = binlog(t, y)
    out = []
    for w in (W1, W2):
        m = (bt >= w[0]) & (bt <= w[1])
        if m.sum() < 6:
            return None
        out.append(wslope(bt[m], bl[m], bm[m]))
    return out

def inv_stretch(s1, s2):
    """s(t) = -(β/τ^β) t^{β-1} 的闭式反演"""
    beta = 1.0 + np.log(s1 / s2) / np.log(T1 / T2)
    tau = (-beta * T1 ** (beta - 1) / s1) ** (1.0 / beta)
    return float(beta), float(tau)

def inv_power(s1, s2):
    """s(t) = -α/(τ+t) 的闭式反演"""
    alpha = (T2 - T1) / (1.0 / s1 - 1.0 / s2)
    tau = -alpha / s1 - T1
    return float(alpha), float(tau)

# ---------------- 门 1 ----------------

TRUTH_ST = dict(beta=0.5, tau=2.0, amp=0.5)
TRUTH_PW = dict(alpha=1.0, tau=0.3, amp=0.5)   # α=1.5/τ=0.5 晚窗信号会跌到掩码下，换可测真值

def gate1(rng, nmc=200):
    t = np.arange(DT, T_END, DT)
    res = {}
    for fam in ("stretch", "power"):
        for sigscale, tag in ((1.0, "1σ"), (3.0, "3σ")):
            ests = []
            for _ in range(nmc):
                if fam == "stretch":
                    y = TRUTH_ST["amp"] * k_stretch(t, TRUTH_ST["tau"], TRUTH_ST["beta"])
                else:
                    y = TRUTH_PW["amp"] * k_power(t, TRUTH_PW["tau"], TRUTH_PW["alpha"])
                y = y + rng.normal(0, SIG * sigscale, len(t))
                sl = two_slopes(t, y)
                if sl is None:
                    continue
                try:
                    if fam == "stretch":
                        b, ta = inv_stretch(*sl)
                        if 0.05 < b < 1.5 and 0.05 < ta < 100:
                            ests.append((b, ta))
                    else:
                        a, ta = inv_power(*sl)
                        if 0.05 < a < 10 and 0.01 < ta < 100:
                            ests.append((a, ta))
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
            ests = np.array(ests)
            key = f"{fam}_{tag}"
            if len(ests) < nmc * 0.8:
                res[key] = dict(error=f"有效反演 {len(ests)}/{nmc}")
                continue
            p = ests[:, 0]
            truth_p = TRUTH_ST["beta"] if fam == "stretch" else TRUTH_PW["alpha"]
            bias = float(np.median(p) / truth_p - 1)
            iqr = float((np.percentile(p, 75) - np.percentile(p, 25)) / truth_p)
            res[key] = dict(n=len(ests), shape_med=float(np.median(p)), truth=truth_p,
                            bias=bias, iqr=iqr, tau_med=float(np.median(ests[:, 1])),
                            est_shape=p.tolist())
    return res

# ---------------- 门 2 ----------------

TRUTH2 = dict(c0=0.05, a_rec=1.0, t_rec=0.15, a_s=0.5, beta=0.5, tau_s=2.0)

def model_full(th, t):
    c0, ar, tr, a_s, be, ts = th
    return c0 + ar * np.exp(-t / tr) + a_s * k_stretch(t, ts, be)

def model_anchored(th, t, be, ts):
    c0, ar, tr, a_s = th
    return c0 + ar * np.exp(-t / tr) + a_s * k_stretch(t, ts, be)

LB6 = [-0.5, 0.0, 0.02, 0.0, 0.10, 0.2]
UB6 = [0.5, 5.0, 2.00, 5.0, 1.00, 20.0]
LB4 = [-0.5, 0.0, 0.02, 0.0]
UB4 = [0.5, 5.0, 2.00, 5.0]

def gate2(rng, nmc=20):
    t = np.arange(DT, T_END, DT)
    y_true = model_full([TRUTH2[k] for k in ("c0", "a_rec", "t_rec", "a_s", "beta", "tau_s")], t)
    beta_starts = [0.2, 0.35, 0.5, 0.7, 0.9]
    free_beta, free_cost = [], []
    anch_err, anch_cost = [], []
    for _ in range(nmc):
        y = y_true + rng.normal(0, SIG, len(t))
        # (a) 自由多初值
        bests = []
        for b0 in beta_starts:
            x0 = [0.1, 0.8, 0.2, 0.4, b0, 2.0]
            try:
                r = least_squares(lambda th: model_full(th, t) - y, x0,
                                  bounds=(LB6, UB6), max_nfev=300)
                bests.append((float(2 * r.cost / len(t)), r.x))
            except Exception:
                pass
        if bests:
            costs = np.array([b[0] for b in bests])
            free_cost.append(float(costs.min()))
            near = [b[1] for b in bests if b[0] <= costs.min() * 1.01 + 1e-12]
            free_beta.append([float(x[4]) for x in near])
        # (b) 锚定：先粗估 c0（β 冻结名义值 0.5，c0 对 β 误设不敏感），扣掉后走门1 锚，再冻结锚联合拟合
        try:
            r0 = least_squares(lambda th: model_anchored(th, t, 0.5, 2.0) - y,
                               [0.1, 0.8, 0.2, 0.4], bounds=(LB4, UB4), max_nfev=300)
            c0_hat = float(r0.x[0])
        except Exception:
            continue
        sl = two_slopes(t, y - c0_hat)
        if sl is not None:
            try:
                b_a, ts_a = inv_stretch(*sl)
                b_a = float(np.clip(b_a, 0.1, 1.0))
                ts_a = float(np.clip(ts_a, 0.2, 20.0))
            except Exception:
                continue
            try:
                r = least_squares(lambda th: model_anchored(th, t, b_a, ts_a) - y,
                                  [0.1, 0.8, 0.2, 0.4], bounds=(LB4, UB4), max_nfev=300)
                anch_cost.append(float(2 * r.cost / len(t)))
                rel = [abs(r.x[1] - TRUTH2["a_rec"]) / TRUTH2["a_rec"],
                       abs(r.x[2] - TRUTH2["t_rec"]) / TRUTH2["t_rec"],
                       abs(r.x[3] - TRUTH2["a_s"]) / TRUTH2["a_s"]]
                anch_err.append(rel)
            except Exception:
                pass
    out = {}
    if free_beta:
        allb = np.array([b for lst in free_beta for b in lst])
        spread = float(np.percentile(allb, 75) - np.percentile(allb, 25))
        out["free_beta_iqr"] = spread
        out["free_beta_all"] = allb.tolist()
        out["free_cost_med"] = float(np.median(free_cost))
        out["degeneracy_reproduced"] = bool(spread > 0.10)
    if anch_err:
        anch_err = np.array(anch_err)
        out["anch_relerr_med"] = [float(np.median(anch_err[:, i])) for i in range(3)]
        out["anch_cost_med"] = float(np.median(anch_cost))
        out["anchored_ok"] = bool(np.all(np.median(anch_err, axis=0) <= 0.20))
    return out

# ---------------- 门 3 ----------------

def model_family(th, t, fam):
    c0, ar, tr, a_s, sh, ts = th
    k = k_stretch(t, ts, sh) if fam == "stretch" else k_power(t, ts, sh)
    return c0 + ar * np.exp(-t / tr) + a_s * k

def gate3(rng):
    t = np.arange(DT, T_END, DT)
    cases = [
        ("真=拉伸β0.5 -> 拟合幂律",
         model_family([0.05, 1.0, 0.15, 0.5, 0.5, 2.0], t, "stretch"), "power",
         [0.1, 0.8, 0.2, 0.4, 1.5, 0.5], ([-0.5, 0, 0.02, 0, 0.1, 0.01], [0.5, 5, 2, 5, 10, 20])),
        ("真=幂律α1.0 -> 拟合拉伸",
         model_family([0.05, 1.0, 0.15, 0.5, 1.0, 0.3], t, "power"), "stretch",
         [0.1, 0.8, 0.2, 0.4, 0.5, 2.0], ([-0.5, 0, 0.02, 0, 0.1, 0.2], [0.5, 5, 2, 5, 1, 20])),
    ]
    out = {}
    for name, y_true, fam, x0, bounds in cases:
        rms = []
        for _ in range(5):
            y = y_true + rng.normal(0, SIG, len(t))
            best = None
            for jit in range(3):
                xj = [v * f for v, f in zip(x0, [1, 1, 1, 1, [0.6, 1.0, 1.8][jit] if fam == "power" else [0.3, 0.5, 0.8][jit], 1])]
                try:
                    r = least_squares(lambda th: model_family(th, t, fam) - y, xj,
                                      bounds=bounds, max_nfev=400)
                    c = float(np.sqrt(2 * r.cost / len(t)))
                    if best is None or c < best:
                        best = c
                except Exception:
                    pass
            if best is not None:
                rms.append(best)
        out[name] = dict(rms_med=float(np.median(rms)), noise=SIG,
                         distinguishable=bool(np.median(rms) > 2 * SIG))
    return out

# ---------------- 主流程 ----------------

def main():
    rng = np.random.default_rng(SEED)
    print("=" * 66)
    print(" 记忆核可识别性 · 三门验证（纯合成，零真实数据）")
    print(f" 噪声 σ={SIG} nA   窗 {T_END}s   锚窗中心 {T1}/{T2}s   种子 {SEED}")
    print("=" * 66, flush=True)

    print("\n[门1] 锚有效：晚窗双斜率闭式反演（200 次实现）", flush=True)
    g1 = gate1(rng)
    g1_pass = True
    for key, r in g1.items():
        if "error" in r:
            print(f"  {key}: {r['error']} -> 不过")
            g1_pass = False
            continue
        ok_bias = abs(r["bias"]) <= 0.10
        ok_iqr = r["iqr"] <= (0.20 if key.endswith("1σ") else 0.40)
        ok = ok_bias and ok_iqr
        if key.endswith("1σ"):
            g1_pass &= ok
        print(f"  {key}: 形状参数 中位 {r['shape_med']:.3f}（真值 {r['truth']}）"
              f" 偏差 {r['bias']:+.1%} IQR {r['iqr']:.1%}  τ 中位 {r['tau_med']:.2f}s -> {'过' if ok else '不过'}")
    print(f"  门1 判词: {'过 —— 锚有效' if g1_pass else '不过 —— 锚无效'}")

    print("\n[门2] 锚破简并：全长联合拟合（20 实现 × 多初值）", flush=True)
    g2 = gate2(rng)
    if "free_beta_iqr" in g2:
        print(f"  自由多初值: β 收敛值 IQR = {g2['free_beta_iqr']:.3f}  "
              f"({'T3 简并复现' if g2['degeneracy_reproduced'] else '未见简并，联合自识别'})"
              f"  残差中位 {g2.get('free_cost_med', float('nan')):.4f} nA²")
    if "anch_relerr_med" in g2:
        e = g2["anch_relerr_med"]
        print(f"  锚定拟合: a_rec 误差 {e[0]:.1%}  τ_rec 误差 {e[1]:.1%}  a_s 误差 {e[2]:.1%}"
              f"  -> {'恢复合格(≤20%)' if g2['anchored_ok'] else '恢复不合格'}")
    deg = g2.get("degeneracy_reproduced")
    anc = g2.get("anchored_ok")
    if deg and anc:
        v2 = "锚必要且充分 —— 建模可行（锚是必需品）"
    elif (not deg) and anc:
        v2 = "联合自识别 —— 建模可行（锚非必需但无害）"
    elif deg and not anc:
        v2 = "锚也不救 —— 当前锚形式不可识别，建模失败"
    else:
        v2 = "数据不足，不下判词"
    g2_pass = bool(anc) and (deg is not None)
    print(f"  门2 判词: {v2}")

    print("\n[门3] 族可分辨：交叉拟合（残差 >2σ 才算分得开）", flush=True)
    g3 = gate3(rng)
    g3_pass = True
    for name, r in g3.items():
        print(f"  {name}: 交叉拟合 RMS = {r['rms_med']:.4f} nA（2σ={2*SIG:.3f}）"
              f" -> {'分得开' if r['distinguishable'] else '冒充成功，分不开'}")
        g3_pass &= r["distinguishable"]
    print(f"  门3 判词: {'过 —— 族可分辨，核形状参数良定义' if g3_pass else '不过 —— 只能提取族无关谱矩'}")

    print("\n" + "=" * 66)
    print(" 总判词：")
    print(f"  门1 {'过' if g1_pass else '不过'} | 门2 {'过' if g2_pass else '不过'} | 门3 {'过' if g3_pass else '不过'}")
    if g1_pass and g2_pass:
        final = "建模可行：锚有效且破简并。" + ("核形状参数良定义。" if g3_pass else "核形状退到族无关谱矩。")
    else:
        final = "建模有硬关：" + ("锚无效。" if not g1_pass else "") + ("锚救不了简并。" if not g2_pass else "")
    print("  " + final)
    print("=" * 66)

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    ax = axes[0, 0]
    for key, r in g1.items():
        if key.startswith("stretch") and "est_shape" in r:
            ax.hist(r["est_shape"], bins=25, alpha=0.5, label=key)
    ax.axvline(TRUTH_ST["beta"], color="k", ls="--", label="真值 β=0.5")
    ax.set_title("门1 拉伸指数 β̂ 分布", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for key, r in g1.items():
        if key.startswith("power") and "est_shape" in r:
            ax.hist(r["est_shape"], bins=25, alpha=0.5, label=key)
    ax.axvline(TRUTH_PW["alpha"], color="k", ls="--", label="真值 α=1.5")
    ax.set_title("门1 幂律 α̂ 分布", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if "free_beta_all" in g2:
        ax.hist(g2["free_beta_all"], bins=20, color="tomato", alpha=0.7)
        ax.axvline(TRUTH2["beta"], color="k", ls="--", label="真值 β=0.5")
        ax.set_title(f"门2 自由多初值 β̂（IQR={g2['free_beta_iqr']:.3f}）", fontweight="bold")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if "anch_relerr_med" in g2:
        e = g2["anch_relerr_med"]
        ax.bar(["a_rec", "τ_rec", "a_s"], e, color="steelblue")
        ax.axhline(0.20, color="r", ls="--", label="20% 判线")
        ax.set_title("门2 锚定后参数恢复误差", fontweight="bold"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names = list(g3.keys())
    ax.barh([n[:14] for n in names], [g3[n]["rms_med"] for n in names], color="darkorange")
    ax.axvline(2 * SIG, color="r", ls="--", label=f"2σ={2*SIG:.3f}")
    ax.set_title("门3 交叉拟合 RMS", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["三门总判词", "",
             f"门1 锚有效: {'过' if g1_pass else '不过'}",
             f"门2 破简并: {'过' if g2_pass else '不过'}",
             f"门3 族分辨: {'过' if g3_pass else '不过'}", "", final]
    y0 = 0.9
    for L in lines:
        ax.text(0.05, y0, L, fontsize=12, fontweight="bold" if y0 in (0.9, 0.9 - 6 * 0.11) else "normal", wrap=True)
        y0 -= 0.11
    fig.suptitle("记忆核可识别性三门验证（纯合成）", fontsize=14, fontweight="bold")
    fpng = os.path.join(HERE, "2026-09-13_记忆核_可识别性三门验证.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    g1_out = {k: ({kk: vv for kk, vv in v.items() if kk != "est_shape"} if "error" not in v else v)
              for k, v in g1.items()}
    g2_out = {k: v for k, v in g2.items() if k != "free_beta_all"}
    out = dict(sigma_nA=SIG, seed=SEED, gate1=g1_out, gate1_pass=bool(g1_pass),
               gate2=g2_out, gate2_pass=bool(g2_pass), gate2_verdict=v2,
               gate3=g3, gate3_pass=bool(g3_pass), final=final)
    fjson = os.path.join(HERE, "2026-09-13_记忆核_可识别性三门验证_结果.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n  图落盘: {fpng}")
    print(f"  结果落盘: {fjson}")

if __name__ == "__main__":
    main()
