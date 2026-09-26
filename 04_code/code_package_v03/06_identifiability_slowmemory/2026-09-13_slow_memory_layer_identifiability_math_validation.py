# -*- coding: utf-8 -*-
# 2026-09-13_慢记忆层_可识别性数学验证.py
# 目的：纯数学验证（不用任何实验数据，秒级）——
#   "快 Markov 层 x 慢记忆层"混合结构的可观测映射有没有等价族（简并）。
# 五个探针：
#   T1 核族混淆矩阵：拉伸指数核不同 (beta,tau) 在观测窗内能否互相冒充
#   T2 离散 5 指数核对连续核的逼近残差（等价族存在性的直接度量）
#   T3 乘积层吸收测试：慢核参数错时，快层振幅能否把残差吸回零
#   T4 Jacobian SVD：混合参数在 2s 窗 / 90s 窗 / 双窗下的零方向计数
#   T5 汇总判词
# 判读标准（事先写死，不做事后调整）：
#   T1: 窗内曲线相对差 <1e-3 且参数相对差 >30% -> 记"该窗存在简并方向"
#   T2: 5 指数逼近残差 <1e-3 -> 记"离散核与连续核在窗内不可区分"
#   T3: 错参慢核+自由快层残差 <真值的 1% -> 记"乘积层存在吸收简并"
#   T4: 奇异值 < 最大奇异值*1e-8 -> 记零方向
# 运行：python 本文件
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "2026-09-13_慢记忆层_可识别性数学验证_结果.json")
OUT_PNG = os.path.join(HERE, "2026-09-13_慢记忆层_可识别性数学验证.png")

TAUS5 = np.logspace(1.0, np.log10(300.0), 5)   # 193 战役的固定慢核网格（秒）


# ---------- 核函数 ----------
def kernel_stretched(t, beta, tau):
    """拉伸指数记忆核的生存/弛豫形式：m(t)=exp(-(t/tau)^beta)，beta=1 退回普通指数"""
    return np.exp(-(t / tau) ** beta)


def kernel_powerlaw(t, gamma, tau):
    """截断幂律：m(t)=(1+t/tau)^(-gamma)"""
    return (1.0 + t / tau) ** (-gamma)


def kernel_discrete5(t, w):
    """战役旧离散核：sum_k w_k exp(-t/tau_k)，sum w = 1"""
    w = np.asarray(w)
    return np.sum(w[:, None] * np.exp(-t[None, :] / TAUS5[:, None]), axis=0)


def rel_diff(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-300))


# ---------- T1：核族混淆矩阵 ----------
def T1():
    windows = {"2s窗": np.linspace(1e-3, 2.0, 2000), "90s窗": np.linspace(1e-3, 92.9, 4000)}
    betas = np.array([0.30, 0.45, 0.60, 0.75, 0.90, 1.00])
    taus = np.array([3.0, 10.0, 30.0, 100.0, 300.0])
    out = {}
    worst = {}
    for wname, t in windows.items():
        n_bad = 0
        n_pair = 0
        for i, b1 in enumerate(betas):
            for j, t1 in enumerate(taus):
                k1 = kernel_stretched(t, b1, t1)
                for b2 in betas:
                    for t2 in taus:
                        if abs(b2 - b1) < 1e-12 and abs(t2 - t1) < 1e-12:
                            continue
                        # 参数相对差
                        pd = max(abs(b2 - b1) / b1, abs(t2 - t1) / t1)
                        if pd < 0.30:
                            continue
                        k2 = kernel_stretched(t, b2, t2)
                        d = rel_diff(k1, k2)
                        n_pair += 1
                        if d < 1e-3:
                            n_bad += 1
                            worst.setdefault(wname, []).append((round(float(b1), 2), float(t1), round(float(b2), 2), float(t2), round(d, 6)))
        out[wname] = {"参数差>30%且曲线差<1e-3 的对数": n_bad, "检测总对数": n_pair}
    return out, worst


# ---------- T2：离散 5 核逼近连续核 ----------
def T2():
    t = np.logspace(-3, np.log10(92.9), 3000)
    A = np.exp(-t[:, None] / TAUS5[None, :])   # 设计矩阵
    res = {}
    targets = {
        "拉伸指数 beta=0.5, tau=30": kernel_stretched(t, 0.5, 30.0),
        "拉伸指数 beta=0.7, tau=100": kernel_stretched(t, 0.7, 100.0),
        "幂律 gamma=0.8, tau=50": kernel_powerlaw(t, 0.8, 50.0),
    }
    for name, y in targets.items():
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = w / w.sum()
        y_hat = A @ w
        res[name] = {"逼近残差(相对最大值)": round(rel_diff(y, y_hat), 6),
                     "最优权重": [round(float(x), 4) for x in w]}
    return res


# ---------- T3：乘积层吸收测试 ----------
def T3():
    t = np.linspace(1e-3, 2.0, 5000)
    # 真模型：快层双指数 x 慢层拉伸指数
    fast_true = lambda t, a1, t1, a2, t2: a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)
    y_true = fast_true(t, 0.6, 0.02, 0.4, 0.15) * kernel_stretched(t, 0.55, 40.0)
    # 错慢核（beta 错）：快层 4 振幅参数自由（双指数振幅自由），用网格+lstsq
    # 快层基：tau 网格上 6 个指数，振幅非负自由
    tau_grid = np.array([0.005, 0.015, 0.04, 0.12, 0.35, 1.0])
    F = np.exp(-t[:, None] / tau_grid[None, :])
    floors = {}
    for beta_wrong, tau_wrong in [(0.55, 40.0), (0.8, 40.0), (0.35, 40.0), (0.55, 8.0)]:
        slow_w = kernel_stretched(t, beta_wrong, tau_wrong)
        M = F * slow_w[:, None]
        c, *_ = np.linalg.lstsq(M, y_true, rcond=None)
        c = np.clip(c, 0, None)
        r = rel_diff(M @ c, y_true)
        floors[f"beta={beta_wrong}, tau={tau_wrong}"] = round(r, 6)
    floors["真参残差基准"] = floors.pop("beta=0.55, tau=40.0")
    return floors


# ---------- T4：Jacobian SVD ----------
def hybrid_response(t, p):
    """混合结构可观测（省略 G 与 (V-E)：只留形状）
    快层：p_A^4 用双指数代理 a1 e^{-t/tf1}+a2 e^{-t/tf2}（a1+a2=1）
    慢层：拉伸指数 beta,tau_s
    参数：tf1, tf2, a1, beta, tau_s（5 个）"""
    tf1, tf2, a1, beta, tau_s = p
    fast = a1 * np.exp(-t / tf1) + (1 - a1) * np.exp(-t / tf2)
    return fast * kernel_stretched(t, beta, tau_s)


def T4():
    p0 = np.array([0.018, 0.149, 0.45, 0.55, 40.0])   # 战役实测量级（18ms/149ms + 慢核）
    names = ["tau_f1(快)", "tau_f2(中)", "a1(快振幅)", "beta(慢谱宽)", "tau_s(慢尺度)"]
    windows = {
        "仅 2s 窗": np.linspace(1e-3, 2.0, 4000),
        "仅 90s 窗": np.logspace(-3, np.log10(92.9), 4000),
        "2s+90s 双窗": None,
    }
    out = {}
    sv_store = {}
    for wname, t in windows.items():
        if t is None:
            t1 = np.linspace(1e-3, 2.0, 2000)
            t2 = np.logspace(-0.5, np.log10(92.9), 2000)
            J1 = _jacobian(p0, t1)
            J2 = _jacobian(p0, t2)
            J = np.vstack([J1, J2])
        else:
            J = _jacobian(p0, t)
        # 列归一（参数尺度差异）
        col = np.linalg.norm(J, axis=0) + 1e-300
        sv = np.linalg.svd(J / col, compute_uv=False)
        rank = int(np.sum(sv > sv[0] * 1e-8))
        out[wname] = {"奇异值": [round(float(x), 4) for x in sv],
                      "有效秩/总参数": f"{rank}/{len(p0)}",
                      "零方向数": len(p0) - rank}
        sv_store[wname] = sv
    return out, names, sv_store


def _jacobian(p0, t):
    J = np.zeros((len(t), len(p0)))
    f0 = hybrid_response(t, p0)
    for j in range(len(p0)):
        h = 1e-6 * max(abs(p0[j]), 1e-6)
        p1 = p0.copy()
        p1[j] += h
        J[:, j] = (hybrid_response(t, p1) - f0) / h
    return J


# ---------- 主程序 ----------
def main():
    print("=" * 64)
    print(" 慢记忆层可识别性数学验证（纯数学，无实验数据）")
    print("=" * 64, flush=True)

    print("\n[T1] 拉伸指数核族内混淆（参数差>30% 且曲线差<1e-3 记简并）", flush=True)
    t1, worst = T1()
    for wname, d in t1.items():
        print(f"  {wname}: 简并 {d['参数差>30%且曲线差<1e-3 的对数']} / {d['检测总对数']} 对")
    t1_verdict = {}
    for wname, d in t1.items():
        t1_verdict[wname] = "存在简并方向" if d["参数差>30%且曲线差<1e-3 的对数"] > 0 else "未见简并"
        print(f"    -> {wname}: {t1_verdict[wname]}")

    print("\n[T2] 离散 5 核逼近连续核（残差<1e-3 记不可区分）", flush=True)
    t2 = T2()
    t2_verdict = {}
    for name, d in t2.items():
        flag = "窗内不可区分" if d["逼近残差(相对最大值)"] < 1e-3 else "可区分"
        t2_verdict[name] = flag
        print(f"  {name}: 残差 {d['逼近残差(相对最大值)']} -> {flag}  权重 {d['最优权重']}")

    print("\n[T3] 乘积层吸收（错慢核 + 自由快层 能否吸回零）", flush=True)
    t3 = T3()
    base = t3["真参残差基准"]
    t3_verdict = {}
    print(f"  真参残差基准: {base}")
    for k, v in t3.items():
        if k == "真参残差基准":
            continue
        flag = "存在吸收简并" if v < max(base * 100, 1e-3) else "不可吸收"
        t3_verdict[k] = flag
        print(f"  错参 {k}: 残差 {v} -> {flag}")

    print("\n[T4] Jacobian SVD 零方向（判线：奇异值 < max*1e-8）", flush=True)
    t4, names, sv_store = T4()
    for wname, d in t4.items():
        print(f"  {wname}: 有效秩 {d['有效秩/总参数']}，零方向 {d['零方向数']}，奇异值 {d['奇异值']}")

    print("\n[T5] 汇总判词", flush=True)
    lines = []
    lines.append(f"T1: 2s窗[{t1_verdict['2s窗']}]  90s窗[{t1_verdict['90s窗']}]")
    n_indist = sum(1 for v in t2_verdict.values() if v == "窗内不可区分")
    lines.append(f"T2: 3 个连续核目标中 {n_indist} 个被离散5核窗内冒充")
    n_absorb = sum(1 for v in t3_verdict.values() if v == "存在吸收简并")
    lines.append(f"T3: 3 组错参中 {n_absorb} 组被快层吸收（乘积层简并）")
    for wname, d in t4.items():
        lines.append(f"T4 {wname}: 秩 {d['有效秩/总参数']}")
    for L in lines:
        print("  " + L)

    overall = ("混合结构可识别" if (t1_verdict["90s窗"] == "未见简并" and n_absorb == 0
               and t4["2s+90s 双窗"]["零方向数"] == 0)
               else "存在简并方向——见上面逐条")
    print(f"\n  总判词：{overall}")

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    t = np.logspace(-3, 2, 2000)
    for b in [0.3, 0.5, 0.7, 1.0]:
        ax.semilogx(t, kernel_stretched(t, b, 30.0), label=f"beta={b}, tau=30")
    ax.set_title("T1 拉伸指数核族（tau=30s, 变 beta）", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.set_ylabel("m(t)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    t2 = np.logspace(-3, np.log10(92.9), 3000)
    y = kernel_stretched(t2, 0.5, 30.0)
    A = np.exp(-t2[:, None] / TAUS5[None, :])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = np.clip(w, 0, None); w /= w.sum()
    ax.semilogx(t2, y, 'k-', lw=2, label="真 拉伸指数 beta=0.5")
    ax.semilogx(t2, A @ w, 'r--', lw=1.5, label="离散5核最优逼近")
    ax.set_title("T2 离散核冒充连续核", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for wname, sv in sv_store.items():
        ax.semilogy(range(1, len(sv) + 1), sv / sv[0], 'o-', label=wname)
    ax.axhline(1e-8, color='r', ls='--', label='零方向判线 1e-8')
    ax.set_title("T4 奇异值谱（列归一）", fontweight="bold")
    ax.set_xlabel("序号"); ax.set_ylabel("sigma/sigma_max"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    y0 = 0.95
    for L in ["可识别性验证判词", ""] + lines + ["", f"总判词：{overall}"]:
        ax.text(0.03, y0, L, fontsize=11, fontweight="bold" if ("判词" in L) else "normal", wrap=True)
        y0 -= 0.085
    fig.suptitle("慢记忆层 · 可识别性数学验证（无数据，纯结构分析）", fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n  图落盘: {OUT_PNG}")

    results = {"T1": t1, "T1_verdict": t1_verdict, "T1_worst_pairs": worst,
               "T2": t2, "T2_verdict": t2_verdict,
               "T3": t3, "T3_verdict": t3_verdict,
               "T4": t4, "param_names": names,
               "overall": overall}
    def _jdefault(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=_jdefault)
    print(f"  结果落盘: {OUT_JSON}")


if __name__ == "__main__":
    main()
