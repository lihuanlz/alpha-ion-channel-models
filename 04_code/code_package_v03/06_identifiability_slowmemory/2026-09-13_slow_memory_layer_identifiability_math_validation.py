# -*- coding: utf-8 -*-
# 2026-09-13_slow-memory-layer_identifiability_math_validation.py
# Purpose: pure math validation (no experimental data, seconds) --
#   does the observable map of the "fast Markov layer x slow memory layer" hybrid structure have an equivalence family (degeneracy)?
# Five probes:
#   T1 kernel-family confusion matrix: can stretched-exponential kernels with different (beta,tau) impersonate each other within the observation window
#   T2 approximation residual of a discrete 5-exponential kernel to continuous kernels (direct measure of equivalence-family existence)
#   T3 product-layer absorption test: with a wrong slow kernel, can free fast-layer amplitudes absorb the residual back to zero
#   T4 Jacobian SVD: null-direction count of hybrid parameters in the 2s window / 90s window / dual window
#   T5 summary verdict
# Reading criteria (pinned in advance, no post-hoc adjustment):
#   T1: in-window curve relative difference <1e-3 with parameter relative difference >30% -> record "degenerate direction exists in this window"
#   T2: 5-exponential approximation residual <1e-3 -> record "discrete and continuous kernels indistinguishable in the window"
#   T3: wrong-parameter slow kernel + free fast layer residual <1% of truth -> record "product-layer absorption degeneracy exists"
#   T4: singular value < max singular value *1e-8 -> record null direction
# Run: python this_file
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

TAUS5 = np.logspace(1.0, np.log10(300.0), 5)   # fixed slow-kernel grid of the 193 campaign (seconds)


# ---------- kernel functions ----------
def kernel_stretched(t, beta, tau):
    """Stretched-exponential memory kernel survival/relaxation form: m(t)=exp(-(t/tau)^beta); beta=1 reduces to an ordinary exponential"""
    return np.exp(-(t / tau) ** beta)


def kernel_powerlaw(t, gamma, tau):
    """Truncated power law: m(t)=(1+t/tau)^(-gamma)"""
    return (1.0 + t / tau) ** (-gamma)


def kernel_discrete5(t, w):
    """Legacy discrete kernel of the campaign: sum_k w_k exp(-t/tau_k), sum w = 1"""
    w = np.asarray(w)
    return np.sum(w[:, None] * np.exp(-t[None, :] / TAUS5[:, None]), axis=0)


def rel_diff(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-300))


# ---------- T1: kernel-family confusion matrix ----------
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
                        # parameter relative difference
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


# ---------- T2: discrete 5-kernel approximation of continuous kernels ----------
def T2():
    t = np.logspace(-3, np.log10(92.9), 3000)
    A = np.exp(-t[:, None] / TAUS5[None, :])   # design matrix
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


# ---------- T3: product-layer absorption test ----------
def T3():
    t = np.linspace(1e-3, 2.0, 5000)
    # true model: fast-layer double exponential x slow-layer stretched exponential
    fast_true = lambda t, a1, t1, a2, t2: a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)
    y_true = fast_true(t, 0.6, 0.02, 0.4, 0.15) * kernel_stretched(t, 0.55, 40.0)
    # wrong slow kernel (beta wrong): 4 fast-layer amplitude parameters free (double-exponential amplitudes free), grid + lstsq
    # fast-layer basis: 6 exponentials on a tau grid, non-negative free amplitudes
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
    """Hybrid-structure observable (G and (V-E) omitted: shape only).
    Fast layer: p_A^4 proxied by a double exponential a1 e^{-t/tf1}+a2 e^{-t/tf2} (a1+a2=1)
    Slow layer: stretched exponential beta, tau_s
    Parameters: tf1, tf2, a1, beta, tau_s (5 total)"""
    tf1, tf2, a1, beta, tau_s = p
    fast = a1 * np.exp(-t / tf1) + (1 - a1) * np.exp(-t / tf2)
    return fast * kernel_stretched(t, beta, tau_s)


def T4():
    p0 = np.array([0.018, 0.149, 0.45, 0.55, 40.0])   # campaign measured magnitudes (18ms/149ms + slow kernel)
    names = ["tau_f1(fast)", "tau_f2(mid)", "a1(fast amplitude)", "beta(slow spectral width)", "tau_s(slow scale)"]
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
        # column normalization (parameter scale differences)
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


# ---------- main program ----------
def main():
    print("=" * 64)
    print(" slow-memory-layer identifiability math validation (pure math, no experimental data)")
    print("=" * 64, flush=True)

    print("\n[T1] within-family confusion of stretched-exponential kernels (parameter diff >30% with curve diff <1e-3 counts as degeneracy)", flush=True)
    t1, worst = T1()
    for wname, d in t1.items():
        print(f"  {wname}: 简并 {d['参数差>30%且曲线差<1e-3 的对数']} / {d['检测总对数']} 对")
    t1_verdict = {}
    for wname, d in t1.items():
        t1_verdict[wname] = "存在简并方向" if d["参数差>30%且曲线差<1e-3 的对数"] > 0 else "未见简并"
        print(f"    -> {wname}: {t1_verdict[wname]}")

    print("\n[T2] discrete 5-kernel approximation of continuous kernels (residual <1e-3 counts as indistinguishable)", flush=True)
    t2 = T2()
    t2_verdict = {}
    for name, d in t2.items():
        flag = "窗内不可区分" if d["逼近残差(相对最大值)"] < 1e-3 else "可区分"
        t2_verdict[name] = flag
        print(f"  {name}: 残差 {d['逼近残差(相对最大值)']} -> {flag}  权重 {d['最优权重']}")

    print("\n[T3] product-layer absorption (can wrong slow kernel + free fast layer absorb back to zero)", flush=True)
    t3 = T3()
    base = t3["真参残差基准"]
    t3_verdict = {}
    print(f"  true-parameter residual baseline: {base}")
    for k, v in t3.items():
        if k == "真参残差基准":
            continue
        flag = "存在吸收简并" if v < max(base * 100, 1e-3) else "不可吸收"
        t3_verdict[k] = flag
        print(f"  错参 {k}: 残差 {v} -> {flag}")

    print("\n[T4] Jacobian SVD null directions (criterion: singular value < max*1e-8)", flush=True)
    t4, names, sv_store = T4()
    for wname, d in t4.items():
        print(f"  {wname}: 有效秩 {d['有效秩/总参数']}，零方向 {d['零方向数']}，奇异值 {d['奇异值']}")

    print("\n[T5] summary verdict", flush=True)
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

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    t = np.logspace(-3, 2, 2000)
    for b in [0.3, 0.5, 0.7, 1.0]:
        ax.semilogx(t, kernel_stretched(t, b, 30.0), label=f"beta={b}, tau=30")
    ax.set_title("T1 stretched-exponential kernel family (tau=30s, varying beta)", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.set_ylabel("m(t)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    t2 = np.logspace(-3, np.log10(92.9), 3000)
    y = kernel_stretched(t2, 0.5, 30.0)
    A = np.exp(-t2[:, None] / TAUS5[None, :])
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = np.clip(w, 0, None); w /= w.sum()
    ax.semilogx(t2, y, 'k-', lw=2, label="true stretched exp beta=0.5")
    ax.semilogx(t2, A @ w, 'r--', lw=1.5, label="discrete 5-kernel best approximation")
    ax.set_title("T2 discrete kernel impersonating a continuous kernel", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for wname, sv in sv_store.items():
        ax.semilogy(range(1, len(sv) + 1), sv / sv[0], 'o-', label=wname)
    ax.axhline(1e-8, color='r', ls='--', label='null-direction criterion 1e-8')
    ax.set_title("T4 singular-value spectrum (column normalized)", fontweight="bold")
    ax.set_xlabel("index"); ax.set_ylabel("sigma/sigma_max"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    y0 = 0.95
    for L in ["可识别性验证判词", ""] + lines + ["", f"总判词：{overall}"]:
        ax.text(0.03, y0, L, fontsize=11, fontweight="bold" if ("判词" in L) else "normal", wrap=True)
        y0 -= 0.085
    fig.suptitle("slow memory layer · identifiability math validation (no data, pure structural analysis)", fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n  figure saved: {OUT_PNG}")

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
    print(f"  results saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
