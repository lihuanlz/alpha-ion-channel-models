# -*- coding: utf-8 -*-
# 2026-09-13_率耦合_可识别性数学验证.py
# 目的（纯数学、合成数据、无实验数据、秒级）：
#   乘性耦合昨天被判"乘积简并"（T3）。今天问：如果慢层走的是【率耦合】——
#   慢变量 m(t) 调制快层速率而不是幅度——这个结构能不能被数据识别？
# 结构定义（事先冻结）：
#   快层：两个模态，瞬时相位 phi_i(t) = ∫ r_i(s) ds，y(t) = Σ a_i exp(-phi_i(t))
#   慢层：m(t) = m1 + (m0-m1)·exp(-(t/tau_s)^beta)，m0=m_ss(V_pre), m1=m_ss(V_test)
#   耦合：r_i(t) = r_i^0 · exp(kappa_i · m(t))    （历史改变快层时间常数）
#   m_ss(V) = 1/(1+exp(-(V-Vh)/kV))   全局 2 参数
# 四个探针（判读标准写死，不事后调整）：
#   P1a 家族可分性（正向）：率耦合合成数据 -> 用乘性模型拟合 -> 残差下限
#       残差 > 1e-3 记"乘性冒充不了率耦合（可分）"
#   P1b 家族可分性（反向）：乘性合成数据 -> 用率耦合模型拟合 -> 残差下限
#       两个方向都 >1e-3 才算"耦合形式可判决"
#   P2  率耦合参数可识别性：8 电压联合 Jacobian SVD，奇异值 <max*1e-8 记零方向
#   P3  多电压必要性：单电压 vs 8 电压联合的零方向对比
# 运行：python 本文件
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
OUT_JSON = os.path.join(HERE, "2026-09-13_率耦合_可识别性数学验证_结果.json")
OUT_PNG = os.path.join(HERE, "2026-09-13_率耦合_可识别性数学验证.png")

V_TEST = [-40, -50, -60, -70, -80, -90, -100, -110]
V_PRE = 50.0
NPT = 2000
T = np.linspace(1e-3, 2.0, NPT)

# 真实结构参数（量级取自战役实测：18ms/149ms 快双模态 + 宽谱慢层）
TRUE_RATE = dict(r1=1/0.018, r2=1/0.149, a1=0.45,
                 kappa1=0.8, kappa2=0.8, beta=0.6, tau_s=30.0,
                 Vh=-20.0, kV=15.0)
TRUE_MULT = dict(r1=1/0.018, r2=1/0.149, a1=0.45, beta=0.6, tau_s=30.0,
                 Vh=-20.0, kV=15.0)


def mss(V, Vh, kV):
    return 1.0 / (1.0 + np.exp(-np.clip((V - Vh) / kV, -500, 500)))


def m_of_t(v_test, beta, tau_s, Vh, kV):
    m0 = mss(V_PRE, Vh, kV)
    m1 = mss(v_test, Vh, kV)
    return m1 + (m0 - m1) * np.exp(-(T / tau_s) ** beta)


def y_rate_coupled(p, v_test):
    """率耦合形状曲线（不含 G、(V-E)，只看形状）"""
    m = m_of_t(v_test, p["beta"], p["tau_s"], p["Vh"], p["kV"])
    f1 = np.exp(p["kappa1"] * m)
    f2 = np.exp(p["kappa2"] * m)
    phi1 = np.concatenate([[0.0], np.cumsum(0.5 * (f1[1:] + f1[:-1]) * np.diff(T))]) * p["r1"]
    phi2 = np.concatenate([[0.0], np.cumsum(0.5 * (f2[1:] + f2[:-1]) * np.diff(T))]) * p["r2"]
    y = p["a1"] * np.exp(-phi1) + (1 - p["a1"]) * np.exp(-phi2)
    return y


def y_multiplicative(p, v_test, scale_v):
    """乘性形状曲线：快双指数 x 慢因子（每电压一个自由总幅度 scale_v）"""
    fast = p["a1"] * np.exp(-p["r1"] * T) + (1 - p["a1"]) * np.exp(-p["r2"] * T)
    m = m_of_t(v_test, p["beta"], p["tau_s"], p["Vh"], p["kV"])
    return scale_v * fast * m


def stack_rate(p):
    return np.concatenate([y_rate_coupled(p, v) for v in V_TEST])


def stack_mult(p, scales):
    return np.concatenate([y_multiplicative(p, v, s) for v, s in zip(V_TEST, scales)])


# ---------- P1a：率耦合真 -> 乘性拟合（扫 kappa 强度） ----------
def P1a():
    rng = np.random.default_rng(20260913)
    table = {}
    for kappa in (0.3, 0.8, 1.5, 2.5):
        truth = dict(TRUE_RATE); truth["kappa1"] = kappa; truth["kappa2"] = kappa
        y_true = stack_rate(truth)
        scales0 = np.ones(len(V_TEST))

        def unpack(x):
            p = dict(r1=np.exp(x[0]), r2=np.exp(x[1]), a1=1/(1+np.exp(-x[2])),
                     beta=1/(1+np.exp(-x[3]))*1.5+0.05, tau_s=np.exp(x[4]),
                     Vh=x[5], kV=np.exp(x[6]))
            sc = np.exp(x[7:])
            return p, sc

        x0 = np.array([np.log(TRUE_MULT["r1"]), np.log(TRUE_MULT["r2"]), 0.0,
                       0.5, np.log(TRUE_MULT["tau_s"]), -20.0, np.log(15.0)] + list(np.log(scales0)))
        best = None
        for k in range(4):
            xs = x0 if k == 0 else x0 + rng.normal(scale=0.4, size=len(x0))
            try:
                r = least_squares(lambda x: stack_mult(*unpack(x)) - y_true, xs,
                                  method="trf", max_nfev=1500)
                if best is None or r.cost < best.cost:
                    best = r
            except Exception:
                pass
        y_hat = stack_mult(*unpack(best.x))
        rel = float(np.max(np.abs(y_hat - y_true)) / (np.max(np.abs(y_true)) + 1e-300))
        rms = float(np.sqrt(np.mean((y_hat - y_true) ** 2)) / (np.std(y_true) + 1e-300))
        table[kappa] = (rel, rms)
    return table


# ---------- P1b：乘性真 -> 率耦合拟合 ----------
def P1b():
    scales_true = np.linspace(1.0, 0.5, len(V_TEST))   # 乘性真值带每电压幅度
    y_true = stack_mult(TRUE_MULT, scales_true)

    def unpack(x):
        return dict(r1=np.exp(x[0]), r2=np.exp(x[1]), a1=1/(1+np.exp(-x[2])),
                    kappa1=x[3], kappa2=x[3],   # 同 kappa 简化
                    beta=1/(1+np.exp(-x[4]))*1.5+0.05, tau_s=np.exp(x[5]),
                    Vh=x[6], kV=np.exp(x[7]))

    x0 = np.array([np.log(TRUE_RATE["r1"]), np.log(TRUE_RATE["r2"]), 0.0,
                   0.8, 0.5, np.log(TRUE_RATE["tau_s"]), -20.0, np.log(15.0)])
    rng = np.random.default_rng(20260914)
    best = None
    for k in range(6):
        xs = x0 if k == 0 else x0 + rng.normal(scale=0.4, size=len(x0))
        try:
            r = least_squares(lambda x: stack_rate(unpack(x)) - y_true, xs,
                              method="trf", max_nfev=2000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    y_hat = stack_rate(unpack(best.x))
    rel = float(np.max(np.abs(y_hat - y_true)) / (np.max(np.abs(y_true)) + 1e-300))
    rms = float(np.sqrt(np.mean((y_hat - y_true) ** 2)) / (np.std(y_true) + 1e-300))
    return rel, rms


# ---------- P2 / P3：率耦合 Jacobian SVD ----------
def jac_rate(params_vec, v_list):
    names = ["r1", "r2", "a1", "kappa", "beta", "tau_s", "Vh", "kV"]

    def model(x):
        p = dict(r1=np.exp(x[0]), r2=np.exp(x[1]), a1=1/(1+np.exp(-x[2])),
                 kappa1=x[3], kappa2=x[3],
                 beta=1/(1+np.exp(-x[4]))*1.5+0.05, tau_s=np.exp(x[5]),
                 Vh=x[6], kV=np.exp(x[7]))
        return np.concatenate([y_rate_coupled(p, v) for v in v_list])

    x0 = np.array([np.log(params_vec["r1"]), np.log(params_vec["r2"]), 0.0,
                   params_vec["kappa1"], 0.6, np.log(params_vec["tau_s"]),
                   params_vec["Vh"], np.log(params_vec["kV"])])
    f0 = model(x0)
    J = np.zeros((len(f0), len(x0)))
    for j in range(len(x0)):
        h = 1e-6 * max(abs(x0[j]), 1e-6)
        x1 = x0.copy(); x1[j] += h
        J[:, j] = (model(x1) - f0) / h
    col = np.linalg.norm(J, axis=0) + 1e-300
    sv = np.linalg.svd(J / col, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-8))
    return sv, rank, len(x0), names


def main():
    print("=" * 64)
    print(" 率耦合可识别性数学验证（合成数据，无实验数据）")
    print("=" * 64, flush=True)

    print("\n[P1a] 率耦合真 -> 乘性拟合，扫 kappa 强度（判线：残差>1e-3 则乘性冒充不了）", flush=True)
    tab_a = P1a()
    v_a_map = {}
    for kappa, (rel, rms) in tab_a.items():
        v = "冒充不了" if max(rel, rms) > 1e-3 else "能冒充"
        v_a_map[kappa] = v
        print(f"  kappa={kappa}: 最大相对残差 {rel:.5f}  相对RMS {rms:.5f}  -> {v}", flush=True)
    rel_a, rms_a = tab_a[0.8]
    v_a = "乘性冒充不了率耦合" if max(rel_a, rms_a) > 1e-3 else "乘性能冒充率耦合"

    print("\n[P1b] 乘性真 -> 率耦合拟合", flush=True)
    rel_b, rms_b = P1b()
    print(f"  最大相对残差 = {rel_b:.5f}   相对RMS = {rms_b:.5f}")
    v_b = "率耦合冒充不了乘性" if max(rel_b, rms_b) > 1e-3 else "率耦合能冒充乘性"
    print(f"  -> {v_b}")

    distinguishable = all(v == "冒充不了" for v in v_a_map.values()) and (max(rel_b, rms_b) > 1e-3)
    print(f"\n  家族可分性判词: {'两个耦合形式在全 kappa 范围可分' if distinguishable else '存在单向冒充——弱调制区耦合形式不可判决'}")

    print("\n[P2] 率耦合参数可识别性（8 电压联合 SVD）", flush=True)
    sv8, rank8, npar, names = jac_rate(TRUE_RATE, V_TEST)
    print(f"  参数({npar}): {names}")
    print(f"  奇异值: {[round(float(x),4) for x in sv8]}")
    print(f"  有效秩 {rank8}/{npar}，零方向 {npar - rank8}")

    print("\n[P3] 单电压（V=-50）对照", flush=True)
    sv1, rank1, _, _ = jac_rate(TRUE_RATE, [-50])
    print(f"  奇异值: {[round(float(x),4) for x in sv1]}")
    print(f"  有效秩 {rank1}/{npar}，零方向 {npar - rank1}")

    lines = [
        f"P1a 乘性冒充率耦合（扫 kappa）: " + "; ".join(f"k={k}:{v_a_map[k]}" for k in sorted(v_a_map)),
        f"P1b 率耦合冒充乘性: 残差 {rel_b:.4f} -> {v_b}",
        f"家族判词: {'全范围可分' if distinguishable else '弱调制区不可判决'}",
        f"P2 八电压联合: 秩 {rank8}/{npar}（零方向 {npar-rank8}）",
        f"P3 单电压: 秩 {rank1}/{npar}（零方向 {npar-rank1}）",
    ]
    print("\n  --- 汇总 ---")
    for L in lines:
        print("  " + L)

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    y_true = stack_rate(TRUE_RATE)
    n = NPT
    ax.plot(T, y_true[:n], 'k-', lw=2, label="率耦合真值 (V=-40)")
    scales0 = np.ones(len(V_TEST))
    # 用 P1a 最优结果重画太贵，直接画乘性在真参下的形状对比
    y_mult = stack_mult(TRUE_MULT, scales0)
    ax.plot(T, y_mult[:n], 'r--', lw=1.5, label="乘性（同参数）")
    ax.set_title("两族结构同参数下的形状差异", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    m = m_of_t(-50, TRUE_RATE["beta"], TRUE_RATE["tau_s"], TRUE_RATE["Vh"], TRUE_RATE["kV"])
    ax2 = ax.twinx()
    ax.plot(T, np.exp(TRUE_RATE["kappa1"] * m), 'b-', label="速率因子 exp(kappa·m)")
    ax2.plot(T, m, 'g--', label="慢变量 m(t)")
    ax.set_title("率耦合机制：慢变量 -> 快层速率", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.legend(loc="upper right"); ax2.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.semilogy(range(1, len(sv8) + 1), sv8 / sv8[0], 'o-', label="8 电压联合")
    ax.semilogy(range(1, len(sv1) + 1), sv1 / sv1[0], 's-', label="单电压 -50")
    ax.axhline(1e-8, color='r', ls='--', label='零方向判线 1e-8')
    ax.set_title("率耦合 Jacobian 奇异值谱", fontweight="bold")
    ax.set_xlabel("序号"); ax.set_ylabel("sigma/sigma_max"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    y0 = 0.95
    for L in ["率耦合可识别性判词", ""] + lines:
        ax.text(0.03, y0, L, fontsize=11, fontweight="bold" if "判词" in L else "normal", wrap=True)
        y0 -= 0.09
    fig.suptitle("率耦合 · 可识别性数学验证（合成数据）", fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n  图落盘: {OUT_PNG}")

    def _jd(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)
    results = {"P1a": {"table": {f"kappa={k}": {"max_rel": float(v[0]), "rms_rel": float(v[1]),
                                               "verdict": v_a_map[k]} for k, v in tab_a.items()}},
               "P1b": {"max_rel": rel_b, "rms_rel": rms_b, "verdict": v_b},
               "distinguishable": bool(distinguishable),
               "P2": {"sv": sv8, "rank": rank8, "n_par": npar, "names": names},
               "P3": {"sv": sv1, "rank": rank1, "n_par": npar}}
    json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=_jd)
    print(f"  结果落盘: {OUT_JSON}")


if __name__ == "__main__":
    main()
