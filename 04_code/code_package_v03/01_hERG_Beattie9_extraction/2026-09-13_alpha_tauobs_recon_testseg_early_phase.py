# 2026-09-13_α模型_τobs侦察_测试段早期相位.py
# 目的（侦察，不是判决）：缺口2（τ_h(V) 全表）+ 缺口1b（τ_obs 判别）设计前看数据。
#   失活协议测试段（测试V×0.150s，前接 -90x0.06s 复位 -> h(0)≈1, m(0)≈0.43 残留在案）：
#   h(t) 从 ~1 向 h_ss(V) 弛豫，早期相位形状携带 τ_obs。端点被两个实测钉死
#   （h(0)=1、h(150ms)=剥离 h_150），只有中段形状携带 τ_obs —— 本脚本回答：
#   在实测噪声下，τ_obs=25/50/100/175ms 四种形状能不能分开（SNR 够不够）？
# 输出：4 细胞 × 7 档 PNG（数据 2ms 分箱中位，归一到末 20ms；预测曲线同口径归一）
#   + 30ms/60ms 比值表（数据 vs 四档 τ 预测）。
# 侦察性质：我自己跑，不写判词，不作判线；正式判决脚本据此设计后再由用户跑。
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
DT = 1e-4
CELLS = ["16713003", "16704007", "16708118", "16708060"]
GEARS = [-70, -60, -50, -40, -30, 0, 40]
TAUS = [0.025, 0.050, 0.100, 0.175]          # 四种 τ_obs 候选形状
TAU_EFF = {-70: 0.81, -60: 1.21, -50: 1.8, -40: 2.0, -30: 2.0, 0: 2.0, 40: 0.29}
MSS = {-70: 0.057, -60: 0.19, -50: 0.46, -40: 1.0, -30: 1.0, 0: 1.0, 40: 1.0}


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
    """测试档：时长~0.15s 且前一段为 -90x0.06s。"""
    info = segments(V)
    out = {}
    for k, (v, s0, n) in enumerate(info):
        if 0.14 < n * DT < 0.16 and k >= 1:
            pv, ps, pn = info[k - 1]
            if abs(pv + 90.0) < 2.0 and 0.05 < pn * DT < 0.07:
                out[int(round(v))] = (s0, n)
    return out


def binmed(t, y, w=0.002):
    nb = int(t[-1] / w)
    tm, ym = [], []
    for b in range(nb):
        m = (t >= b * w) & (t < (b + 1) * w)
        if m.sum() >= 3:
            tm.append((b + 0.5) * w)
            ym.append(float(np.median(y[m])))
    return np.array(tm), np.array(ym)


def pred_shape(t, tau_obs, h150, v, m0=0.435):
    """h_ss 由 (h150, τ_obs) 族定；m 单 τ_eff 近似（侦察口径）。归一到 t=150ms。"""
    r = np.exp(-0.150 / tau_obs)
    hss = (h150 - r) / (1.0 - r) if r < 0.999 else h150
    hss = max(hss, 0.0)
    h = hss + (1.0 - hss) * np.exp(-t / tau_obs)
    te = TAU_EFF.get(v, 2.0)
    mss = MSS.get(v, 1.0)
    m = mss + (m0 - mss) * np.exp(-t / te)
    y = m * h
    return y / y[-1]


def main():
    hssj = json.load(open(os.path.join(
        HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json"), encoding="utf-8"))

    fig, axes = plt.subplots(len(CELLS), len(GEARS), figsize=(3.0 * len(GEARS), 2.6 * len(CELLS)),
                             squeeze=False)
    print("=" * 96)
    print(" τ_obs 侦察：测试段早期相位（归一 i_end=末20ms均值；30/60ms 比值 数据 vs 预测）")
    print("=" * 96)
    for ri, cell in enumerate(CELLS):
        V, I = load_mat("inactivation_protocol.mat", cell, "inactivation")
        ts = test_segments(V)
        # 噪声：末尾 -80x1.34s 段末 500ms 去趋势 std
        info = segments(V)
        sig = np.nan
        for v, s0, n in info:
            if abs(v + 80) < 2 and n * DT > 1.0:
                seg = I[s0 + n - 5000: s0 + n]
                seg = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg)))
                sig = float(np.std(seg))
        hc = hssj["cells"].get(cell, {}).get("curve", {})
        print(f"\n细胞 {cell}（σ={sig:.4f} nA）")
        for ci, g in enumerate(GEARS):
            ax = axes[ri][ci]
            if g not in ts or I is None:
                ax.set_title(f"{g}mV 无档", fontsize=8)
                continue
            s0, n = ts[g]
            t = np.arange(n) * DT
            i = I[s0: s0 + n]
            i_end = float(np.mean(i[-2000:]))
            tb, yb = binmed(t, i / i_end)
            e = hc.get(str(g))
            h150 = float(e["h"]) if (e and e.get("qc") and e["h"] > 0) \
                else float(hssj["gears"][str(g)]["med"])
            for tau, col in zip(TAUS, ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
                ax.plot(t, pred_shape(t, tau, h150, g), lw=0.9, color=col,
                        label=f"τ={tau * 1e3:.0f}ms")
            ax.plot(tb, yb, "k.", ms=2.5, label="数据")
            ax.set_title(f"{cell[-3:]} {g}mV  h150={h150:.3f}", fontsize=8)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6, loc="upper right")
            # 30/60ms 比值（数据 ±2ms 窗）
            for tq in (0.030, 0.060):
                m = np.abs(t - tq) < 0.002
                if m.any():
                    pass
            r30 = float(np.mean(i[np.abs(t - 0.030) < 0.002])) / i_end
            r60 = float(np.mean(i[np.abs(t - 0.060) < 0.002])) / i_end
            p = {tau: (pred_shape(np.array([0.030, 0.060]), tau, h150, g)
                       / 1.0) for tau in TAUS}
            print(f"  {g:>4}mV: 30ms 数据{r30:.2f} | τ25 {p[0.025][0]:.2f} τ50 {p[0.050][0]:.2f} "
                  f"τ100 {p[0.100][0]:.2f} τ175 {p[0.175][0]:.2f} || "
                  f"60ms 数据{r60:.2f} | τ25 {p[0.025][1]:.2f} τ50 {p[0.050][1]:.2f} "
                  f"τ100 {p[0.100][1]:.2f} τ175 {p[0.175][1]:.2f}")
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_τobs侦察_测试段早期相位.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")
    print(f"\n  图落盘: {fpng}")


if __name__ == "__main__":
    main()
