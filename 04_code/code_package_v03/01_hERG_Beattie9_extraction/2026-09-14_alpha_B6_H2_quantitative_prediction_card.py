# -*- coding: utf-8 -*-
"""
2026-09-14 · α模型 · B6-H2 定量预测卡（纯仿真，预注册）
==========================================================
B6 悬案：孤立拍 envelope m@100ms≈8% m∞ vs AP 复极峰需 m≈29% m∞（3×）。
H2（混合态/跨拍累积，唯一获方向性支持的候选）的可证伪硬预测：
**累积量必须随起搏周长（CL）单调变**——CL 短，静息期去激活放不完，垫层高；
CL 长，垫层消失，m 回落到孤立拍口径。

本卡用 α 正式前向引擎在数字化 AP 波形上扫 CL，把三条预测曲线钉死，
作为预注册：将来恒等波形连发列数据（grandi_2hz 类，本地 0/9 在案）到手，
对曲线即结 B6；实测 R(CL) 全平 -> H2 否决，B6 归 H3。

做法：ap_protocol 取一个代表拍（vmax 中位拍），-80 静息拼接成 CL 规则连发列
（20 拍，初值 m=m_ss(-80)），逐拍记：
  m_onset(n)   拍头 m（垫层）；
  m@100ms(n)   拍内 100ms 的 m（对应 envelope 口径）；
  J(n)         复极窗归一化峰电流 = max I/(v-E_rev)；
  R = 末5拍中位/首2拍中位（与 H2 判决卡同口径）。
孤立 +40 阶跃参照：m@100ms（envelope 复刻）。

预注册判线（数据到位后执行）：
  P1 模型预测 R(CL=500) ≥ 3 且 R(CL=2000) ≤ 1.5；
  P2 实测 R(CL) 与预测曲线对数误差 ≤ 0.35 -> B6 结案子 H2（语境差异）；
  P3 实测 R(CL) 全 CL ∈ [0.8, 1.3] -> H2 否决，B6 归 H3。

运行：本侧直接运行（纯仿真轻量）。输出 _结果.json/.png。

修订记录：v2（2026-09-14）数据对拍前修三处：
 ① 复极窗对齐 bug——原窗 I_beat[500:2500] 对 CL>500 落在静息段（rest_n>500），
   且 v_w 错用 V_beat 切片归一，J 绝对值与 R(CL>=750) 失真（伪 14x 跨 CL 差）；
   改为 I_beat[rest_n+500:rest_n+2500] 对 V_beat[500:2500]。
 ② m_onset 口径由"静息入口"改为"进拍瞬间"（真垫层）。
 ③ 孤立阶跃期望注释更正：模型 tau_m(+40) 锚 0.29s -> m@100ms≈30%，
   与 envelope 实测 ~8% 之差即 B6 本体，模型不还原 envelope 侧。
"""

import os
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
E_REV = -88.33
CL_LIST_MS = [500, 750, 1000, 1500, 2000, 4000]
N_BEATS = 20


def load_engine():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "alpha_engine", os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def representative_beat(V):
    """取 vmax 中位拍：峰前 50ms 到峰后 400ms，基线 -80。"""
    idx = np.where(V > 0)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 100)[0] + 1)
    pks = [(g[np.argmax(V[g])], float(V[g[np.argmax(V[g])]])) for g in grp]
    pks.sort(key=lambda x: x[1])
    pk = pks[len(pks) // 2][0]
    a, b = int(pk) - 500, int(pk) + 4000
    return V[a:b].copy()


def simulate_cell(eng, tabs, V_beat):
    """返回 {CL: dict(m_onset, m100, R, J轨迹)} + 孤立阶跃参照。"""
    ms_tab, tm_tab = tabs["m_ss"], tabs["tau_m"]
    hs_tab, th_tab = tabs["h_ss"], tabs["tau_h"]
    G = tabs["G"]

    def run_train(CL_ms):
        rest_n = int(round(CL_ms / 1000.0 / DT)) - len(V_beat)
        if rest_n < 0:
            return None
        V = np.concatenate([np.full(rest_n, -80.0), V_beat])
        per = len(V)
        Vt = np.tile(V, N_BEATS)
        # 逐样本弛豫（eng.forward 同式，但保留逐拍切片需要全轨迹）
        ms = np.array([ms_tab(v) for v in Vt])
        tm = np.array([tm_tab(v) for v in Vt])
        hs = np.array([hs_tab(v) for v in Vt])
        th = np.array([th_tab(v) for v in Vt])
        em = np.exp(-DT / tm)
        eh = np.exp(-DT / th)
        m0, h0 = ms[0], hs[0]
        m_on, m100, J = [], [], []
        beat_len = per
        i_up = rest_n + 500                     # 峰位置（拍内）
        for b in range(N_BEATS):
            seg0 = b * beat_len
            m_beat = np.empty(beat_len)
            I_beat = np.empty(beat_len)
            for i in range(beat_len):
                k = seg0 + i
                m0 = ms[k] + (m0 - ms[k]) * em[k]
                h0 = hs[k] + (h0 - hs[k]) * eh[k]
                m_beat[i] = m0
                I_beat[i] = G * m0 * h0 * (Vt[k] - E_REV)
            m_on.append(m_beat[rest_n])          # 拍头 = 静息结束进拍瞬间（真垫层）
            m100.append(m_beat[min(rest_n + 1000, beat_len - 1)])
            w = I_beat[rest_n + 500:rest_n + 2500]  # 复极窗（峰~峰后200ms，对齐拍起点）
            v_w = V_beat[500:2500]
            k = int(np.argmax(w))
            J.append(float(w[k] / (v_w[k] - E_REV)))
        R = float(np.median(J[-5:]) / np.median(J[:2])) if np.median(J[:2]) > 0 else float("nan")
        return dict(CL_ms=CL_ms, m_onset_last=float(m_on[-1]), m100_last=float(m100[-1]),
                    R=R, J=J, m_onset=m_on, m100=m100)

    out = {}
    for CL in CL_LIST_MS:
        r = run_train(CL)
        if r:
            out[str(CL)] = r

    # 孤立 +40 阶跃（envelope 口径复刻）：-80 100ms -> +40 400ms
    Vs = np.concatenate([np.full(1000, -80.0), np.full(4000, 40.0)])
    m0 = float(ms_tab(-80.0))
    for i, v in enumerate(Vs):
        m0 = ms_tab(v) + (m0 - ms_tab(v)) * np.exp(-DT / tm_tab(v))
        if i == 1999:                            # 阶跃后 100ms
            m_step100 = float(m0)
    return out, m_step100


def main():
    t0 = time.time()
    print("=" * 74, flush=True)
    print(" α模型 · B6-H2 定量预测卡（纯仿真预注册）" + ("（冒烟 16713003）" if SMOKE else "（九细胞）"), flush=True)
    print(" 预测：R(CL=500)≥3 且 R(CL=2000)≤1.5；实测曲线对数误差≤0.35 -> B6 结 H2", flush=True)
    print("=" * 74, flush=True)

    eng = load_engine()
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    V_ap, _ = eng.load_mat("ap_protocol.mat", "16713003", "ap")
    V_beat = representative_beat(V_ap)
    print(f" 代表拍: 时长 {len(V_beat) * DT * 1000:.0f}ms，vmax={V_beat.max():.1f}mV", flush=True)

    per_cell = {}
    for cell in CELLS:
        Vi, Ii = eng.load_mat("inactivation_protocol.mat", cell, "inactivation")
        tabs = eng.build_tabs(cell, amp, hook, inact, hss, Ii, Vi)
        out, m_step100 = simulate_cell(eng, tabs, V_beat)
        per_cell[cell] = {"cl": out, "m_step100_isolated": m_step100}
        line = f"  {cell}: 孤立+40 m@100ms={m_step100 * 100:5.1f}%  | "
        line += "  ".join(f"CL{c}: R={out[c]['R']:.2f} 垫层={out[c]['m_onset_last'] * 100:.0f}%"
                          for c in ("500", "1000", "2000") if c in out)
        print(line, flush=True)

    # 群体曲线
    print("\n[群体预测曲线（中位）]", flush=True)
    curve = {}
    for CL in CL_LIST_MS:
        rs = [per_cell[c]["cl"][str(CL)]["R"] for c in per_cell if str(CL) in per_cell[c]["cl"]]
        ons = [per_cell[c]["cl"][str(CL)]["m_onset_last"] for c in per_cell if str(CL) in per_cell[c]["cl"]]
        m100s = [per_cell[c]["cl"][str(CL)]["m100_last"] for c in per_cell if str(CL) in per_cell[c]["cl"]]
        curve[str(CL)] = dict(R_med=float(np.median(rs)), R_lo=float(np.min(rs)), R_hi=float(np.max(rs)),
                              onset_med=float(np.median(ons)), m100_med=float(np.median(m100s)))
        print(f"  CL={CL:5d}ms: R 中位={curve[str(CL)]['R_med']:.2f}"
              f" [{curve[str(CL)]['R_lo']:.2f}~{curve[str(CL)]['R_hi']:.2f}]"
              f"  垫层 m_onset={curve[str(CL)]['onset_med'] * 100:.1f}%"
              f"  m@100ms={curve[str(CL)]['m100_med'] * 100:.1f}%", flush=True)
    iso = [per_cell[c]["m_step100_isolated"] for c in per_cell]
    print(f"  孤立 +40 阶跃 m@100ms 中位 = {np.median(iso) * 100:.1f}%"
          f"（模型 τ_m(+40) 锚 0.29s 口径，应≈30%；与 envelope 实测 ~8% 之差即 B6 本体）", flush=True)
    p1 = curve["500"]["R_med"] >= 3.0 and curve["2000"]["R_med"] <= 1.5
    print(f"\n 预注册 P1（R(500)≥3 且 R(2000)≤1.5）-> 模型预测{'满足' if p1 else '不满足——H2 机制量级不足，登记'}", flush=True)

    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_定量预测卡{tag}.json")
    out = {"meta": {"smoke": SMOKE, "cl_list_ms": CL_LIST_MS, "n_beats": N_BEATS,
                    "runtime_s": time.time() - t0},
           "per_cell": per_cell, "curve": curve,
           "isolated_step_m100_median": float(np.median(iso)), "P1_pass": p1}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\n 结果落盘: {fjson}", flush=True)

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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    cls = [int(c) for c in CL_LIST_MS]
    Rmed = [curve[str(c)]["R_med"] for c in cls]
    Rlo = [curve[str(c)]["R_lo"] for c in cls]
    Rhi = [curve[str(c)]["R_hi"] for c in cls]
    ax.fill_between(cls, Rlo, Rhi, alpha=0.2, color="tab:blue")
    ax.plot(cls, Rmed, "o-", color="tab:blue", label="R(CL) 群体中位")
    ax.axhline(3.0, color="tab:green", ls=":", lw=1, label="预注册线 R(500)≥3")
    ax.axhline(1.5, color="tab:red", ls=":", lw=1, label="预注册线 R(2000)≤1.5")
    ax.axhline(3.6, color="k", ls="--", lw=1, label="B6 缺口 3.6×")
    ax.set_xscale("log"); ax.set_xticks(cls); ax.set_xticklabels(cls)
    ax.set_xlabel("起搏周长 CL (ms)"); ax.set_ylabel("累积比 R = 末5拍/首2拍")
    ax.set_title("H2 预测：累积比 vs 周长"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    onmed = [curve[str(c)]["onset_med"] * 100 for c in cls]
    m1med = [curve[str(c)]["m100_med"] * 100 for c in cls]
    ax.plot(cls, onmed, "o-", label="垫层 m_onset（末拍）")
    ax.plot(cls, m1med, "s--", label="拍内 m@100ms（末拍）")
    ax.axhline(float(np.median(iso)) * 100, color="k", ls="--", lw=1,
               label=f"孤立拍 m@100ms = {np.median(iso) * 100:.0f}%")
    ax.axhline(29, color="tab:red", ls=":", lw=1, label="AP 需求 29%")
    ax.set_xscale("log"); ax.set_xticks(cls); ax.set_xticklabels(cls)
    ax.set_xlabel("CL (ms)"); ax.set_ylabel("m (% m∞)")
    ax.set_title("垫层与拍内 m vs 周长"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    c0 = CELLS[0]
    for CLs, col in [("500", "tab:red"), ("1000", "tab:blue"), ("4000", "tab:green")]:
        if CLs in per_cell[c0]["cl"]:
            J = per_cell[c0]["cl"][CLs]["J"]
            ax.plot(np.arange(1, len(J) + 1), J, "o-", ms=3, color=col, label=f"CL={CLs}ms")
    ax.set_xlabel("拍序 n"); ax.set_ylabel("J(n)")
    ax.set_title(f"逐拍归一化峰电流（{c0}）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("B6-H2 定量预测卡（预注册，待恒等连发列数据结 B6）"
                 + ("（冒烟）" if SMOKE else ""), fontsize=12)
    fpng = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_定量预测卡{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" 图落盘: {fpng}", flush=True)


if __name__ == "__main__":
    main()
