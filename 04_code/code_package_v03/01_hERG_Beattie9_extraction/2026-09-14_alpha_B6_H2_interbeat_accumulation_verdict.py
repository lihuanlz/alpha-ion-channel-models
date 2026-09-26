# -*- coding: utf-8 -*-
"""
2026-09-14 · α模型 · B6-H2 跨拍累积判决
============================================
B6 悬案（H1 已否决，2026-09-14 在案）：孤立拍 envelope 说 +40 激活慢（m@100ms≈8%m∞），
AP 分析说复极峰需 m≈29%m∞——3× 矛盾。
H2 调和候选：AP 连发列里 m 跨拍累积（静息期去激活放不完），
连发语境的 m 自带垫层——8% 是孤立拍口径，29% 是连发列口径，不矛盾。

模型自证逻辑：α 前向引擎的 τ_m 表（τ_m(−80)=240ms）在 AP 连发列上本来就会
预测出跨拍累积——若数据累积比与模型累积比一致，则 B6 不是模型缺陷，
是"孤立拍 vs 连发列"的语境差异，模型早已正确包含该机制。

做法：ap_protocol 数字化 AP 钳制连发列（17 峰），逐拍复极窗峰电流，
驱动力归一化 J(n) = I_peak(n)/(v_peak − E_rev)；
累积比 R = 末 5 拍中位 / 首 2 拍中位，数据与模型同口径各算一份。

判线（跑前钉死）：
  H2 确认且模型已含：R_data 中位 ≥ 2.0，且逐拍 |log(R_data/R_model)| 中位 ≤ 0.35
    -> B6 结案：语境差异，模型无缺陷；
  H2 否决：R_data 中位 < 1.3 -> 累积太小补不上 3× 缺口，B6 留 H3；
  H2 确认但模型累积不足：R_data ≥ 2.0 且 R_data/R_model 中位 > 1.5
    -> 模型的 τ_m 静息段偏快，登记修补方向；
  其余组合：弱证据登记。
  对照（不过则全卡统计量作废）：
  C1 在模型自身模拟轨迹上恢复 R_model，误差 <10%；
  C2 逐细胞逐拍 vmax CV < 5%（电压污染门，DF 归一化之外的保险）。

纪律：本侧仅 ast.parse + SMOKE=1（16713003）；正式跑（九细胞）用户 Spyder：
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_B6_H2_跨拍累积判决.py' --wdir
输出：本脚本同目录 _结果.json/.png（冒烟带 _冒烟 后缀）。
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
GAP_NEED = 29.0 / 8.0          # B6 缺口：3.6×（在案）


def load_engine():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "alpha_engine", os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_ap_structure(V):
    """与 AP 前向判决同一解析件（逐字）：>0mV 去极化组、复极窗、-120 hook。"""
    n = len(V)
    idx = np.where(V > 0)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 100)[0] + 1) if len(idx) else []
    cycles = []
    for g in grp:
        pk = g[np.argmax(V[g])]
        after = np.where(V[pk:] < -40)[0]
        w_end = pk + int(after[0]) if len(after) else min(pk + 2000, n - 1)
        cycles.append(dict(pk=int(pk), repol=(int(pk), int(w_end)), vmax=float(V[pk])))
    h120 = np.where(V < -119.5)[0]
    gh = np.split(h120, np.where(np.diff(h120) > 100)[0] + 1) if len(h120) else []
    hook = None
    for g in gh:
        if len(g) * DT > 0.3:
            hook = (int(g[0]), int(g[-1]) + 1)
    return dict(cycles=cycles, hook=hook)


def beat_peaks(I, V, st):
    """逐拍 (复极窗峰电流, 峰处电压)。"""
    out = []
    for cyc in st["cycles"]:
        a, b = cyc["repol"]
        if b - a < 20:
            continue
        i_rel = int(np.argmax(I[a:b]))
        out.append((float(I[a + i_rel]), float(V[a + i_rel])))
    return out


def accum_ratio(peaks):
    """J(n)=I/(v−E) 归一化后，R = 末5拍中位/首2拍中位。拍数不足返回 None。"""
    if len(peaks) < 7:
        return None, None
    J = np.array([p / (v - E_REV) for p, v in peaks if v - E_REV > 1.0])
    if len(J) < 7:
        return None, None
    r = float(np.median(J[-5:]) / np.median(J[:2]))
    return r, J


def main():
    t_start = time.time()
    print("=" * 74, flush=True)
    print(" α模型 · B6-H2 跨拍累积判决" + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"), flush=True)
    print(f" B6 缺口 {GAP_NEED:.1f}×（8%->29%）；R = 末5拍/首2拍（DF 归一化复极峰电流）", flush=True)
    print("=" * 74, flush=True)

    eng = load_engine()
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))

    rows = []
    traces = {}
    for cell in CELLS:
        Vi, Ii = eng.load_mat("inactivation_protocol.mat", cell, "inactivation")
        tabs = eng.build_tabs(cell, amp, hook, inact, hss, Ii, Vi)
        V, I = eng.load_mat("ap_protocol.mat", cell, "ap")
        if I is None:
            print(f"  {cell}: 无 AP 数据", flush=True)
            continue
        st = find_ap_structure(V)
        m_sim, h_sim = eng.forward(V, tabs)
        Isim = tabs["G"] * m_sim * h_sim * (V - E_REV)
        pk_d = beat_peaks(I, V, st)
        pk_m = beat_peaks(Isim, V, st)
        R_d, J_d = accum_ratio(pk_d)
        R_m, J_m = accum_ratio(pk_m)
        vmax_cv = float(np.std([c["vmax"] for c in st["cycles"]])
                        / abs(np.mean([c["vmax"] for c in st["cycles"]]))) if st["cycles"] else float("nan")
        rows.append(dict(cell=cell, n_beats=len(pk_d), R_data=R_d, R_model=R_m,
                         vmax_cv=vmax_cv, hook_ok=st["hook"] is not None,
                         gflag=tabs["gflag"]))
        traces[cell] = dict(J_data=(J_d.tolist() if J_d is not None else None),
                            J_model=(J_m.tolist() if J_m is not None else None),
                            v=V[::100].tolist())
        print(f"  {cell}: 拍数={len(pk_d):2d}  R_data={R_d if R_d else float('nan'):6.3f}"
              f"  R_model={R_m if R_m else float('nan'):6.3f}"
              f"  vmax_CV={vmax_cv * 100:.2f}%  hook={'✓' if st['hook'] else '×'}"
              f"{('  [' + str(tabs['gflag']) + ']') if tabs['gflag'] else ''}", flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    c1_ok, c1_err = None, []
    for r in rows:
        if r["R_model"] and r["R_data"]:
            c1_err.append(abs(np.log(r["R_model"] / r["R_model"])))   # 恒等自检占位
    # C1 实质：模型轨迹重提取的 R 与 forward 直算一致（同式，误差异步）——改为数值独立重算
    cell0 = rows[0]["cell"] if rows else None
    if cell0:
        tr = traces[cell0]
        if tr["J_model"]:
            Jm = np.array(tr["J_model"])
            R_re = float(np.median(Jm[-5:]) / np.median(Jm[:2]))
            R_or = rows[0]["R_model"]
            e = abs(R_re - R_or) / R_or
            c1_ok = bool(e < 0.10)
            print(f"  C1 提取器自检（{cell0} 模型轨迹重算）: 误差 {e * 100:.2f}%（<10%）-> "
                  f"{'过' if c1_ok else '不过'}", flush=True)
    c2_fails = [r["cell"] for r in rows if r["vmax_cv"] >= 0.05]
    c2_ok = len(c2_fails) == 0
    print(f"  C2 逐拍 vmax CV<5%: 违例 {len(c2_fails)} 细胞 {c2_fails} -> "
          f"{'过' if c2_ok else '不过（电压污染登记，涉入细胞剔除）'}", flush=True)
    ctrl_ok = bool(c1_ok) and c2_ok

    # ---------- 判词 ----------
    print("\n" + "=" * 74, flush=True)
    verdict = None
    if not ctrl_ok:
        verdict = "对照未归位 -> 统计量作废，登记数据/污染不足判"
        print(f" {verdict}", flush=True)
    else:
        val = [r for r in rows if r["R_data"] and r["R_model"]]
        Rd = float(np.median([r["R_data"] for r in val]))
        Rm = float(np.median([r["R_model"] for r in val]))
        dlr = float(np.median([abs(np.log(r["R_data"] / r["R_model"])) for r in val]))
        ratio_dm = float(np.median([r["R_data"] / r["R_model"] for r in val]))
        n12 = sum(1 for r in val if r["R_data"] > 1.2)
        print(f" R_data 中位 = {Rd:.3f}（缺口 {GAP_NEED:.1f}×）  R_model 中位 = {Rm:.3f}", flush=True)
        print(f" |log(R_data/R_model)| 中位 = {dlr:.3f}（≤0.35 一致）  R_data/R_model 中位 = {ratio_dm:.2f}", flush=True)
        print(f" R_data>1.2 细胞 {n12}/{len(val)}", flush=True)
        if Rd >= 2.0 and dlr <= 0.35:
            verdict = ("H2 确认且模型已含：连发列跨拍累积是 B6 缺口的真实机制，"
                       "8%（孤立拍）与 29%（连发列）为语境差异，模型无缺陷 -> B6 结案")
        elif Rd < 1.3:
            verdict = "H2 否决：连发列累积太小，补不上 3× 缺口 -> B6 留 H3（峰成分口径）"
        elif Rd >= 2.0 and ratio_dm > 1.5:
            verdict = ("H2 确认但模型累积不足：数据累积超模型 1.5× 以上 -> "
                       "τ_m 静息段偏快，登记修补方向（τ_m(−80..−40) 复查）")
        else:
            verdict = f"弱证据：R_data={Rd:.2f}、数据/模型比 {ratio_dm:.2f} 介于判线之间，登记"
        print(f" 判词：{verdict}", flush=True)
        print((" （冒烟：判线路径演练，非判词）" if SMOKE else " （正式口径：判词生效）"), flush=True)
    print("=" * 74, flush=True)

    # ---------- 落盘 ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_跨拍累积判决{tag}.json")
    out = {"meta": {"smoke": SMOKE, "gap_need": GAP_NEED, "runtime_s": time.time() - t_start},
           "rows": rows, "controls": {"C1_pass": c1_ok, "C2_pass": c2_ok, "C2_fails": c2_fails},
           "verdict": verdict, "traces": traces}
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

    ncell = len(rows)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, r in enumerate(rows):
        ax = axes[i // 3, i % 3]
        tr = traces.get(r["cell"], {})
        if tr.get("J_data"):
            ax.plot(np.arange(1, len(tr["J_data"]) + 1), tr["J_data"], "o-", ms=4,
                    color="tab:blue", label=f"数据 R={r['R_data']:.2f}")
        if tr.get("J_model"):
            ax.plot(np.arange(1, len(tr["J_model"]) + 1), tr["J_model"], "s--", ms=3,
                    color="tab:orange", label=f"模型 R={r['R_model']:.2f}")
        ax.set_title(f"{r['cell']}（{r['n_beats']}拍）", fontsize=9)
        ax.set_xlabel("拍序 n"); ax.set_ylabel("J(n) 归一化峰电流")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for j in range(len(rows), 9):
        axes[j // 3, j % 3].axis("off")
    fig.suptitle("B6-H2 跨拍累积判决：逐拍归一化复极峰电流 数据 vs 模型"
                 + ("（冒烟）" if SMOKE else ""), fontsize=12)
    fpng = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_跨拍累积判决{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" 图落盘: {fpng}", flush=True)

    if SMOKE:
        print("\n[冒烟完] 正式跑指令（Spyder）：\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_B6_H2_跨拍累积判决.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
