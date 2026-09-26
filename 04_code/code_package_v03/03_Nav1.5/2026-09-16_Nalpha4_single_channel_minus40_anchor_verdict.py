# -*- coding: utf-8 -*-
"""
Nα-4: Nav1.5 single-channel -40 mV single-point anchoring verdict (first order: pure computation on the paper's extracted tables)
Preregistration: 04_细胞线4/结果/预注册_Nα4_Nav15单通道负40单点锚定判决_2026-09-16.md (v1.0 pinned before the run)
Input: 公开数据/Nα4探路/ five xlsx files + Nα-2 whole-cell results JSON (tau_h(-40) anchor)
Output: _结果.json / _结果.csv / _结果.png
"""
import sys, json, glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass

ROOT = Path(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4")
PROBE = ROOT / "公开数据" / "Nα4探路"
NA2_JSON = ROOT / "α模型" / "2026-09-15_Nα2_Nav15整通道一次验证_结果.json"
OUT_BASE = ROOT / "α模型" / "2026-09-16_Nα4_单通道负40单点锚定判决"

# ---------- criteria (preregistration v1.0 pinned, do not change) ----------
J1_CV, J1_RANGE = 0.30, 2.0        # crit-1 tau_decay: CV<=0.30 and max/min<=2.0
J2_CV, J2_RANGE = 0.50, 4.0        # crit-2 Po_peak: CV<=0.50 and max/min<=4.0
J3_LO, J3_HI = 0.5, 2.0            # crit-3 median ratio in [0.5, 2.0]
WC_TH40_POOL = 2.157684300189219   # whole-cell pooled tau_h(-40) ms (Nα-2)
WC_TH40_CELL = {"008": 1.6420456346535064, "005": 1.6813111998402044,
                "011": 3.1496960660739464}


def load_sheet(xlsx, sheet, cols):
    """Read a sheet; cols={new_name: original_column}; drop rows with empty file name; cast numeric columns to float."""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df[df.iloc[:, 0].notna()]
    out = pd.DataFrame({"file": df.iloc[:, 0].astype(str).str.strip()})
    for new_name, col_idx in cols.items():
        out[new_name] = pd.to_numeric(df.iloc[:, col_idx], errors="coerce")
    return out.dropna(subset=list(cols.keys())).reset_index(drop=True)


def stats(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(x.size), "median": float(np.median(x)),
            "mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)),
            "cv": float(np.std(x, ddof=1) / np.mean(x)),
            "min": float(x.min()), "max": float(x.max()),
            "range_ratio": float(x.max() / x.min())}


def main():
    print("=" * 72)
    print(" Na-4: Nav1.5 single-channel -40 mV single-point anchoring verdict (first order, paper extracted tables)")
    print(" Preregistration v1.0 (pinned before the 2026-09-16 run)")
    print("=" * 72)

    fx = {p.name.split("__")[0]: p for p in PROBE.glob("*.xlsx")}
    main_x = [v for k, v in fx.items() if "late and peak INa analysis" in k and "NaV1.6" not in k][0]
    dk_x = [v for k, v in fx.items() if "deltaKPQ" in k][0]
    rc_x = [v for k, v in fx.items() if "recovery" in k][0]
    ld_x = [v for k, v in fx.items() if "lidocaine" in k][0]
    n16_x = [v for k, v in fx.items() if "NaV1.6" in k][0]

    # ---- read tables ----
    sc = load_sheet(main_x, "single-channel recordings",
                    {"tau_decay_ms": 1, "t_peak_ms": 2, "po_peak": 3,
                     "late_pct": 4, "peak_pA": 5})
    mc_pl = load_sheet(main_x, "mult-ch. peak and late",
                       {"late_pct": 1, "peak_pA": 2})
    mc_tau = load_sheet(main_x, "mult-ch. taus, DKL and VMR",
                        {"tau_decay_ms": 1, "dkl": 2, "vmr": 3})
    dk = load_sheet(dk_x, "dKPQ cell-attached",
                    {"peak_pA": 1, "late_pct": 2})
    rc = load_sheet(rc_x, "recovery from inactivation", {"p2_p1": 1})
    ld = load_sheet(ld_x, "lido",
                    {"peak_ctr": 1, "peak_lido": 3, "ratio": 4})
    n16 = load_sheet(n16_x, "CHO-NaV1.6 recordings",
                     {"late_pct": 1, "peak_pA": 2})

    print(f"[tables] single-channel {len(sc)} | multi-channel tau {len(mc_tau)} | multi-channel peak/late {len(mc_pl)}")
    print(f"[tables] dKPQ {len(dk)} | recovery {len(rc)} | lido {len(ld)} | NaV1.6 {len(n16)}")

    # ---- statistics ----
    s_tau = stats(mc_tau["tau_decay_ms"])
    s_po = stats(sc["po_peak"])
    s_tpeak = stats(sc["t_peak_ms"])
    s_vmr = stats(mc_tau["vmr"])
    s_dkl = stats(mc_tau["dkl"])
    s_late_wt = stats(pd.concat([sc["late_pct"], mc_pl["late_pct"]]))
    s_late_dk = stats(dk["late_pct"])
    s_rc = stats(rc["p2_p1"])
    s_ld = stats(ld["ratio"])
    s_late_16 = stats(n16["late_pct"])

    # Subgroup structure registry (batches by filename prefix)
    def batch(f):
        f = str(f)
        return f[:5] if len(f) >= 5 else f
    mc_pl2 = mc_pl.copy(); mc_pl2["batch"] = mc_pl2["file"].map(batch)
    subgrp = (mc_pl2.groupby("batch")["late_pct"]
              .agg(["count", "median"]).reset_index()
              .rename(columns={"count": "n", "median": "late_pct_median"}))

    # ---- four verdicts ----
    j1_pass = (s_tau["cv"] <= J1_CV) and (s_tau["range_ratio"] <= J1_RANGE)
    j2_pass = (s_po["cv"] <= J2_CV) and (s_po["range_ratio"] <= J2_RANGE)
    j3_ratio = s_tau["median"] / WC_TH40_POOL
    j3_pass = J3_LO <= j3_ratio <= J3_HI
    j3_cell = {c: s_tau["median"] / v for c, v in WC_TH40_CELL.items()}
    j4_pass = s_late_dk["median"] > s_late_wt["median"]

    verdicts = {
        "判1_τ_decay池内可携带": {
            "cv": s_tau["cv"], "range_ratio": s_tau["range_ratio"],
            "线": f"CV≤{J1_CV} 且 max/min≤{J1_RANGE}", "过": bool(j1_pass)},
        "判2_Po_peak池内可携带": {
            "cv": s_po["cv"], "range_ratio": s_po["range_ratio"],
            "线": f"CV≤{J2_CV} 且 max/min≤{J2_RANGE}", "过": bool(j2_pass)},
        "判3_跨模态对拍": {
            "中位τ_decay_ms": s_tau["median"], "全细胞锚τ_h40_ms": WC_TH40_POOL,
            "比值": j3_ratio, "线": f"比值∈[{J3_LO},{J3_HI}]", "过": bool(j3_pass),
            "逐细胞比值_登记": j3_cell},
        "判4_方向性自检_ΔKPQ": {
            "ΔKPQ_late中位": s_late_dk["median"], "WT_late中位": s_late_wt["median"],
            "线": "ΔKPQ中位 > WT中位", "过": bool(j4_pass)},
    }

    # ---- outcome classification (preregistration section 5) ----
    if not j4_pass:
        ending = "D"
    elif j1_pass and j2_pass and j3_pass:
        ending = "A"
    elif j1_pass and j2_pass and not j3_pass:
        ending = "B"
    else:
        ending = "C"
    ending_txt = {
        "A": "结局A：−40mV单点可携带+跨模态一致；全细胞脚部崩溃坐实为记录伪迹，N-arm折中路径开",
        "B": "结局B：单通道池内自洽但与全细胞锚系统偏离；全细胞τ_h(−40)锚降级为登记值",
        "C": "结局C：单通道池内散布失控，−40mV单点不可携带；N-arm永久关闭终审",
        "D": "结局D：提取表方向性错误，整套论文参数表不可信，转二阶自建管线",
    }[ending]

    # ---- print ----
    print("\n[crit-1] tau_decay within-pool scatter (multi-channel, n=%d)" % s_tau["n"])
    print(f"  median {s_tau['median']:.3f} ms | CV {s_tau['cv']:.3f} (limit<={J1_CV}) | "
          f"max/min {s_tau['range_ratio']:.2f} (limit<={J1_RANGE}) -> {'pass' if j1_pass else 'fail'}")
    print(f"[crit-2] Po_peak within-pool scatter (single-channel, n=%d)" % s_po["n"])
    print(f"  median {s_po['median']:.4f} | CV {s_po['cv']:.3f} (limit<={J2_CV}) | "
          f"max/min {s_po['range_ratio']:.2f} (limit<={J2_RANGE}) -> {'pass' if j2_pass else 'fail'}")
    print(f"[crit-3] cross-modality check: single-channel median {s_tau['median']:.3f} ms vs whole-cell anchor "
          f"{WC_TH40_POOL:.3f} ms -> ratio {j3_ratio:.3f} (limit[{J3_LO},{J3_HI}])"
          f"-> {'pass' if j3_pass else 'fail'}")
    for c, r in j3_cell.items():
        print(f"       registry cell{c}: ratio {r:.3f}")
    print(f"[crit-4] directionality self-check: dKPQ late% median {s_late_dk['median']:.3f} vs "
          f"WT {s_late_wt['median']:.3f} -> {'pass' if j4_pass else 'fail'}")
    print(f"\n[registry] t_peak median {s_tpeak['median']:.3f} ms ({s_tpeak['min']:.3f}-{s_tpeak['max']:.3f})")
    print(f"[registry] VMR median {s_vmr['median']:.3f} | DKL median {s_dkl['median']:.3f}")
    print(f"[registry] recovery P2/P1 median {s_rc['median']:.3f} | lido block ratio median {s_ld['median']:.3f}")
    print(f"[registry] NaV1.6 late% median {s_late_16['median']:.3f} (n={s_late_16['n']})")
    print("[registry] WT late% batch subgroups:")
    for _, r in subgrp.iterrows():
        print(f"       {r['batch']}: n={int(r['n'])} median {r['late_pct_median']:.3f}%")
    print(f"\n>>> {ending_txt}")

    # ---- save ----
    result = {"verdicts": verdicts, "ending": ending, "ending_txt": ending_txt,
              "stats": {"tau_decay": s_tau, "po_peak": s_po, "t_peak": s_tpeak,
                        "vmr": s_vmr, "dkl": s_dkl, "late_wt": s_late_wt,
                        "late_dkpq": s_late_dk, "recovery_p2p1": s_rc,
                        "lido_ratio": s_ld, "late_nav16": s_late_16},
              "subgroup_late": subgrp.to_dict("records"),
              "note": "first-order verdict: uses the paper's own parameter extraction tables, not re-checked by our ABF pipeline; temperature not annotated, registered as room temperature 22±2°C"}
    with open(str(OUT_BASE) + "_结果.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    det = pd.concat([
        mc_tau.assign(modality="mult-ch", metric="tau_decay_ms",
                      value=mc_tau["tau_decay_ms"]),
        sc.assign(modality="single-ch", metric="tau_decay_ms",
                  value=sc["tau_decay_ms"]),
        sc.assign(modality="single-ch", metric="po_peak", value=sc["po_peak"]),
        mc_pl.assign(modality="mult-ch", metric="late_pct", value=mc_pl["late_pct"]),
        dk.assign(modality="dKPQ", metric="late_pct", value=dk["late_pct"]),
    ])[["file", "modality", "metric", "value"]]
    det.to_csv(str(OUT_BASE) + "_结果.csv", index=False, encoding="utf-8-sig")

    # ---- figure ----
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    ax = axes[0]
    ax.hist(mc_tau["tau_decay_ms"], bins=20, color="#4C72B0", edgecolor="w")
    ax.axvline(s_tau["median"], color="r", ls="--", label=f"median {s_tau['median']:.2f}")
    ax.set_title(f"crit-1 tau_decay within-pool scatter\nCV={s_tau['cv']:.3f} max/min={s_tau['range_ratio']:.2f} "
                 f"-> {'pass' if j1_pass else 'fail'}")
    ax.set_xlabel("τ_decay (ms)"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(sc["po_peak"], bins=12, color="#55A868", edgecolor="w")
    ax.axvline(s_po["median"], color="r", ls="--", label=f"median {s_po['median']:.3f}")
    ax.set_title(f"crit-2 Po_peak within-pool scatter\nCV={s_po['cv']:.3f} max/min={s_po['range_ratio']:.2f} "
                 f"-> {'pass' if j2_pass else 'fail'}")
    ax.set_xlabel("single-channel peak Po"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(mc_tau["tau_decay_ms"], bins=20, color="#4C72B0", edgecolor="w", alpha=.7,
            label="single-channel ensemble tau_decay")
    ax.axvline(WC_TH40_POOL, color="k", lw=2, label=f"whole-cell pooled anchor {WC_TH40_POOL:.2f}")
    for c, v in WC_TH40_CELL.items():
        ax.axvline(v, color="gray", ls=":", lw=1)
    ax.axvspan(WC_TH40_POOL * J3_LO, WC_TH40_POOL * J3_HI, color="g", alpha=.12,
               label=f"pass window [{WC_TH40_POOL*J3_LO:.2f},{WC_TH40_POOL*J3_HI:.2f}]")
    ax.set_title(f"crit-3 cross-modality check ratio={j3_ratio:.3f} -> {'pass' if j3_pass else 'fail'}")
    ax.set_xlabel("τ (ms)"); ax.legend(fontsize=8)

    ax = axes[3]
    bp = ax.boxplot([pd.concat([sc['late_pct'], mc_pl['late_pct']]), dk["late_pct"]],
                    labels=["WT", "ΔKPQ"], showfliers=True)
    ax.set_yscale("log")
    ax.set_title(f"crit-4 directionality self-check late%\ndKPQ {s_late_dk['median']:.2f} vs WT "
                 f"{s_late_wt['median']:.2f} -> {'pass' if j4_pass else 'fail'}")
    ax.set_ylabel("late INa (% of peak, log)")

    fig.suptitle(f"Na-4 single-point anchoring verdict (first order, paper extracted tables) -> {ending_txt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(OUT_BASE) + "_结果.png", dpi=150, bbox_inches="tight")
    print(f"\n  results saved: {OUT_BASE}_结果.json/.csv/.png")


if __name__ == "__main__":
    main()
