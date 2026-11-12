# -*- coding: utf-8 -*-
"""
官方 acell 特征对拍 v5 —— 验证我们的提取器 vs Channelpedia 官方管线
官方字段: Activation/repetition[rep]/phase[*]/NI/{Volt, Act_Tau}
哨兵值过滤: Act_Tau ∈ {100, 3099.7} 或 <=1 或 >=5000 -> 视为拟合失败剔除
口径: 我方 tau_act r2>=0.7; 逐细胞逐电压配对; log10 空间相关与中位比
产出: features/validate_acell_v5.json + validate_acell_v5.csv + validate_acell_v5.png
"""
import json, re, sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
F = ROOT / "features"
ACJ = ROOT / "acell_full"
SENTINELS = {100.0, 3099.7, 3099.7000000000003}

def official_tau(cid):
    """返回 [(V, tau_official)] 列表 (activation step 相)"""
    p = ACJ / f"c{cid}.json"
    if not p.exists(): return []
    try:
        j = json.load(p.open(encoding="utf-8", errors="ignore"))
        a = j["aCell"]
        repid = a.get("isGood_param", {}).get("repID")
        reps = a.get("Activation", {}).get("repetition", [])
        if not reps: return []
        # 优先 isGood rep (1-based), 其余reps兜底
        order = [min(max((repid or 2) - 1, 0), len(reps) - 1)]
        order += [i for i in range(len(reps)) if i != order[0]]
        for idx in order:
            rep = reps[idx]
            phases = rep.get("phase", [])
            cands = []
            for ph in phases:
                ni = ph.get("NI", {}) if isinstance(ph, dict) else {}
                if not isinstance(ni, dict): continue
                V, tau = ni.get("Volt"), ni.get("Act_Tau")
                if V and tau and len(V) == len(tau) and len(V) >= 8:
                    cands = [(float(v), float(t)) for v, t in zip(V, tau)
                             if t not in SENTINELS and 1 < t < 5000]
                    if len(cands) >= 5: return cands
        return []
    except Exception:
        return []

def main():
    df = pd.read_csv(F / "features_corpus_v5.csv", low_memory=False)
    ta = df[(df.parameter == "tau_act") & (df.r2 >= 0.7) & (df.value > 0)]
    cells = ta.cell_id.unique()
    print(f"我方有 tau_act 的细胞: {len(cells)}")
    with ProcessPoolExecutor(max_workers=8) as ex:
        offs = list(ex.map(official_tau, cells, chunksize=64))
    off_map = {c: o for c, o in zip(cells, offs) if o}
    print(f"官方有 Act_Tau 的细胞: {len(off_map)}")
    rows = []
    ta_d = {(r.cell_id, float(r.voltage_mV)): r.value for r in ta.itertuples()}
    for cid, pairs in off_map.items():
        for v, t_off in pairs:
            t_our = ta_d.get((cid, v))
            if t_our:
                rows.append(dict(cell_id=cid, voltage_mV=v, tau_our=t_our, tau_off=t_off))
    M = pd.DataFrame(rows)
    M.to_csv(F / "validate_acell_v5.csv", index=False)
    print(f"配对点数: {len(M)}, 覆盖细胞: {M.cell_id.nunique()}")
    lo = np.log10(M.tau_our.values); lf = np.log10(M.tau_off.values)
    r = float(np.corrcoef(lo, lf)[0, 1])
    ratio = M.tau_our / M.tau_off
    res = dict(n_pairs=len(M), n_cells=int(M.cell_id.nunique()),
               pearson_log10=r, median_ratio_our_off=float(ratio.median()),
               within_2x=float(ratio.between(0.5, 2).mean()),
               within_1_25x=float(ratio.between(0.8, 1.25).mean()))
    # 逐电压中位比
    per_v = M.groupby("voltage_mV").apply(lambda g: (g.tau_our/g.tau_off).median()).to_dict()
    res["per_voltage_median_ratio"] = {str(k): float(v) for k, v in per_v.items()}
    json.dump(res, open(F / "validate_acell_v5.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:800])

    # 散点图
    import matplotlib
    matplotlib.use("Agg")
    import sys as _s
    _s.path.insert(0, str(Path(_s.executable).parent.parent.parent))
    try:
        from daimon_runtime import setup_plot; setup_plot()
    except Exception:
        pass
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(M.tau_off, M.tau_our, s=1, alpha=0.15, c="0.2")
    lim = [0.3, 5000]
    ax.plot(lim, lim, "r-", lw=1, label="1:1")
    ax.plot(lim, [2*l for l in lim], "r--", lw=0.7, label="2x")
    ax.plot(lim, [0.5*l for l in lim], "r--", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("official Act_Tau (ms)")
    ax.set_ylabel("our tau_act (ms)")
    ax.set_title(f"extractor validation, n={len(M)}, r={r:.3f}")
    ax.legend()
    fig.savefig(F / "validate_acell_v5.png", bbox_inches="tight", dpi=150)
    print("saved figure")

if __name__ == "__main__":
    main()
