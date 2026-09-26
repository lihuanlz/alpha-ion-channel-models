# -*- coding: utf-8 -*-
"""
Nα-drug · deepseek second-review supplement pack (2026-09-15)
  Part A: Nav morphology self-check (is the kappa* metric trustworthy on the Nav fast-peak shape within a 20 ms window)
  Part B: R_in gate sensitivity (does the kappa*/eps_x distribution change across R_in ranges)
  Part C: ATX-II verification (metadata evidence, no computation; conclusion written to JSON)
Reuses the same-code functions from the original verdict script (module import), guaranteeing zero code-path deviation.
"""
import importlib.util, json, os, sys
import numpy as np
from scipy.signal import medfilt
from scipy.stats import spearmanr

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
MOD = os.path.join(ROOT, "α模型", "2026-09-15_Nα药_Nav15药物形状调制判决.py")
OUT = os.path.join(ROOT, "α模型", "2026-09-15_Nα药_补做包_结果.json")

spec = importlib.util.spec_from_file_location("nayao", MOD)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)  # main() is __name__-guarded, import has no side effects

rng = np.random.default_rng(20260915)

# ================= Part A: Nav morphology self-check =================
DT = 0.05            # ms，20 kHz
T = np.arange(0.0, 20.0, DT)   # 20 ms window, 400 points, same as WF

def nav_wf(tm=0.5, tf=5.0, ts=300.0, fslow=0.15):
    """Nav1.5 fast-peak shape: m^3 activation x double-exponential inactivation. Inward (negative) before peak normalization."""
    m = 1.0 - np.exp(-T / tm)
    h = (1.0 - fslow) * np.exp(-T / tf) + fslow * np.exp(-T / ts)
    return -(m ** 3) * h

def pipe(w, ntr=5, sig=0.02):
    """Mirror pipeline: average n noisy traces -> medfilt3 -> |peak| normalization."""
    tr = [w + rng.normal(0, sig, len(w)) for _ in range(ntr)]
    w2 = medfilt(np.mean(tr, axis=0), 3)
    pk = np.min(w2)
    return w2 / abs(pk)

def sim_case(wd_maker, n=300, sig=0.02):
    w0 = nav_wf()
    kaps, exs = [], []
    for _ in range(n):
        wc = pipe(w0, sig=sig)
        wd = pipe(wd_maker(w0), sig=sig)
        ex, kap, _ = M.eps_kappa(wc, wd)
        kaps.append(kap); exs.append(ex)
    kaps = np.array(kaps); exs = np.array(exs)
    return {"med_abs_kappa": float(np.median(np.abs(kaps))),
            "med_kappa": float(np.median(kaps)),
            "q95_abs_kappa": float(np.percentile(np.abs(kaps), 95)),
            "med_eps_x": float(np.median(exs))}

def partA():
    out = {}
    for sig in (0.0, 0.02, 0.05):
        tag = f"σ={sig}"
        out[tag] = {
            "T0_空分布(同形态双实现)": sim_case(lambda w: w, sig=sig),
            "T1_纯幅度×0.4":          sim_case(lambda w: w * 0.4, sig=sig),
            "T2a_真warp+0.05":        sim_case(lambda w: np.interp(T, T / 1.05, w), sig=sig),
            "T2b_真warp+0.20":        sim_case(lambda w: np.interp(T, T / 1.20, w), sig=sig),
            "T2c_真warp-0.20":        sim_case(lambda w: np.interp(T, T / 0.80, w), sig=sig),
            "T3_τf加快20%(5→4ms)":    sim_case(lambda w: nav_wf(tf=4.0), sig=sig),
            "T4_慢分量权重15→30%":     sim_case(lambda w: nav_wf(fslow=0.30), sig=sig),
            "T5_平台偏移+0.05":       sim_case(lambda w: w + 0.05 * abs(np.min(w)), sig=sig),
        }
        out[tag]["T2a_回收med_kappa"] = out[tag]["T2a_真warp+0.05"]["med_kappa"]
        out[tag]["T2b_回收med_kappa"] = out[tag]["T2b_真warp+0.20"]["med_kappa"]
        out[tag]["T2c_回收med_kappa"] = out[tag]["T2c_真warp-0.20"]["med_kappa"]
    return out

# ================= Part B: R_in gate sensitivity =================
def lite_cell(row):
    """Same code as the v1.0 coarse gate, but the R_in gate is not applied; r_in is still recorded."""
    import re
    if re.search(r"died|lost clamp|unstable", str(row.get("note", "")), re.I):
        return None
    files, _ = M.cell_files("INa_P", row["drug"].strip(), row["file_id"])
    if files is None:
        return None
    import pyabf
    sweeps = []
    for fp in files:
        a = pyabf.ABF(fp)
        for s in range(a.sweepCount):
            sweeps.append(M.parse_sweep(a, s))
    r1 = M.parse_range(row.get("sw1", "")); r2 = M.parse_range(row.get("sw2", ""))
    c1 = M.parse_conc(row.get("c1", "")); c2 = M.parse_conc(row.get("c2", ""))
    if r1 is None or c1 is None:
        return None
    ctrl_end = r1[0] - 1
    if ctrl_end < 8 or r1[1] > len(sweeps) or (r2 and r2[1] > len(sweeps)):
        return None
    def cond_traces(rr):
        idx = list(range(rr[0], rr[1] + 1))[-M.LASTN:]
        out = [M.sweep_metrics(*sweeps[g - 1]) for g in idx]
        out = [m for m in out if m]
        return out if len(out) >= 3 else None
    ctrl_tr = cond_traces((1, ctrl_end))
    if ctrl_tr is None:
        return None
    pk = np.array([m["step_pk"] for m in ctrl_tr])
    if np.std(pk) / abs(np.mean(pk)) > 0.15 or abs(np.mean(pk)) < 500:
        return None
    r_in = float(np.nanmedian([m["r_in"] for m in ctrl_tr]))
    wc, pkw = M.mean_wf(ctrl_tr)
    if wc is None:
        return None
    conds = []
    for conc, rr in [(c1, r1), (c2, r2)]:
        if conc is None or rr is None:
            continue
        tr = cond_traces(rr)
        if tr is None:
            continue
        wd, pkd = M.mean_wf(tr)
        if wd is None:
            continue
        f_step = 1.0 - abs(pkd) / abs(pkw)
        ex, kap, _ = M.eps_kappa(wc, wd)
        conds.append({"conc": conc, "f_step": f_step, "eps_x": ex, "kappa": kap})
    return {"drug": row["drug"].strip(), "file_id": row["file_id"][:40], "r_in": r_in, "conds": conds}

def partB():
    meta = json.load(open(M.CELLS_JSON, encoding="utf-8"))
    rows = [r for r in meta["INa_P"] if str(r.get("drug", "")).strip()]
    recs = []
    for j, r in enumerate(rows):
        rec = lite_cell(r)
        if rec:
            recs.append(rec)
        print(f"  [{j+1}/{len(rows)}] {r['drug'][:12]} r_in=" + (f"{rec['r_in']:.0f}" if rec else "skip"), flush=True)
    # Flatten to condition level; pool = f_step>=0.15 and valid kappa* (same pool rule as criterion 2)
    pool = []
    for rec in recs:
        for cd in rec["conds"]:
            if cd["f_step"] >= 0.15 and cd["kappa"] is not None:
                pool.append({"r_in": rec["r_in"], "kappa": cd["kappa"], "eps_x": cd["eps_x"],
                             "drug": rec["drug"], "conc": cd["conc"]})
    ak = np.array([abs(p["kappa"]) for p in pool])
    ri = np.array([p["r_in"] for p in pool])
    ex = np.array([p["eps_x"] for p in pool])
    def gated(lo, hi):
        m = (ri >= lo) & (ri <= hi)
        return {"n": int(m.sum()), "med_abs_kappa": float(np.median(ak[m])) if m.sum() >= 3 else None}
    bins = {}
    for lo, hi, tag in [(0, 50, "<50"), (50, 100, "50–100"), (100, 300, "100–300"),
                        (300, 500, "300–500"), (500, 1e9, ">500(原剔除)")]:
        m = (ri >= lo) & (ri < hi)
        bins[tag] = {"n": int(m.sum()),
                     "med_abs_kappa": float(np.median(ak[m])) if m.sum() >= 3 else None,
                     "med_eps_x": float(np.median(ex[m])) if m.sum() >= 3 else None}
    return {
        "n_cells": len(recs), "n_pool": len(pool),
        "sanity_原门50_500": gated(50, 500),
        "gate_100_500": gated(100, 500), "gate_50_300": gated(50, 300),
        "gate_50_1000": gated(50, 1000), "gate_不限": gated(0, 1e9),
        "spearman_rin_abskappa": float(spearmanr(ri, ak).statistic) if len(pool) >= 5 else None,
        "spearman_rin_abskappa_p": float(spearmanr(ri, ak).pvalue) if len(pool) >= 5 else None,
        "spearman_rin_epsx": float(spearmanr(ri, ex).statistic) if len(pool) >= 5 else None,
        "spearman_rin_epsx_p": float(spearmanr(ri, ex).pvalue) if len(pool) >= 5 else None,
        "bins": bins,
    }

# ================= Part C: ATX-II verification (metadata) =================
PARTC = {
    "INa_P_header": "Notes: 1) All recordings were carried out at 37oC; ... （无 ATX-II）",
    "INa_L_header": "Note: 1) Agonist = ATX II; ... （整组共用 150 nM ATX-II）",
    "结论": [
        "判1/判2 池 = INa_P 组（f_step≥0.15 条件）→ 元数据载明无 ATX-II → κ*=0.20 与 ATX-II 无关，deepseek 审计点1 的 ATX-II 推断对判2 不成立。",
        "判3 = INa_L 组内 Δf=f_ramp−f_step，同一细胞同一浓度下两协议均含 ATX-II → 差分对消主效应；残余候选（ATX-II 对两协议差异影响）无 ATX 自由臂不可排除，登记为数据限制。",
    ],
}

if __name__ == "__main__":
    res = {"PartA_Nav形态自检": partA()}
    print("[A] done", flush=True)
    res["PartB_R_in敏感性"] = partB()
    res["PartC_ATXII核实"] = PARTC
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[saved] {OUT}", flush=True)
