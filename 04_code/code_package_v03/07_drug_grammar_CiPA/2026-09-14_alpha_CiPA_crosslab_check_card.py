# 2026-09-14_alpha-model_CiPA_cross-lab_verification_card.py
# ============================================================================
# α model · CiPA cross-laboratory third-party verification card (criteria pinned before the run, see
#   预注册_α模型_CiPA跨实验室核实卡_2026-09-14.md, under "结果" in the same directory)
#
# Data source: local run package data/cipa/*.npz (pimozide subtracted, lab3/4/5, 35-37.5°C)
# Model: formal assembly engine with four tables (population medianized + Q10 temperature bridge + E_rev=-92.1); G is the only per-cell
#   free quantity (closed-form least squares). Simulation dt=0.2ms (npz native), exact exponential update (same formula as the engine).
#
# Criteria (pinned):
#   feature pass: sim/meas in [0.5, 2.0] (not judgeable below |meas|<0.02nA)
#   cell pass: F1(+40 plateau) AND F2(ramp -20) AND F3(ramp -70)
#   dataset pass line: main-pool overall pass rate >= 2/3 and per-lab >= 1/2 (main line Q10=2.5, E_rev=-92.1)
#
# Run: Spyder %runfile '...py' --wdir. Outputs JSON + PNG to this directory.
# ============================================================================
import os
import json
import hashlib
import importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG_PATH = os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py")
CIPA_DIR = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/结果/"
            "hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包/data/cipa")

DT_NPZ = 2e-4          # npz native sampling 0.2 ms (5 kHz)
E_REV_CIPA = -92.1     # Nernst 37°C (149 measurement chain)
T_MAIN = 21.5          # main dataset temperature (Beattie 2018 on record)
T_LAB = {"lab3": 35.5, "lab4": 36.1, "lab5": 37.4}   # preregistration section 2
Q10_MAIN = 2.5
RATIO_LO, RATIO_HI = 0.5, 2.0
MIN_MEAS = 0.02        # nA, below which a feature is not judgeable
MIN_PLATEAU = 0.05     # nA, below which the cell signal is insufficient and excluded
MIN_SWEEPS = 8

# ---------------------------------------------------------------- engine loading
spec = importlib.util.spec_from_file_location("alpha_engine", ENG_PATH)
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)

amp = json.load(open(eng.F_AMP, encoding="utf-8"))
hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
inact = json.load(open(eng.F_INACT, encoding="utf-8"))
hss = json.load(open(eng.F_HSS, encoding="utf-8"))

# ------------------------------------------------- population-median ingredients (computed at runtime)
yss = {}
for r in amp["rows"]:
    if r["v"] in (-70, -60, -50):
        yss.setdefault(r["v"], {})[r["cell"]] = max(r["y_ss"], 1e-4)
YMED = {v: float(np.median(list(d.values()))) for v, d in yss.items()}

cells9 = sorted({r["cell"] for r in amp["rows"]})
h40s, h50s = [], []
for c in cells9:
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", c, "inactivation")
    t9 = eng.build_tabs(c, amp, hook, inact, hss, Ii, Vi)
    if t9["h40"] and np.isfinite(t9["h40"]):
        h40s.append(float(t9["h40"]))
    if t9["h50"] and np.isfinite(t9["h50"]):
        h50s.append(float(t9["h50"]))
H40_POP = float(np.median(h40s))
H50_POP = float(np.median(h50s)) if h50s else None
print(f"[population table] y_ss median -70/-60/-50 = {YMED.get(-70):.4f}/{YMED.get(-60):.4f}"
      f"/{YMED.get(-50):.4f}  h40_pop={H40_POP:.4f}  h50_pop={H50_POP}", flush=True)


def build_pop_tabs(T_lab, q10, E_rev):
    """Same formula and anchors as the engine's build_tabs, except: per-cell quantities -> nine-cell medians; tau tables divided by Q10^(dT/10)."""
    hs = hook["summary"]
    hB1 = {v: float(hss["gears"][str(v)]["med"])
           for v in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30)}
    m_anchors = {-130: 1e-4, -120: 1e-4, -110: 1e-4, -100: 1e-4, -90: 1e-4,
                 -80: min(1.0, YMED[-70] * 0.5 / hB1[-80]),
                 -70: min(1.0, YMED[-70] / hB1[-70]),
                 -60: min(1.0, YMED[-60] / hB1[-60]),
                 -50: min(1.0, YMED[-50] / hB1[-50]),
                 -40: min(1.0, YMED[-50] * 3.0 / hB1[-40]),
                 -30: min(1.0, YMED[-50] * 8.0 / hB1[-30]),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    tm_anchors = {-130: hs["-120"]["td_med"], -120: hs["-120"]["td_med"],
                  -110: hs["-110"]["td_med"], -100: hs["-100"]["td_med"],
                  -90: 0.072, -80: 0.24,
                  -70: amp["tau_used"]["TM"]["-70"], -60: amp["tau_used"]["TM"]["-60"],
                  -50: amp["tau_used"]["TAU_L"]["-50"], -40: amp["tau_used"]["TAU_L"]["-40"],
                  -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
                  20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
    h40 = H40_POP
    h_anchors = {-130: 1.0, -100: 1.0}
    h_anchors.update(hB1)
    h_anchors[40] = h40
    h_anchors[50] = H50_POP if H50_POP else h40
    h_anchors[60] = H50_POP if H50_POP else h40
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: 0.0065, -70: 0.0070, -60: 0.0070, -50: 0.0089,
                  -40: 0.0144, -30: 0.0144, -20: 0.05, 0: 0.0015, 60: 0.0015}
    fac = q10 ** ((T_lab - T_MAIN) / 10.0)
    tm_anchors = {k: v / fac for k, v in tm_anchors.items()}
    th_anchors = {k: v / fac for k, v in th_anchors.items()}
    return dict(m_ss=eng.Tab(m_anchors), tau_m=eng.Tab(tm_anchors),
                h_ss=eng.Tab(h_anchors), tau_h=eng.Tab(th_anchors), E_rev=E_rev)


def forward_dt(V, tabs, dt):
    """Same formula as the engine's forward (exact exponential update), only dt switched to the npz-native 0.2ms."""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hs = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-dt / tm)
    eh = np.exp(-dt / th)
    n = len(V)
    m = np.empty(n)
    h = np.empty(n)
    m0 = ms[0]
    h0 = hs[0]
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hs[i] + (h0 - hs[i]) * eh[i]
        m[i] = m0
        h[i] = h0
    return m, h


# ------------------------------------------------------------ file inventory and grouping
def lab_of(name):
    if name.startswith("Hh11"):
        return "lab4"
    if name.startswith(("2020_", "207", "208")):
        return "lab5"
    if name[:3].isdigit() and "_" in name:      # 143_B_04_AY_...
        return "lab3"
    return "manual"                              # lab1_/lab5_/labX_ reported as a separate group


files = sorted(f for f in os.listdir(CIPA_DIR) if f.endswith(".npz"))
pool_main, pool_manual = [], []
seen_hash = {}
dupes = []
for f in files:
    d = np.load(os.path.join(CIPA_DIR, f))
    i_arr = d["i"]
    iok = i_arr[np.isfinite(i_arr)]
    hh = hashlib.md5(iok.tobytes()).hexdigest() if len(iok) else "empty"
    if hh in seen_hash:
        dupes.append((f, seen_hash[hh]))
        continue
    seen_hash[hh] = f
    (pool_manual if lab_of(f) == "manual" else pool_main).append(f)

print(f"[inventory] total files {len(files)}; hash-dedup collapsed {len(dupes)} items: {dupes}", flush=True)
print(f"[inventory] main pool {len(pool_main)} (lab3 {sum(1 for f in pool_main if lab_of(f)=='lab3')} "
      f"lab4 {sum(1 for f in pool_main if lab_of(f)=='lab4')} "
      f"lab5 {sum(1 for f in pool_main if lab_of(f)=='lab5')}）；"
      f"separate report group {len(pool_manual)}", flush=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    pool_main, pool_manual = pool_main[:1], []
    print("[smoke] SMOKE=1: running only the first main-pool cell, output carries the _冒烟 suffix, not a verdict", flush=True)

# ------------------------------------------------------------ feature extraction (pinned window positions)
MS = 5  # points per ms (0.2ms/point)


def sweep_onsets(v):
    up = np.where((v[:-1] < 30.0) & (v[1:] >= 30.0))[0] + 1
    return up


def extract(v, i_sig, t0):
    """Return dict(F1..F5) or None (sweep invalid). Window positions in ms from t0 (pinned)."""
    def seg(ms0, ms1):
        a, b = t0 + int(ms0 * MS), min(t0 + int(ms1 * MS), len(i_sig))
        return a, b
    a1, b1 = seg(400, 495)
    if b1 <= a1 or np.isnan(i_sig[a1:b1]).any():
        return None
    F1 = float(np.mean(i_sig[a1:b1]))
    # ramp segment: take the v window within t0+500..640 ms
    ra, rb = seg(500, 640)
    if rb <= ra:
        return None
    vv = v[ra:rb]
    ii = i_sig[ra:rb]
    m2 = (vv >= -21.0) & (vv <= -19.0) & np.isfinite(ii)
    m3 = (vv >= -71.0) & (vv <= -69.0) & np.isfinite(ii)
    if m2.sum() < 2 or m3.sum() < 2:
        return None
    F2 = float(np.mean(ii[m2]))
    F3 = float(np.mean(ii[m3]))
    a4, b4 = seg(800, 850)
    F4 = float(np.mean(i_sig[a4:b4])) if b4 <= len(i_sig) and b4 > a4 and not np.isnan(i_sig[a4:b4]).any() else np.nan
    # F5: +40 rising-phase t50 (50% of its own plateau)
    half = 0.5 * F1
    a5 = t0
    b5 = min(t0 + int(400 * MS), len(i_sig))
    seg_i = i_sig[a5:b5]
    above = np.where(np.isfinite(seg_i) & (seg_i >= half))[0]
    F5 = float(above[0] / MS) if len(above) else np.nan
    return dict(F1=F1, F2=F2, F3=F3, F4=F4, F5=F5)


def run_cell(fn, tabs_by_cfg, cfgs):
    """Per-config simulation + features + closed-form G fit. Returns a per-cfg dict."""
    d = np.load(os.path.join(CIPA_DIR, fn))
    v, i_meas = d["v"].astype(float), d["i"].astype(float)
    t0s = sweep_onsets(v)
    out = {}
    for cfg in cfgs:
        tabs = tabs_by_cfg[cfg]
        E = tabs["E_rev"]
        m, h = forward_dt(v, tabs, DT_NPZ)
        gden = m * h * (v - E)
        # G: closed-form LS over the +40 hold + downward ramp (t0..t0+600ms)
        num = den = 0.0
        for t0 in t0s:
            a, b = t0, min(t0 + int(600 * MS), len(v))
            gg = gden[a:b]
            ii = i_meas[a:b]
            ok = np.isfinite(ii)
            num += float(np.sum(gg[ok] * ii[ok]))
            den += float(np.sum(gg[ok] ** 2))
        G = num / den if den > 0 else np.nan
        i_sim = G * gden
        rows = []
        for k, t0 in enumerate(t0s):
            fm = extract(v, i_meas, t0)
            fs = extract(v, i_sim, t0)
            if fm is None or fs is None:
                continue
            row = {"sweep": k + 1}
            for key in ("F1", "F2", "F3", "F4", "F5"):
                mv, sv = fm[key], fs[key]
                row[key + "_mea"] = mv
                row[key + "_sim"] = sv
                row[key + "_r"] = (sv / mv) if (np.isfinite(mv) and abs(mv) >= MIN_MEAS) else None
            rows.append(row)
        out[cfg] = dict(G=G, n_sweeps=len(t0s), n_valid=len(rows), rows=rows)
    return out


def med_ratio(rows, key):
    rs = [r[key + "_r"] for r in rows if r[key + "_r"] is not None and np.isfinite(r[key + "_r"])]
    return float(np.median(rs)) if rs else None


def cell_verdict(res_main):
    """Cell verdict under the main-line config. Returns (verdict, details)."""
    if res_main["n_sweeps"] < MIN_SWEEPS:
        return "excluded_few_sweeps", {}
    rows = res_main["rows"]
    if len(rows) < MIN_SWEEPS:
        return "excluded_few_valid", {}
    f1m = [r["F1_mea"] for r in rows if np.isfinite(r["F1_mea"])]
    if not f1m or abs(float(np.median(f1m))) < MIN_PLATEAU:
        return "excluded_low_signal", {}
    det = {}
    passes = []
    for key in ("F1", "F2", "F3"):
        r = med_ratio(rows, key)
        det[key] = r
        if r is None:
            det[key + "_判"] = "不可判"
            passes.append(False)     # F1/F2/F3 main features not judgeable -> does not count as pass
        else:
            ok = RATIO_LO <= r <= RATIO_HI
            det[key + "_判"] = "过" if ok else "不过"
            passes.append(ok)
    det["F4_r"] = med_ratio(rows, "F4")
    det["F5_r"] = med_ratio(rows, "F5")
    # cumulative observation: F1 sweep25/sweep1
    if len(rows) >= 2:
        det["drift_mea"] = rows[-1]["F1_mea"] / rows[0]["F1_mea"] if rows[0]["F1_mea"] else None
        det["drift_sim"] = rows[-1]["F1_sim"] / rows[0]["F1_sim"] if rows[0]["F1_sim"] else None
    return ("pass" if all(passes) else "fail"), det


# ------------------------------------------------------------ main run
CFGS = [("Q10_2.5_E-92.1", 2.5, -92.1),
        ("Q10_2.0_E-92.1", 2.0, -92.1),
        ("Q10_3.0_E-92.1", 3.0, -92.1),
        ("Q10_2.5_E-89.1", 2.5, -89.1),
        ("Q10_2.5_E-95.1", 2.5, -95.1)]
MAIN_CFG = "Q10_2.5_E-92.1"

result = {"预注册": "预注册_α模型_CiPA跨实验室核实卡_2026-09-14.md",
          "判线": {"特征窗": [RATIO_LO, RATIO_HI], "细胞过": "F1^F2^F3",
                   "数据集过线": "主池总过率>=2/3 且每实验室>=1/2", "主线": MAIN_CFG},
          "群体表": {"y_ss_med": {str(k): v for k, v in YMED.items()},
                     "h40_pop": H40_POP, "h50_pop": H50_POP},
          "dupes": dupes, "cells": {}}

for pool_name, pool in (("main", pool_main), ("manual", pool_manual)):
    for fn in pool:
        lab = lab_of(fn)
        T = T_LAB.get(lab, 35.5)
        tabs_by_cfg = {name: build_pop_tabs(T, q, E) for (name, q, E) in CFGS}
        try:
            res = run_cell(fn, tabs_by_cfg, [c[0] for c in CFGS])
        except Exception as e:
            result["cells"][fn] = {"lab": lab, "pool": pool_name, "判定": "error", "err": str(e)}
            print(f"  {fn}: ERROR {e}", flush=True)
            continue
        verdict, det = cell_verdict(res[MAIN_CFG])
        entry = {"lab": lab, "pool": pool_name, "判定": verdict,
                 "G": {c[0]: res[c[0]]["G"] for c in CFGS},
                 "n_sweeps": res[MAIN_CFG]["n_sweeps"], "n_valid": res[MAIN_CFG]["n_valid"],
                 "main": det}
        for cname, q, E in CFGS[1:]:
            v2, d2 = cell_verdict(res[cname])
            entry[cname] = {"判定": v2, "F1": d2.get("F1"), "F2": d2.get("F2"), "F3": d2.get("F3")}
        result["cells"][fn] = entry
        print(f"  {fn} [{lab}] {verdict}  G={res[MAIN_CFG]['G']:.4f} "
              f"F1={det.get('F1')} F2={det.get('F2')} F3={det.get('F3')}", flush=True)

# ------------------------------------------------------------ summary and verdict
def summarize(pool_name):
    sel = {k: v for k, v in result["cells"].items() if v["pool"] == pool_name}
    judged = [v for v in sel.values() if v["判定"] in ("pass", "fail")]
    npass = sum(1 for v in judged if v["判定"] == "pass")
    per_lab = {}
    for lab in ("lab3", "lab4", "lab5"):
        lj = [v for v in sel.values() if v["lab"] == lab and v["判定"] in ("pass", "fail")]
        lp = sum(1 for v in lj if v["判定"] == "pass")
        per_lab[lab] = {"n": len(lj), "pass": lp,
                        "rate": (lp / len(lj)) if lj else None}
    excl = {}
    for v in sel.values():
        if v["判定"] not in ("pass", "fail"):
            excl[v["判定"]] = excl.get(v["判定"], 0) + 1
    return {"n_judged": len(judged), "n_pass": npass,
            "rate": (npass / len(judged)) if judged else None,
            "per_lab": per_lab, "excluded": excl}

summ = {"main": summarize("main"), "manual": summarize("manual")}
# sensitivity pass rates (main pool)
sens = {}
for cname, q, E in CFGS[1:]:
    sel = [v for v in result["cells"].values() if v["pool"] == "main" and v.get(cname, {}).get("判定") in ("pass", "fail")]
    sp = sum(1 for v in sel if v[cname]["判定"] == "pass")
    sens[cname] = {"n": len(sel), "pass": sp, "rate": (sp / len(sel)) if sel else None}
summ["sensitivity"] = sens
result["summary"] = summ

main = summ["main"]
line1 = (main["rate"] is not None and main["rate"] >= 2.0 / 3.0)
line2 = all((l["rate"] is not None and l["rate"] >= 0.5) for l in main["per_lab"].values() if l["n"] > 0)
if line1 and line2:
    verdict = "过线：α 模型跨实验室第三方核实成立（幅度+形状级，Q10 温度桥在案）"
elif main["rate"] is not None and main["rate"] >= 0.5:
    verdict = "部分过：按预注册 §五归因，登记局部失败坐标"
else:
    verdict = "不过：转移失败，按预注册 §五归因定位断链环节"
result["总判词"] = verdict
if SMOKE:
    result["总判词"] = "[冒烟非判决] " + verdict

print("=" * 76, flush=True)
print(f"[main line {MAIN_CFG}] main pool {main['n_pass']}/{main['n_judged']} = "
      f"{main['rate'] if main['rate'] is not None else float('nan'):.3f} (criterion >=2/3)", flush=True)
for lab, l in main["per_lab"].items():
    if l["n"]:
        print(f"    {lab}: {l['pass']}/{l['n']} = {l['rate']:.3f} (criterion >=1/2)", flush=True)
print(f"    exclusion registry: {main['excluded']}", flush=True)
for cname, s in sens.items():
    print(f"[sensitivity {cname}] {s['pass']}/{s['n']} = "
          f"{s['rate'] if s['rate'] is not None else float('nan'):.3f}", flush=True)
print(f"[overall verdict] {verdict}", flush=True)

sfx = "_冒烟" if SMOKE else ""
fjson = os.path.join(HERE, f"2026-09-14_α模型_CiPA跨实验室核实卡{sfx}_结果.json")
json.dump(result, open(fjson, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  results saved: {fjson}", flush=True)

# ------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
names = [MAIN_CFG] + [c[0] for c in CFGS[1:]]
rates = [main["rate"]] + [sens[c]["rate"] for c in names[1:]]
rates = [r if r is not None else 0 for r in rates]
ax.bar(range(len(names)), rates, color=["#c0392b"] + ["#7f8c8d"] * 4)
ax.axhline(2 / 3, ls="--", c="k", lw=1, label="criterion 2/3")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_title("main-pool pass rate: main line vs sensitivity")
ax.legend(fontsize=8)

ax = axes[0, 1]
for key, mk in (("F1", "o"), ("F2", "s"), ("F3", "^")):
    for lab, col in (("lab3", "#2980b9"), ("lab4", "#27ae60"), ("lab5", "#8e44ad")):
        xs, ys = [], []
        for fn, v in result["cells"].items():
            if v["pool"] == "main" and v["lab"] == lab and v["判定"] in ("pass", "fail"):
                r = v["main"].get(key)
                if r is not None:
                    xs.append(key)
                    ys.append(r)
        ax.scatter([f"{key}"] * len(ys), ys, marker=mk, alpha=0.6, s=18)
ax.axhline(0.5, ls="--", c="r", lw=1)
ax.axhline(2.0, ls="--", c="r", lw=1)
ax.axhline(1.0, ls="-", c="k", lw=0.5)
ax.set_yscale("log")
ax.set_title("main-line feature-ratio distribution (three labs pooled)")
ax.set_ylabel("sim/meas (log axis)")

# full-sweep overlays of two representative cells (main-line config): median ratio closest to 1 and farthest from 1
ax = axes[1, 0]
cands = [(fn, v["main"].get("F1")) for fn, v in result["cells"].items()
         if v["pool"] == "main" and v["判定"] in ("pass", "fail") and v["main"].get("F1")]
if cands:
    best = min(cands, key=lambda x: abs(np.log(x[1])))[0]
    d = np.load(os.path.join(CIPA_DIR, best))
    v, im = d["v"].astype(float), d["i"].astype(float)
    lab = lab_of(best)
    tabs = build_pop_tabs(T_LAB.get(lab, 35.5), Q10_MAIN, E_REV_CIPA)
    m, h = forward_dt(v, tabs, DT_NPZ)
    G = result["cells"][best]["G"][MAIN_CFG]
    isim = G * m * h * (v - E_REV_CIPA)
    t0s = sweep_onsets(v)
    if len(t0s) >= 3:
        t0 = t0s[2]
        sl = slice(t0 - int(200 * MS), t0 + int(1100 * MS))
        tt = (np.arange(sl.start, sl.stop) - t0) / MS
        ax.plot(tt, im[sl], "k-", lw=0.8, label="measured")
        ax.plot(tt, isim[sl], "r-", lw=0.8, label="α-model simulated")
        ax.plot(tt, v[sl] / 50, "b-", lw=0.5, alpha=0.4, label="V/50")
        ax.set_title(f"representative cell {best} (G={G:.3f})")
        ax.set_xlabel("ms from the +40 up-step")
        ax.legend(fontsize=8)

ax = axes[1, 1]
for lab, col in (("lab3", "#2980b9"), ("lab4", "#27ae60"), ("lab5", "#8e44ad")):
    gs = [v["G"][MAIN_CFG] for fn, v in result["cells"].items()
          if v["pool"] == "main" and v["lab"] == lab and np.isfinite(v["G"][MAIN_CFG])]
    if gs:
        ax.hist(gs, bins=12, alpha=0.5, color=col, label=f"{lab} (n={len(gs)})")
ax.set_title("per-cell G fit distribution (nA/mV)")
ax.legend(fontsize=8)

fig.tight_layout()
fpng = os.path.join(HERE, f"2026-09-14_α模型_CiPA跨实验室核实卡{sfx}.png")
fig.savefig(fpng, dpi=130, bbox_inches="tight")
print(f"  figure saved: {fpng}", flush=True)
