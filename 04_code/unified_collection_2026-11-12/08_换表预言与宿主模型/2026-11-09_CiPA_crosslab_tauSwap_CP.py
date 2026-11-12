# 2026-11-09_CiPA_crosslab_tauSwap_CP.py
# ============================================================================
# CiPA 跨实验室核实 · Channelpedia τ_m 换表测试
#
# 问题：把原 α 模型（Beattie9 拟合）的 τ_m(V) 表换成 Channelpedia 群体实测
#       （35°C，n≈100+ 细胞/档，QC: flag=active + 剖面CI成功 + r2>=0.9），
#       其余（m_ss/h_ss/τ_h 锚点、E_rev=-92.1、Q10=2.5、特征窗、判线）全部冻结，
#       同一预注册判线下判决是否仍然过线？
#
# 三配置（同数据同判线）：
#   ORIG    —— 原封卷 τ_m 表（内控，应复现 65/68）
#   CP_full —— τ_m(-70..+60) 全换 Channelpedia（-70..-30 τ_deact；-20..+60 τ_act）
#   CP_act  —— 仅 τ_m(-20..+60) 换 Channelpedia（归因臂）
#
# Channelpedia 35°C QC 中位（ms）×3.445 折回 21.5°C 基准（Q10=2.5 桥与原卡同构）：
#   τ_deact: -70:37.7 -60:43.4 -50:56.7 -40:80.0 -30:85.1
#   τ_act:   -20:280.7 -10:323.0 0:268.0 10:150.6 20:90.2 30:64.1 40:57.5 50:71.5 60:105.6
# 稳态表/失活门：Channelpedia 侧 Boltzmann 拟合 QC 全灭（Vhalf_act 25°C 82→7→1），
#   故 m_ss/h_ss/τ_h 保持原封卷锚点——这本身即"中点不可识别、时间常数保守"的实证。
#
# 数据：归档 npz（pimozide subtracted, lab3/4/5 + manual 池）
# 判线（冻结）：特征 sim/meas∈[0.5,2.0]（|meas|<0.02nA 不判）；
#   细胞过 = F1∧F2∧F3；数据集过线 = 主池≥2/3 且每实验室≥1/2
# ============================================================================
import os, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIPA_DIR = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/"
            "04_细胞线4/_归档/结果/hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包/data/cipa")

DT_NPZ = 2e-4
E_REV_CIPA = -92.1
T_MAIN = 21.5
T_LAB = {"lab3": 35.5, "lab4": 36.1, "lab5": 37.4}
Q10_MAIN = 2.5
RATIO_LO, RATIO_HI = 0.5, 2.0
MIN_MEAS = 0.02
MIN_PLATEAU = 0.05
MIN_SWEEPS = 8

# ------------------------------------------------ 冻结的原封卷成分（从结果 JSON 抄录）
YMED = {-70: 0.029477868155508298, -60: 0.03945937095037588, -50: 0.08393733831866093}
H40_POP = 0.014753124449143482
H50_POP = 0.011385195620532567
hB1 = {-80: 0.7980112179405232, -70: 0.42458878766834396, -60: 0.3098949595189373,
       -50: 0.2548959330145549, -40: 0.20781431816537388, -30: 0.16969269623291475,
       -20: 0.13749178922473596, -10: 0.11350819372481134, 0: 0.0904564201253725,
       10: 0.07195623486448417, 20: 0.053275325401300214, 30: 0.039430262401624006}
# 原 τ_h 锚点（s, @21.5°C）：tr_med -120/-110/-100 + 手钉值
TH_ORIG = {-130: 0.0030376559246143267, -120: 0.0030376559246143267,
           -110: 0.004254082928519305, -100: 0.005045399817563913,
           -90: 0.0065, -70: 0.0070, -60: 0.0070, -50: 0.0089,
           -40: 0.0144, -30: 0.0144, -20: 0.05, 0: 0.0015, 60: 0.0015}
# 原 τ_m 锚点（s, @21.5°C）
TM_ORIG = {-130: 0.03033214284254048, -120: 0.03033214284254048,
           -110: 0.049199997974127406, -100: 0.08577126043784405,
           -90: 0.072, -80: 0.24,
           -70: 0.8127, -60: 1.20549, -50: 10.14073, -40: 24.0539,
           -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
           20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
# Channelpedia 35°C QC 中位 ×3.445（Q10^1.35）折回 21.5°C（s）
_FAC = 2.5 ** 1.35
CP_DEACT = {-70: 37.7, -60: 43.4, -50: 56.7, -40: 80.0, -30: 85.1}          # ms @35°C
CP_ACT = {-20: 280.7, -10: 323.0, 0: 268.0, 10: 150.6, 20: 90.2,
          30: 64.1, 40: 57.5, 50: 71.5, 60: 105.6}                            # ms @35°C
TM_CP_FULL = dict(TM_ORIG)
TM_CP_FULL.update({v: ms / 1000 * _FAC for v, ms in CP_DEACT.items()})
TM_CP_FULL.update({v: ms / 1000 * _FAC for v, ms in CP_ACT.items()})
TM_CP_ACT = dict(TM_ORIG)
TM_CP_ACT.update({v: ms / 1000 * _FAC for v, ms in CP_ACT.items()})


class Tab:
    """log-linear interpolation table on sorted anchors (same as engine)."""
    def __init__(self, anchors):
        self.xs = np.array(sorted(anchors), float)
        self.ly = np.log(np.array([anchors[int(x)] for x in self.xs], float))


def build_tabs(T_lab, tm_anchors, q10=Q10_MAIN, E_rev=E_REV_CIPA):
    m_anchors = {-130: 1e-4, -120: 1e-4, -110: 1e-4, -100: 1e-4, -90: 1e-4,
                 -80: min(1.0, YMED[-70] * 0.5 / hB1[-80]),
                 -70: min(1.0, YMED[-70] / hB1[-70]),
                 -60: min(1.0, YMED[-60] / hB1[-60]),
                 -50: min(1.0, YMED[-50] / hB1[-50]),
                 -40: min(1.0, YMED[-50] * 3.0 / hB1[-40]),
                 -30: min(1.0, YMED[-50] * 8.0 / hB1[-30]),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    h_anchors = {-130: 1.0, -100: 1.0}
    h_anchors.update(hB1)
    h_anchors[40] = H40_POP
    h_anchors[50] = H50_POP
    h_anchors[60] = H50_POP
    fac = q10 ** ((T_lab - T_MAIN) / 10.0)
    tm = {k: v / fac for k, v in tm_anchors.items()}
    th = {k: v / fac for k, v in TH_ORIG.items()}
    return dict(m_ss=Tab(m_anchors), tau_m=Tab(tm), h_ss=Tab(h_anchors),
                tau_h=Tab(th), E_rev=E_rev)


def forward_dt(V, tabs, dt):
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hs = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-dt / tm); eh = np.exp(-dt / th)
    n = len(V); m = np.empty(n); h = np.empty(n)
    m0 = ms[0]; h0 = hs[0]
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hs[i] + (h0 - hs[i]) * eh[i]
        m[i] = m0; h[i] = h0
    return m, h


# ------------------------------------------------------------ 数据清单与分组（逐字同原卡）
def lab_of(name):
    if name.startswith("Hh11"):
        return "lab4"
    if name.startswith(("2020_", "207", "208")):
        return "lab5"
    if name[:3].isdigit() and "_" in name:
        return "lab3"
    return "manual"


files = sorted(f for f in os.listdir(CIPA_DIR) if f.endswith(".npz"))
pool_main, pool_manual = [], []
seen_hash = {}; dupes = []
for f in files:
    d = np.load(os.path.join(CIPA_DIR, f))
    i_arr = d["i"]; iok = i_arr[np.isfinite(i_arr)]
    hh = hashlib.md5(iok.tobytes()).hexdigest() if len(iok) else "empty"
    if hh in seen_hash:
        dupes.append((f, seen_hash[hh])); continue
    seen_hash[hh] = f
    (pool_manual if lab_of(f) == "manual" else pool_main).append(f)
print(f"[inventory] files={len(files)} dupes={len(dupes)} main={len(pool_main)} manual={len(pool_manual)}", flush=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"
if SMOKE:
    pool_main, pool_manual = pool_main[:1], []
    print("[smoke] 1 cell only", flush=True)

# ------------------------------------------------------------ 特征提取（逐字同原卡）
MS = 5

def sweep_onsets(v):
    return np.where((v[:-1] < 30.0) & (v[1:] >= 30.0))[0] + 1

def extract(v, i_sig, t0):
    def seg(ms0, ms1):
        a, b = t0 + int(ms0 * MS), min(t0 + int(ms1 * MS), len(i_sig))
        return a, b
    a1, b1 = seg(400, 495)
    if b1 <= a1 or np.isnan(i_sig[a1:b1]).any():
        return None
    F1 = float(np.mean(i_sig[a1:b1]))
    ra, rb = seg(500, 640)
    if rb <= ra:
        return None
    vv = v[ra:rb]; ii = i_sig[ra:rb]
    m2 = (vv >= -21.0) & (vv <= -19.0) & np.isfinite(ii)
    m3 = (vv >= -71.0) & (vv <= -69.0) & np.isfinite(ii)
    if m2.sum() < 2 or m3.sum() < 2:
        return None
    F2 = float(np.mean(ii[m2])); F3 = float(np.mean(ii[m3]))
    a4, b4 = seg(800, 850)
    F4 = float(np.mean(i_sig[a4:b4])) if b4 <= len(i_sig) and b4 > a4 and not np.isnan(i_sig[a4:b4]).any() else np.nan
    half = 0.5 * F1
    b5 = min(t0 + int(400 * MS), len(i_sig))
    seg_i = i_sig[t0:b5]
    above = np.where(np.isfinite(seg_i) & (seg_i >= half))[0]
    F5 = float(above[0] / MS) if len(above) else np.nan
    return dict(F1=F1, F2=F2, F3=F3, F4=F4, F5=F5)

def med_ratio(rows, key):
    rs = [r[key + "_r"] for r in rows if r[key + "_r"] is not None and np.isfinite(r[key + "_r"])]
    return float(np.median(rs)) if rs else None

def cell_verdict(rows, n_sweeps):
    if n_sweeps < MIN_SWEEPS:
        return "excluded_few_sweeps", {}
    if len(rows) < MIN_SWEEPS:
        return "excluded_few_valid", {}
    f1m = [r["F1_mea"] for r in rows if np.isfinite(r["F1_mea"])]
    if not f1m or abs(float(np.median(f1m))) < MIN_PLATEAU:
        return "excluded_low_signal", {}
    det = {}; passes = []
    for key in ("F1", "F2", "F3"):
        r = med_ratio(rows, key)
        det[key] = r
        if r is None:
            det[key + "_判"] = "不可判"; passes.append(False)
        else:
            ok = RATIO_LO <= r <= RATIO_HI
            det[key + "_判"] = "过" if ok else "不过"; passes.append(ok)
    det["F4_r"] = med_ratio(rows, "F4"); det["F5_r"] = med_ratio(rows, "F5")
    if len(rows) >= 2:
        det["drift_mea"] = rows[-1]["F1_mea"] / rows[0]["F1_mea"] if rows[0]["F1_mea"] else None
        det["drift_sim"] = rows[-1]["F1_sim"] / rows[0]["F1_sim"] if rows[0]["F1_sim"] else None
    return ("pass" if all(passes) else "fail"), det


# ------------------------------------------------------------ 主运行：三配置
CFGS = [("ORIG", TM_ORIG), ("CP_full", TM_CP_FULL), ("CP_act", TM_CP_ACT)]

result = {"说明": "CiPA跨实验室·Channelpedia τ_m换表测试；判线/特征窗/E_rev/Q10 全部冻结",
          "换表设计": {"CP_full_deact_ms35": CP_DEACT, "CP_act_ms35": CP_ACT,
                        "折回因子": _FAC, "基准温度": T_MAIN},
          "判线": {"特征窗": [RATIO_LO, RATIO_HI], "细胞过": "F1^F2^F3",
                   "数据集过线": "主池总过率>=2/3 且每实验室>=1/2"},
          "dupes": dupes, "cells": {}}

for pool_name, pool in (("main", pool_main), ("manual", pool_manual)):
    for fn in pool:
        lab = lab_of(fn)
        T = T_LAB.get(lab, 35.5)
        tabs_by_cfg = {name: build_tabs(T, tm) for name, tm in CFGS}
        d = np.load(os.path.join(CIPA_DIR, fn))
        v, i_meas = d["v"].astype(float), d["i"].astype(float)
        t0s = sweep_onsets(v)
        entry = {"lab": lab, "pool": pool_name, "n_sweeps": len(t0s), "cfgs": {}}
        for cname, _ in CFGS:
            tabs = tabs_by_cfg[cname]
            E = tabs["E_rev"]
            m, h = forward_dt(v, tabs, DT_NPZ)
            gden = m * h * (v - E)
            num = den = 0.0
            for t0 in t0s:
                a, b = t0, min(t0 + int(600 * MS), len(v))
                gg = gden[a:b]; ii = i_meas[a:b]
                ok = np.isfinite(ii)
                num += float(np.sum(gg[ok] * ii[ok]))
                den += float(np.sum(gg[ok] ** 2))
            G = num / den if den > 0 else np.nan
            i_sim = G * gden
            rows = []
            for k, t0 in enumerate(t0s):
                fm = extract(v, i_meas, t0); fs = extract(v, i_sim, t0)
                if fm is None or fs is None:
                    continue
                row = {"sweep": k + 1}
                for key in ("F1", "F2", "F3", "F4", "F5"):
                    mv, sv = fm[key], fs[key]
                    row[key + "_mea"] = mv; row[key + "_sim"] = sv
                    row[key + "_r"] = (sv / mv) if (np.isfinite(mv) and abs(mv) >= MIN_MEAS) else None
                rows.append(row)
            verdict, det = cell_verdict(rows, len(t0s))
            entry["cfgs"][cname] = dict(G=G, n_valid=len(rows), 判定=verdict, **det)
        result["cells"][fn] = entry
        print(f"  {fn} [{lab}] " + "  ".join(
            f"{c}:{entry['cfgs'][c]['判定']}" for c, _ in CFGS), flush=True)

# ------------------------------------------------------------ 汇总
def summarize(pool_name, cname):
    sel = {k: v for k, v in result["cells"].items() if v["pool"] == pool_name}
    judged = [v for v in sel.values() if v["cfgs"][cname]["判定"] in ("pass", "fail")]
    npass = sum(1 for v in judged if v["cfgs"][cname]["判定"] == "pass")
    per_lab = {}
    for lab in ("lab3", "lab4", "lab5"):
        lj = [v for v in sel.values() if v["lab"] == lab and v["cfgs"][cname]["判定"] in ("pass", "fail")]
        lp = sum(1 for v in lj if v["cfgs"][cname]["判定"] == "pass")
        per_lab[lab] = {"n": len(lj), "pass": lp, "rate": (lp / len(lj)) if lj else None}
    excl = {}
    for v in sel.values():
        vd = v["cfgs"][cname]["判定"]
        if vd not in ("pass", "fail"):
            excl[vd] = excl.get(vd, 0) + 1
    return {"n_judged": len(judged), "n_pass": npass,
            "rate": (npass / len(judged)) if judged else None,
            "per_lab": per_lab, "excluded": excl}

summ = {}
for cname, _ in CFGS:
    summ[cname] = {"main": summarize("main", cname), "manual": summarize("manual", cname)}
result["summary"] = summ

verdicts = {}
for cname, _ in CFGS:
    main = summ[cname]["main"]
    line1 = (main["rate"] is not None and main["rate"] >= 2.0 / 3.0)
    line2 = all((l["rate"] is not None and l["rate"] >= 0.5) for l in main["per_lab"].values() if l["n"] > 0)
    if line1 and line2:
        verdicts[cname] = "过线"
    elif main["rate"] is not None and main["rate"] >= 0.5:
        verdicts[cname] = "部分过"
    else:
        verdicts[cname] = "不过"
result["总判"] = verdicts

print("=" * 76, flush=True)
for cname, _ in CFGS:
    main = summ[cname]["main"]
    print(f"[{cname}] main pool {main['n_pass']}/{main['n_judged']} = "
          f"{main['rate'] if main['rate'] is not None else float('nan'):.3f}  判:{verdicts[cname]}", flush=True)
    for lab, l in main["per_lab"].items():
        if l["n"]:
            print(f"    {lab}: {l['pass']}/{l['n']} = {l['rate']:.3f}", flush=True)
    print(f"    排除登记: {main['excluded']}", flush=True)
    man = summ[cname]["manual"]
    print(f"    manual 池: {man['n_pass']}/{man['n_judged']}", flush=True)

# 逐细胞 ORIG vs CP_full 翻转清单
flips = []
for fn, v in result["cells"].items():
    a, b = v["cfgs"]["ORIG"]["判定"], v["cfgs"]["CP_full"]["判定"]
    if a != b:
        flips.append({"cell": fn, "lab": v["lab"], "ORIG": a, "CP_full": b,
                      "F1_cp": v["cfgs"]["CP_full"].get("F1"),
                      "F2_cp": v["cfgs"]["CP_full"].get("F2"),
                      "F3_cp": v["cfgs"]["CP_full"].get("F3")})
result["翻转清单"] = flips
print(f"[翻转] ORIG vs CP_full 判定不同: {len(flips)} 个", flush=True)
for f in flips:
    print("   ", json.dumps(f, ensure_ascii=False), flush=True)

sfx = "_冒烟" if SMOKE else ""
fjson = os.path.join(HERE, f"2026-11-09_CiPA跨实验室_换表测试{sfx}_结果.json")
json.dump(result, open(fjson, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  saved: {fjson}", flush=True)
