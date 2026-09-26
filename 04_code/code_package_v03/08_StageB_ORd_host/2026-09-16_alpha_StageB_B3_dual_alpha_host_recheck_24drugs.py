# -*- coding: utf-8 -*-
"""
2026-09-16 · alpha-model Stage B · B3 dual-alpha host recheck arm (24 drugs)
====================================================================================
Preregistration: 结果\预注册_α模型_StageB_ORd宿主对拍_2026-09-14.md section 11.9 (v7) + section 11.10 (v7.1),
all criteria pinned before the run. This script = B2 v6.1 drug module (verbatim) + ORd assembly v1.1 host (Kr+Ks dual swap-in).

Host (pinned in v7 item 1.1, not recalibrated):
  IKr = alpha four-table product gate (G_Kr frozen at the assembly calibrated value); IKs = alpha directly-extracted table (G_Ks likewise frozen);
  INa/ICaL/INaL/Ito/IK1 = official. Temperature bridges Kr /4.138, Ks /3.952 (Q10=2.5).
Drug module (v7 item 1.4 = B2 verbatim):
  hERG dynamic binding dD/dt = k_on*C*(1-h)*(1-D) - k_off*D (card-3 medians; k_off=k_on x IC50);
  other channels static Hill: LateNa/PeakNa/IK1/CaL/Ito via the fc interface (=1/mult),
  **the IKs multiplier acts on the alpha-IKs conductance** (G_Ks_eff = G_Ks x mult_IKs; convention difference registered, v7 items 1.4/3.4).

Criteria (v7 item 2 + v7.1, pinned before the run):
  C1 control APD90 within 2% of the B0 formal value 252.118 ms (rerun confirmation, expected pass; fail -> host rebuild investigation);
  C2 dofetilide 100xCmax: APD90 prolongation >40% and qNet decrease; C3 nifedipine 100xCmax: APD90 shortening;
  R1  rho(B3,B2) >= 0.95 (qNet_ratio and APD90_ratio dual conventions, per drug over 24 drugs);
  R1b per-drug |delta qNet_ratio| median <= 0.20 (numerical consistency, v7.1 D1);
  R2  |delta rho(V1)| <= 0.15 (v7.1 D2 relaxation) and B3 V2 AUC >= 0.80; increment criterion B3 vs B1 not inferior;
  R3  Kendall discordant pairs <= 2 (operationalized in v7.1);
  registry: strong-IKs-suppression subset (IKsIC50>0 and <10xCmax) per-drug check (v7.1 D4, no criterion);
        ex-metoprolol V1 (same convention as verdict card section 7 A3, registry);
        last-beat six-current integrals (piggyback refill of assembly verdict card second-review B1/B4: official reference arm 2 beats vs main-arm last beat).
56 states: 0-48 ORd, 49 qNet, 50 m_Kr, 51 h_Kr, 52 D_Kr, 53/54 unused, 55 a_dyn_Ks.

Environment variables: SMOKE=1 smoke; NWORKERS (default 1); NBEATS_CTRL/NBEATS_DRUG.
Discipline: this side runs ast.parse + SMOKE=1 smoke; the formal run is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-16_α模型_StageB_B3_双α化宿主复验_24药.py' --wdir
Output: _结果.json/.csv/.png next to this script (smoke carries the _冒烟 suffix) + StageB_B3稳态_CL1000.json.
"""

import os
import sys
import csv
import json
import math
import time
import importlib.util

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(BASE, "CiPA官方锚")
CIPA_CSV = os.path.join(ANCHOR, "newCiPA.csv")
KA3_JSON = os.path.join(BASE, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")
B0_SS = os.path.join(BASE, "StageB_B0稳态_CL1000.json")
B1_JSON = os.path.join(BASE, "2026-09-14_α模型_StageB_B1_官方静态臂_24药_结果.json")
B2_JSON = os.path.join(BASE, "2026-09-14_α模型_StageB_B2_α替换_24药_tauA1_Q2.5_ka3_结果.json")
ASM_JSON = os.path.join(BASE, "2026-09-15_ORd三通道α化组装_结果.json")
FEDIDA_CSV = os.path.join(BASE, "2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv")
B3_SS = os.path.join(BASE, "StageB_B3稳态_CL1000.json")

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_CTRL = int(os.environ.get("SMOKE_BEATS", "10")) if SMOKE else int(os.environ.get("NBEATS_CTRL", "500"))
N_DRUG = int(os.environ.get("SMOKE_BEATS", "20")) if SMOKE else int(os.environ.get("NBEATS_DRUG", "500"))
NWORKERS = int(os.environ.get("NWORKERS", "15"))
CL = 1000.0
DT_REC = 0.1
B0_FORMAL_APD90 = 252.1181387725961   # B0 formal-run last beat (on record 2026-09-14)

# temperature bridges (v7 item 1.1: consistent with the assembly)
QFAC_KR = 2.5 ** ((37.0 - 21.5) / 10.0)     # ÷4.138
QFAC_KS = 2.5 ** 1.5                        # ÷3.952

RT_F = 8314.0 * 310.0 / 96485.0
NAO = 140.0
PKNA = 0.01833
BETA = 0.30

# 24 drugs (card-4 section 3, verbatim identical to B1/B2)
DRUGS_HIGH = ["azimilide", "bepridil", "disopyramide", "dofetilide", "ibutilide", "sotalol", "vandetanib"]
DRUGS_MID = ["astemizole", "chlorpromazine", "cisapride", "clarithromycin", "clozapine",
             "domperidone", "droperidol", "ondansetron", "pimozide", "risperidone", "terfenadine"]
DRUGS_LOW = ["diltiazem", "metoprolol", "mexiletine", "ranolazine", "tamoxifen", "verapamil"]
DRUGS_24 = DRUGS_HIGH + DRUGS_MID + DRUGS_LOW
SMOKE_DRUGS = ["dofetilide", "cisapride", "verapamil"]

# fc channels (hERG goes dynamic binding, IKs goes alpha conductance; neither listed here): (name, IC50 column, h column, pars index)
CHANNELS_FC = [("LateNa", "Late_sodiumIC50", "Late_sodiumh", 2),
               ("PeakNa", "Peak_sodiumIC50", "Peak_sodiumh", 3),
               ("IK1", "IK1IC50", "IK1h", 5),
               ("CaL", "CaLIC50", "CaLh", 6),
               ("Ito", "ItoIC50", "Itoh", 7)]
IKS_CSV = ("IKsIC50", "IKsh")


# ----------------------------------------------------------------------------
def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cipa(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["drug"].strip()] = row
    return rows


class Lut:
    def __init__(self, xs, ys):
        o = np.argsort(xs)
        self.xs = np.asarray(xs, dtype=float)[o]
        self.ys = np.asarray(ys, dtype=float)[o]
    def __call__(self, v):
        return float(np.interp(v, self.xs, self.ys))


# ----------------------------------------------------------------------------
# card-3 dynamic-binding parameters (same logic as B2, verbatim)
# ----------------------------------------------------------------------------
def ka3_params():
    d = json.load(open(KA3_JSON, encoding="utf-8"))
    ds = d["datasets"]

    def kon_of(e):
        from collections import defaultdict
        byc = defaultdict(list)
        for u in e["units"]:
            if u["k_obs"] is not None and u["conc_nM"] is not None:
                byc[u["conc_nM"]].append(u["k_obs"])
        if len(byc) < 2:
            return None
        cs = np.array(sorted(byc))
        ks = np.array([np.median(byc[c]) for c in cs])
        A = np.vstack([cs, np.ones_like(cs)]).T
        return float(np.linalg.lstsq(A, ks, rcond=None)[0][0])

    per_drug, ic50_self, lib = {}, {}, []
    for k, e in ds.items():
        drug = k.split("|")[1]
        if e["B"]["判"] == "浓度依赖可证":
            s = kon_of(e)
            if s:
                per_drug.setdefault(drug, []).append(s)
                lib.append(s)
        v = e["L1"].get("ic50_self")
        if v:
            ic50_self.setdefault(drug, []).append(v)
    lib_med = float(np.median(lib))
    out = {}
    for drug in DRUGS_24 + ["nifedipine"]:
        kon = float(np.median(per_drug[drug])) if drug in per_drug else lib_med
        out[drug] = {"k_on": kon, "k_on_default": drug not in per_drug,
                     "ic50_self": (float(np.median(ic50_self[drug])) if drug in ic50_self else None)}
    return out, lib_med


# ----------------------------------------------------------------------------
# alpha tables (same construction as the assembly: hERG four tables TAU_ARM=1; IKs Fedida 25 points)
# ----------------------------------------------------------------------------
def build_tabs_kr(eng):
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)

    def scaled(tab):
        out = {}
        for x, ly in zip(tab.xs, tab.ly):
            out[float(x)] = float(np.exp(ly)) * 1000.0 / QFAC_KR
        return eng.Tab(out, log_y=True)

    return {"m_ss": tabs["m_ss"], "h_ss": tabs["h_ss"],
            "tau_m": scaled(tabs["tau_m"]), "tau_h": scaled(tabs["tau_h"])}


def build_tabs_ks():
    xs, ys = [], []
    with open(FEDIDA_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["V_mV"]))
            ys.append(float(row["tau_s"]) * 1000.0 / QFAC_KS)
    return {"a_ss": lambda v: 1.0 / (1.0 + math.exp(-(v - 25.05) / 22.6)),
            "tau_act": Lut(xs, ys)}


# ----------------------------------------------------------------------------
# B3 rhs: B0 port + Kr/Ks dual swap-in + D state (52); official hERG/IKs gates frozen
# ----------------------------------------------------------------------------
def build_b3(b0, P_d, tabs, G, kon_per_nM_s, koff_per_s, C_nM):
    rhs0 = b0.build_model(P_d)          # P_d[1]=P_d[4]=inf set by the caller
    ko = P_d[b0.PARS_NAMES.index("ko")]
    rad, L = 0.0011, 0.01
    vcell = 1000 * 3.14 * rad * rad * L
    Ageo = 2 * 3.14 * rad * rad + 2 * 3.14 * rad * L
    Acap_Fvmyo = 2 * Ageo / (96485.0 * 0.68 * vcell)
    tkr, tks = tabs["Kr"], tabs["Ks"]
    kon_ms = kon_per_nM_s * C_nM / 1000.0
    koff_ms = koff_per_s / 1000.0

    def rhs(t, y):
        dy, cur = rhs0(t, y)
        v = y[0]
        dy_a = list(dy) + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(39, 49):           # official hERG Markov six states + bound states frozen
            dy_a[i] = 0.0
        dy_a[33] = 0.0                    # official IKs gate (xs1/xs2) frozen
        dy_a[34] = 0.0
        m, h, D = y[50], y[51], y[52]
        ad = y[55]
        dy_a[50] = (tkr["m_ss"](v) - m) / tkr["tau_m"](v)
        dy_a[51] = (tkr["h_ss"](v) - h) / tkr["tau_h"](v)
        dy_a[52] = kon_ms * (1.0 - h) * (1.0 - D) - koff_ms * D
        EK = RT_F * math.log(ko / y[3])
        IKr_a = G["Kr"] * m * h * (1.0 - D) * (v - EK)
        ass = tks["a_ss"](v)
        dy_a[55] = ((1.0 - BETA) * ass - ad) / tks["tau_act"](v)
        EKs = RT_F * math.log((ko + PKNA * NAO) / (y[3] + PKNA * y[1]))
        IKs_a = G["Ks"] * (BETA * ass + ad) * (v - EKs)
        dy_a[0] -= (IKr_a + IKs_a)
        dy_a[3] -= (IKr_a + IKs_a) * Acap_Fvmyo
        dy_a[49] += (IKr_a + IKs_a)
        return dy_a, (cur[0], cur[1], cur[2], cur[3], IKr_a, IKs_a, cur[6])

    return rhs


def run_beats(b0, rhs, y0, nbeats, label="", keep_currents=False):
    from scipy.integrate import solve_ivp
    f = lambda t, y: rhs(t, y)[0]
    t_rec = np.arange(0.0, CL + 1e-9, DT_REC)
    y = np.array(y0, dtype=float)
    apd = np.full(nbeats, np.nan)
    apa = np.full(nbeats, np.nan)
    qnet = np.full(nbeats, np.nan)
    out = {"t": None, "v": None, "cur": None, "y_end": None}
    t0 = time.time()
    for b in range(nbeats):
        q0 = y[49]
        sol = solve_ivp(f, (0.0, CL), y, method="LSODA", t_eval=t_rec,
                        rtol=1e-7, atol=1e-9)
        if not sol.success:
            print(f"    {label} beat {b + 1}: LSODA failed {sol.message}", flush=True)
            break
        v = sol.y[0]
        y = sol.y[:, -1].copy()
        apd[b], apa[b], _, _ = b0.apd90(t_rec, v)
        qnet[b] = (y[49] - q0) * 1e-3
        if b == nbeats - 1 and keep_currents:
            out["t"] = t_rec
            out["v"] = v.copy()
            cur = np.empty((7, len(t_rec)))
            for k in range(len(t_rec)):
                _, c7 = rhs(float(t_rec[k]), sol.y[:, k].tolist())
                cur[:, k] = c7
            out["cur"] = cur
        if (b + 1) % 50 == 0 or b == 0:
            print(f"    {label} beat {b + 1}/{nbeats}  APD90={apd[b]:7.2f}ms"
                  f"  qNet={qnet[b]:.4f}  took {time.time() - t0:.0f}s", flush=True)
    out["y_end"] = y
    return apd, apa, qnet, out


def last_valid(x):
    ok = x[~np.isnan(x)]
    return float(ok[-1]) if len(ok) else float("nan")


def steady_flag(apd):
    ok = apd[~np.isnan(apd)]
    if len(ok) < 10:
        return float("nan"), False
    r = float(np.max(ok[-10:]) - np.min(ok[-10:]))
    return r, bool(r < 0.5)


def auc_pos(pos, neg):
    n_g, n_t = 0.0, 0
    for a in pos:
        for b in neg:
            n_g += 1.0 if a > b else (0.5 if a == b else 0.0)
            n_t += 1
    return n_g / n_t if n_t else float("nan")


def judge(results):
    from scipy.stats import spearmanr
    risk = np.array([r["CiPA"] for r in results], dtype=float)
    qr = np.array([r["qNet_ratio"] for r in results], dtype=float)
    ar = np.array([r["APD90_ratio"] for r in results], dtype=float)
    okq, oka = ~np.isnan(qr), ~np.isnan(ar)
    rho_q, p_q = spearmanr(risk[okq], qr[okq])
    rho_a, p_a = spearmanr(risk[oka], ar[oka])
    sup = 1.0 - qr
    auc_q = auc_pos(sup[(risk == 2) & okq], sup[(risk == 0) & okq])
    return {"V1_rho_qNet": float(rho_q), "V1_p": float(p_q),
            "V1_pass": bool(rho_q <= -0.60 and p_q < 0.01),
            "V2_AUC_qNetSupp_hi_lo": float(auc_q), "V2_pass": bool(auc_q >= 0.80),
            "reg_APD90_rho": float(rho_a),
            "reg_n_EAD_or_fail": int(sum(r["EAD_or_fail"] for r in results)),
            "reg_n_steady_fail": int(sum(not r["steady_ok"] for r in results))}


def kendall_discordant(x, y):
    """Discordant-pair count (v7.1 R3 operationalization)."""
    n, d = len(x), 0
    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) < 0:
                d += 1
    return d


# ----------------------------------------------------------------------------
# single-drug worker (NWORKERS parallelism; the IKs multiplier acts on the alpha-IKs conductance, v7 item 1.4)
# ----------------------------------------------------------------------------
def one_drug(task):
    (drug, C_scale, G_Kr, G_Ks, ss_file) = task
    b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")
    rows = load_cipa(CIPA_CSV)
    ka3, _ = ka3_params()
    row = rows[drug]
    C = float(row["therapeutic"]) * C_scale
    P_ctrl = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    P_d = list(P_ctrl)
    P_d[1] = float("inf")                 # GKrfc -> official GKr = 0 (swap-in)
    P_d[4] = float("inf")                 # GKsfc -> official GKs = 0 (swap-in)
    mults = {}
    for nm, cIC, ch, idx in CHANNELS_FC:
        IC50, h = float(row[cIC]), float(row[ch])
        if IC50 <= 0.0 or h <= 0.0:
            mults[nm] = 1.0
            continue
        mult = 1.0 / (1.0 + (C / IC50) ** h)
        mults[nm] = mult
        P_d[idx] = 1.0 / mult
    IC50_ks, h_ks = float(row[IKS_CSV[0]]), float(row[IKS_CSV[1]])
    mult_ks = 1.0 / (1.0 + (C / IC50_ks) ** h_ks) if (IC50_ks > 0 and h_ks > 0) else 1.0
    ic50_dyn = ka3[drug]["ic50_self"]
    if ic50_dyn is None or ic50_dyn <= 0:
        IC50_h, _h_h = float(row["hERGIC50"]), float(row["hERGh"])
        ic50_dyn = IC50_h if IC50_h > 0 else float("inf")
    kon = ka3[drug]["k_on"]
    koff = kon * ic50_dyn

    eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")
    tabs = {"Kr": build_tabs_kr(eng), "Ks": build_tabs_ks()}
    with open(ss_file, encoding="utf-8") as fh:
        y0 = json.load(fh)["states56"]

    G = {"Kr": G_Kr, "Ks": G_Ks * mult_ks}
    rhs = build_b3(b0, P_d, tabs, G, kon, koff, C)
    apd, apa, qnet, _ = run_beats(b0, rhs, y0, N_DRUG, label=f"{drug}")
    nd = int(np.sum(~np.isnan(apd)))
    APD_d, qNet_d = last_valid(apd), last_valid(qnet)
    rng, okd = steady_flag(apd)
    return {"drug": drug, "CiPA": int(float(row["CiPA"])), "C_nM": C,
            "mult_IKs": mult_ks, "mults_fc": mults,
            "k_on": kon, "k_off": koff, "k_on_default": ka3[drug]["k_on_default"],
            "ic50_used": ic50_dyn, "APD90_ms": APD_d, "qNet_uC_uF": qNet_d,
            "steady_range_ms": rng, "steady_ok": okd,
            "EAD_or_fail": bool(nd < N_DRUG or math.isnan(APD_d)), "n_beats_ok": nd}


def main():
    t_start = time.time()
    tag = "_冒烟" if SMOKE else "_结果"
    print("=" * 76, flush=True)
    print(" alpha-model Stage B · B3 dual-alpha host recheck arm (alpha-Kr + alpha-Ks host, 24-drug dynamic binding)", flush=True)
    print(f" mode: {'smoke SMOKE' if SMOKE else 'formal'}  control {N_CTRL} beats + drug {N_DRUG} beats"
          f"  NWORKERS={NWORKERS}", flush=True)
    print("=" * 76, flush=True)

    for f in (B0_SS, B2_JSON, ASM_JSON):
        if not os.path.exists(f):
            raise RuntimeError(f"missing prerequisite asset: {f}")
    b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")
    P_ctrl = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    P_ctrl[1] = float("inf")
    P_ctrl[4] = float("inf")
    with open(B0_SS, encoding="utf-8") as fh:
        y_ss = json.load(fh)["states50"]

    asm = json.load(open(ASM_JSON, encoding="utf-8"))
    G_Kr = float(asm["calibration"]["G_Kr"])
    G_Ks = float(asm["calibration"]["G_Ks"])
    assert abs(G_Kr / 0.32266 - 1) < 0.02 and abs(G_Ks / 0.004014 - 1) < 0.02, \
        f"assembly calibrated values drifted G_Kr={G_Kr} G_Ks={G_Ks} (expected anchors 0.32266/0.004014, v7 item 1.2)"
    print(f"[host] assembly calibration frozen values G_Kr={G_Kr:.5f}  G_Ks={G_Ks:.6f} (v7 item 1.1, not recalibrated)", flush=True)

    eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")
    tabs = {"Kr": build_tabs_kr(eng), "Ks": build_tabs_ks()}
    v0 = y_ss[0]
    y0 = list(y_ss) + [tabs["Kr"]["m_ss"](v0), tabs["Kr"]["h_ss"](v0), 0.0,
                       0.0, 0.0, (1.0 - BETA) * tabs["Ks"]["a_ss"](v0)]
    print(f"[tables] hERG four tables (16713003, /{QFAC_KR:.3f}) + IKs Fedida 25 points (/{QFAC_KS:.3f}) loaded; "
          f"init m={y0[50]:.4f} h={y0[51]:.4f} a_dyn={y0[55]:.5f} @v={v0:.2f}mV", flush=True)

    # ---------- control: steady-state rebuild + C1 + save B3 steady state ----------
    print(f"\n[control] steady-state rebuild {N_CTRL} beats (from B0 steady state + alpha-state init) ...", flush=True)
    G0 = {"Kr": G_Kr, "Ks": G_Ks}
    rhs_c = build_b3(b0, P_ctrl, tabs, G0, 0.0, 0.0, 0.0)
    apd_c, apa_c, qn_c, out_c = run_beats(b0, rhs_c, y0, N_CTRL, label="control",
                                          keep_currents=True)
    APD_ctrl, qNet_ctrl = last_valid(apd_c), last_valid(qn_c)
    r_c, ok_c = steady_flag(apd_c)
    c1 = abs(APD_ctrl - B0_FORMAL_APD90) / B0_FORMAL_APD90
    C1 = bool(c1 <= 0.02)
    print(f"  control last-beat APD90={APD_ctrl:.3f}ms qNet={qNet_ctrl:.4f}  steady range={r_c:.3f}ms"
          f"  C1 |diff|={c1 * 100:.2f}% -> {'pass (rerun confirmed)' if C1 else 'fail - triggers host rebuild investigation; no drug verdicts issued'}", flush=True)

    ss_file = B3_SS if not SMOKE else B3_SS.replace(".json", "_冒烟.json")
    with open(ss_file, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"from": "B0 steady state + alpha-state init", "beats": N_CTRL,
                            "G_Kr": G_Kr, "G_Ks": G_Ks, "APD90": APD_ctrl,
                            "steady_range_ms": r_c, "C1_rel": c1, "smoke": SMOKE},
                   "states56": [float(x) for x in out_c["y_end"]]}, fh, ensure_ascii=False)
    print(f"  B3 steady state saved: {ss_file}", flush=True)

    # ---------- last-beat six-current integrals (piggyback: refill assembly verdict card second-review B1/B4) ----------
    print("\n[integral registry] official reference arm 2 beats (from B0 steady state) vs B3 main-arm last beat ...", flush=True)
    P_ref = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    rhs_r0 = b0.build_model(P_ref)
    rhs_r = lambda t, y: rhs_r0(t, y)
    apd_r, apa_r, qn_r, out_r = run_beats(b0, rhs_r, y_ss, 2, label="ref-arm", keep_currents=True)
    integ = {}
    for i, nm in enumerate(("INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1")):
        q_ref = float(np.trapezoid(out_r["cur"][i], out_r["t"]))
        q_b3 = float(np.trapezoid(out_c["cur"][i], out_c["t"]))
        integ[nm] = {"q_ref_pAms": q_ref, "q_B3_pAms": q_b3,
                     "ratio": (q_b3 / q_ref if abs(q_ref) > 1e-12 else None)}
    print(f"  IKr charge: ref {integ['IKr']['q_ref_pAms']:.2f} vs B3 {integ['IKr']['q_B3_pAms']:.2f}"
          f" pA/pF*ms (ratio {integ['IKr']['ratio']:.3f})", flush=True)
    print(f"  IKs charge: ref {integ['IKs']['q_ref_pAms']:.2f} vs B3 {integ['IKs']['q_B3_pAms']:.2f}"
          f" pA/pF*ms (ratio {integ['IKs']['ratio']:.3f})", flush=True)

    # ---------- C2 / C3 self-checks ----------
    rows = load_cipa(CIPA_CSV)
    checks = {}
    for tagc, drugc in [("C2", "dofetilide"), ("C3", "nifedipine")]:
        rec = one_drug((drugc, 100.0, G_Kr, G_Ks, ss_file))
        r_apd = rec["APD90_ms"] / APD_ctrl
        r_q = rec["qNet_uC_uF"] / qNet_ctrl
        ok = bool(r_apd > 1.40 and r_q < 1.0) if tagc == "C2" else bool(r_apd < 1.0)
        checks[tagc] = {"drug": drugc, "C": "100xCmax", "APD90_ratio": r_apd,
                        "qNet_ratio": r_q, "pass": ok}
        print(f"  [{tagc}] {drugc} 100×Cmax: APD90 {r_apd:.3f}×  qNet {r_q:.3f}×  "
              f"-> {'pass' if ok else 'fail'}", flush=True)

    # ---------- 24 drugs ----------
    drug_list = SMOKE_DRUGS if SMOKE else DRUGS_24
    tasks = [(d, 1.0, G_Kr, G_Ks, ss_file) for d in drug_list]
    print(f"\n[{len(tasks)} drugs] {N_DRUG} beats each, NWORKERS={NWORKERS}", flush=True)
    results = []
    if NWORKERS > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            for rec in ex.map(one_drug, tasks):
                results.append(rec)
                print(f"  [done] {rec['drug']} APD90={rec['APD90_ms']:.2f}"
                      f"  qNet={rec['qNet_uC_uF']:.4f}", flush=True)
    else:
        for i, tk in enumerate(tasks):
            t0 = time.time()
            rec = one_drug(tk)
            results.append(rec)
            print(f"  [{i + 1}/{len(tasks)}] {rec['drug']:15s} CiPA={rec['CiPA']}"
                  f"  APD90={rec['APD90_ms']:8.2f}  qNet={rec['qNet_uC_uF']:.4f}"
                  f"  steady {rec['steady_range_ms']:.2f}{'v' if rec['steady_ok'] else 'x'}"
                  f"  {time.time() - t0:.0f}s", flush=True)

    for rec in results:
        rec["APD90_ratio"] = rec["APD90_ms"] / APD_ctrl if not math.isnan(rec["APD90_ms"]) else float("nan")
        rec["qNet_ratio"] = rec["qNet_uC_uF"] / qNet_ctrl if not math.isnan(rec["qNet_uC_uF"]) else float("nan")

    # ---------- B2 comparison (R1/R1b/R2/R3) ----------
    from scipy.stats import spearmanr
    b2 = json.load(open(B2_JSON, encoding="utf-8"))
    b2dyn = {r["drug"]: r for r in b2["results"] if r["mode"] == "dyn"}
    vd_b3 = judge(results) if len(results) >= 4 else None
    R = {}
    common = [r for r in results if r["drug"] in b2dyn and not math.isnan(r["qNet_ratio"])]
    if len(common) >= 4:
        q3 = np.array([r["qNet_ratio"] for r in common])
        q2 = np.array([b2dyn[r["drug"]]["qNet_ratio"] for r in common])
        a3 = np.array([r["APD90_ratio"] for r in common])
        a2 = np.array([b2dyn[r["drug"]]["APD90_ratio"] for r in common])
        rho_q, _ = spearmanr(q3, q2)
        rho_a, _ = spearmanr(a3, a2)
        dev_med = float(np.median(np.abs(q3 - q2)))
        ndisc = kendall_discordant(list(q3), list(q2))
        R = {"n_common": len(common),
             "R1_rho_qNet": float(rho_q), "R1_rho_APD90": float(rho_a),
             "R1_pass": bool(rho_q >= 0.95 and rho_a >= 0.95),
             "R1b_dev_median": dev_med, "R1b_pass": bool(dev_med <= 0.20),
             "R3_kendall_discordant": int(ndisc), "R3_pass": bool(ndisc <= 2)}
        if vd_b3:
            b2_v1 = float(b2["verdicts"]["dyn"]["V1_rho_qNet"])
            R["R2_drho"] = abs(vd_b3["V1_rho_qNet"] - b2_v1)
            R["R2_pass"] = bool(R["R2_drho"] <= 0.15 and vd_b3["V2_pass"])
    lab = "criterion-path rehearsal (smoke, not a verdict)" if SMOKE else "criteria"
    print("\n" + "-" * 76, flush=True)
    if vd_b3:
        print(f" [B3·{lab}] V1 ρ={vd_b3['V1_rho_qNet']:.3f}(p={vd_b3['V1_p']:.2e})"
              f"  V2 AUC={vd_b3['V2_AUC_qNetSupp_hi_lo']:.3f}", flush=True)
    if R:
        print(f" [R1] ρ(B3,B2): qNet {R['R1_rho_qNet']:.3f} / APD90 {R['R1_rho_APD90']:.3f}"
              f"(>=0.95) -> {'pass' if R['R1_pass'] else 'fail'}", flush=True)
        print(f" [R1b] per-drug |delta qNet_ratio| median {R['R1b_dev_median']:.3f} (<=0.20) -> "
              f"{'pass' if R['R1b_pass'] else 'fail'}", flush=True)
        print(f" [R3] Kendall discordant pairs {R['R3_kendall_discordant']} (<=2) -> "
              f"{'pass' if R['R3_pass'] else 'fail'}", flush=True)
        if "R2_drho" in R:
            print(f" [R2] |delta rho(V1)|={R['R2_drho']:.3f} (<=0.15) and V2>=0.80 -> "
                  f"{'pass' if R['R2_pass'] else 'fail'}", flush=True)

    # ---------- increment criterion B3 vs B1 ----------
    incr = None
    if os.path.exists(B1_JSON) and vd_b3:
        b1v = json.load(open(B1_JSON, encoding="utf-8"))["verdict"]
        incr = {"B1_V1_rho": b1v["V1_rho_qNet"], "B1_V2_AUC": b1v["V2_AUC_qNetSupp_hi_lo"],
                "B3_V1_rho": vd_b3["V1_rho_qNet"], "B3_V2_AUC": vd_b3["V2_AUC_qNetSupp_hi_lo"]}
        incr["not_inferior"] = bool(vd_b3["V1_rho_qNet"] <= b1v["V1_rho_qNet"]
                                    and vd_b3["V2_AUC_qNetSupp_hi_lo"] >= b1v["V2_AUC_qNetSupp_hi_lo"])
        print(f" [increment] B3 vs B1: rho {incr['B3_V1_rho']:.3f} vs {incr['B1_V1_rho']:.3f}; "
              f"AUC {incr['B3_V2_AUC']:.3f} vs {incr['B1_V2_AUC']:.3f} -> "
              f"{'not inferior' if incr['not_inferior'] else 'inferior'}", flush=True)

    # ---------- registry: strong-IKs-suppression subset (v7.1 D4) + ex-metoprolol ----------
    iks_strong = []
    for rec in results:
        row = rows[rec["drug"]]
        IC50_ks = float(row[IKS_CSV[0]])
        C1x = float(row["therapeutic"])
        if IC50_ks > 0 and IC50_ks < 10.0 * C1x:
            iks_strong.append({"drug": rec["drug"], "IKsIC50_nM": IC50_ks, "Cmax_nM": C1x,
                               "mult_IKs": rec["mult_IKs"], "qNet_ratio": rec["qNet_ratio"],
                               "APD90_ratio": rec["APD90_ratio"]})
    print(f" [registry] strong-IKs-suppression subset (IC50<10xCmax): "
          f"{[d['drug'] for d in iks_strong] if iks_strong else 'none'}", flush=True)
    exmet = None
    if len(results) >= 5:
        sub = [r for r in results if r["drug"] != "metoprolol"]
        exmet = {"n": len(sub), "V1_rho": judge(sub)["V1_rho_qNet"]}
        print(f" [registry] ex-metoprolol V1 rho={exmet['V1_rho']:.3f} (n={exmet['n']}, post-run convention registry)", flush=True)

    # ---------- save ----------
    fjson = os.path.join(BASE, f"2026-09-16_α模型_StageB_B3_双α化宿主复验_24药{tag}.json")
    out = {"meta": {"smoke": SMOKE, "prereg": "v7 §11.9 + v7.1 §11.10",
                    "host": {"G_Kr": G_Kr, "G_Ks": G_Ks, "source": "ORd三通道α化组装_结果.json"},
                    "control_APD90_ms": APD_ctrl, "control_qNet_uC_uF": qNet_ctrl,
                    "C1_rel_diff": c1, "C1_pass": C1, "steady_range_ms": r_c,
                    "drug_list": drug_list,
                    "k_on_default_drugs": [d for d in drug_list
                                           if ka3_params()[0][d]["k_on_default"]],
                    "runtime_s": time.time() - t_start},
           "checks": checks, "charge_integration": integ,
           "results": results, "verdict_B3": vd_b3, "R_lines": R,
           "incremental_vs_B1": incr, "IKs_strong_subset": iks_strong,
           "ex_metoprolol": exmet}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    fcsv = os.path.join(BASE, f"2026-09-16_α模型_StageB_B3_双α化宿主复验_24药{tag}.csv")
    with open(fcsv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drug", "CiPA", "C_nM", "mult_IKs", "k_on", "k_off", "k_on_default",
                    "ic50_used", "B3_APD90_ms", "B3_APD90_ratio", "B3_qNet", "B3_qNet_ratio",
                    "B2dyn_qNet_ratio", "B2dyn_APD90_ratio", "delta_qNet_ratio",
                    "steady_range_ms", "steady_ok", "EAD_or_fail"])
        for r in sorted(results, key=lambda x: x["drug"]):
            b2r = b2dyn.get(r["drug"], {})
            w.writerow([r["drug"], r["CiPA"], r["C_nM"], r["mult_IKs"], r["k_on"], r["k_off"],
                        r["k_on_default"], r["ic50_used"], r["APD90_ms"], r["APD90_ratio"],
                        r["qNet_uC_uF"], r["qNet_ratio"],
                        b2r.get("qNet_ratio"), b2r.get("APD90_ratio"),
                        (r["qNet_ratio"] - b2r["qNet_ratio"]) if b2r else None,
                        r["steady_range_ms"], r["steady_ok"], r["EAD_or_fail"]])
    print(f"\n result saved: {fjson}\n per-drug CSV: {fcsv}", flush=True)

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ttl = f"B3 dual-alpha host recheck · {'smoke' if SMOKE else '24 drugs'}"
    if R:
        ttl += f" · R1 ρq={R['R1_rho_qNet']:.3f}{'✓' if R['R1_pass'] else '×'}"
    fig.suptitle(ttl, fontsize=13)

    ax = axes[0, 0]
    if R and common:
        q2v = [b2dyn[r["drug"]]["qNet_ratio"] for r in common]
        q3v = [r["qNet_ratio"] for r in common]
        cls = [r["CiPA"] for r in common]
        cols = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}
        for c in (0, 1, 2):
            xs = [q2v[i] for i in range(len(common)) if cls[i] == c]
            ys = [q3v[i] for i in range(len(common)) if cls[i] == c]
            ax.scatter(xs, ys, c=cols[c], s=42, zorder=3, label=f"CiPA {c}")
        lim = [min(q2v + q3v) * 0.95, max(q2v + q3v) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("B2 qNet_ratio"); ax.set_ylabel("B3 qNet_ratio")
        ax.set_title(f"R1: rho={R['R1_rho_qNet']:.3f} (>=0.95 {'pass' if R['R1_pass'] else 'fail'})"
                     f"  R1b={R['R1b_dev_median']:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "smoke drug count <4\nR1/R1b not rehearsed", ha="center", va="center", fontsize=12)
        ax.set_title("R1 qNet comparison (enabled in the formal run)")

    ax = axes[0, 1]
    if R and common:
        a2v = [b2dyn[r["drug"]]["APD90_ratio"] for r in common]
        a3v = [r["APD90_ratio"] for r in common]
        for c in (0, 1, 2):
            xs = [a2v[i] for i in range(len(common)) if cls[i] == c]
            ys = [a3v[i] for i in range(len(common)) if cls[i] == c]
            ax.scatter(xs, ys, c=cols[c], s=42, zorder=3, label=f"CiPA {c}")
        lim = [min(a2v + a3v) * 0.95, max(a2v + a3v) * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("B2 APD90_ratio"); ax.set_ylabel("B3 APD90_ratio")
        ax.set_title(f"R1 APD90 convention: rho={R['R1_rho_APD90']:.3f}  R3 discordant pairs={R['R3_kendall_discordant']}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "smoke drug count <4\nR1(APD90)/R3 not rehearsed", ha="center", va="center", fontsize=12)
        ax.set_title("R1 APD90 comparison (enabled in the formal run)")

    ax = axes[1, 0]
    for r in results:
        c = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}[r["CiPA"]]
        mk = "D" if any(d["drug"] == r["drug"] for d in iks_strong) else "o"
        ax.scatter(r["CiPA"] + (np.random.rand() - 0.5) * 0.15, r["qNet_ratio"],
                   c=c, s=42, marker=mk, zorder=3, alpha=0.85)
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "intermediate", "high"])
    ax.set_ylabel("B3 qNet_ratio"); ax.set_title("B3 risk-class spread (diamond = strong-IKs-suppression subset)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"C1 {c1 * 100:.2f}% {'✓' if C1 else '×'}  "
             f"C2 {'✓' if checks.get('C2', {}).get('pass') else '×'}  "
             f"C3 {'✓' if checks.get('C3', {}).get('pass') else '×'}"]
    if vd_b3:
        lines.append(f"B3 V1 ρ={vd_b3['V1_rho_qNet']:.3f}  V2={vd_b3['V2_AUC_qNetSupp_hi_lo']:.3f}")
    if R:
        lines.append(f"R1 {'✓' if R['R1_pass'] else '×'}  R1b {'✓' if R['R1b_pass'] else '×'}  "
                     f"R3 {'✓' if R['R3_pass'] else '×'}" + (f"  R2 {'✓' if R['R2_pass'] else '×'}" if "R2_pass" in R else ""))
    if incr:
        lines.append(f"increment B3 vs B1: {'not inferior' if incr['not_inferior'] else 'inferior'}")
    lines.append(f"IKs strong subset: {[d['drug'] for d in iks_strong] if iks_strong else 'none'}")
    lines.append(f"elapsed {time.time() - t_start:.0f}s ({'smoke rehearsal' if SMOKE else 'formal'})")
    ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=11)

    fpng = os.path.join(BASE, f"2026-09-16_α模型_StageB_B3_双α化宿主复验_24药{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" figure saved: {fpng}", flush=True)

    if SMOKE:
        print("\n[smoke done] formal-run command (Spyder, about 1-1.5h parallel):\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-16_α模型_StageB_B3_双α化宿主复验_24药.py' --wdir\n"
              "  parallel: set env NWORKERS=6~8 first, then the same command", flush=True)


if __name__ == "__main__":
    main()
