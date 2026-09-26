# -*- coding: utf-8 -*-
"""
2026-09-14 · alpha-model Stage B · B2 alpha-substitution arm (24 drugs)
====================================================
In the ORd host the official hERG Markov (states 39-48) is frozen and IKr is replaced by the alpha-model four-table product gate:
  IKr_α = G_α · m · h · (1−D) · (v − EK_Nernst)
  dm/dt = (m_ss(v)−m)/τ_m(v)   dh/dt = (h_ss(v)−h)/τ_h(v)
  dD/dt = k_on*C*(1-h)*(1-D) - k_off*D      (inactivated-state-preferring binding, card-6 route)
The other six channels use static Hill multipliers with the same convention as B1. Two arms:
  dyn  = B2 dynamic binding (card-3 k_on/k_off tables);
  stat = B2s static control (D=0, IKr_alpha x mult_hERG official table) - 2x2 demixing.

Criteria (preregistration section 6 + section 11.7 v5 + section 11.8 v6, pinned before the run):
  V1/V2 verbatim identical to B1 (qNet sign pinned in v5: high risk -> qNet decrease);
  increment criterion: B2 V1 rho and V2 AUC >= the corresponding B1 values -> "alpha not inferior to official Markov";
  C1 control APD90 within 2% of the B0 formal value 252.118 ms (G_alpha calibrated independently per arm);
  C2 dofetilide 100xCmax: APD90 prolongation >40% and qNet DECREASE (post-v6-erratum convention);
  C3 nifedipine 100xCmax: APD90 shortening (CaL pathway probe).
  tau_act arms {0.3,0.5,1,2} (only the tau_m anchor table V>=-30 segment is scaled; the deactivation ladder is untouched) run ALL 24 drugs
  (preregistration section 11.6.4/11.6.11 pinned: the across-arm V1 rho range criterion requires same-n comparability with the baseline);
  Q10 arms {2.0,3.0} and the IC50 official arm run a 6-drug stratified subset (section 11.6.11).
  [v6.1 fix] the previous drug_list condition wrongly included TAU_ARM==1.0, so the three tau_act arms mistakenly ran the 6-drug subset;
  those three arms lose eligibility and must be rerun; the Q10/IC50 three arms with 6 drugs are by design and valid.

Environment variables: SMOKE=1 smoke; TAU_ACT_ARM in {0.3,0.5,1,2} (default 1);
  Q10 (default 2.5); IC50_SRC in {ka3,official} (default ka3);
  NWORKERS (default 1 serial); NBEATS_CTRL/NBEATS_DRUG/CAL_BEATS/CAL_ITERS.

Discipline: this side runs only ast.parse + SMOKE=1 smoke; the full run is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_StageB_B2_α替换_24药.py' --wdir
Output: _结果.json/.csv/.png next to this script (smoke carries the _冒烟 suffix; file names carry the arm tag).
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

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_CTRL = int(os.environ.get("SMOKE_BEATS", "40")) if SMOKE else int(os.environ.get("NBEATS_CTRL", "500"))
N_DRUG = int(os.environ.get("SMOKE_BEATS", "40")) if SMOKE else int(os.environ.get("NBEATS_DRUG", "500"))
CAL_BEATS = int(os.environ.get("CAL_BEATS", "20" if SMOKE else "100"))
CAL_ITERS = int(os.environ.get("CAL_ITERS", "7" if SMOKE else "10"))
TAU_ARM = float(os.environ.get("TAU_ACT_ARM", "1"))
Q10 = float(os.environ.get("Q10", "2.5"))
IC50_SRC = os.environ.get("IC50_SRC", "ka3")
NWORKERS = int(os.environ.get("NWORKERS", "1"))
CL = 1000.0
DT_REC = 0.1
B0_FORMAL_APD90 = 252.1181387725961   # B0 full-run last beat (on record 2026-09-14 14:51)
TEMP_SRC = 21.5                        # alpha four-table source-data temperature in C (Beattie 2018, v6 item 2.3)
TEMP_DST = 37.0
QFAC = Q10 ** ((TEMP_DST - TEMP_SRC) / 10.0)

assert TAU_ARM in (0.3, 0.5, 1.0, 2.0), f"TAU_ACT_ARM={TAU_ARM} not in the preregistered arm set"
assert IC50_SRC in ("ka3", "official")

# 24 drugs (card-4 preregistration section 3, verbatim identical to B1)
DRUGS_HIGH = ["azimilide", "bepridil", "disopyramide", "dofetilide", "ibutilide", "sotalol", "vandetanib"]
DRUGS_MID = ["astemizole", "chlorpromazine", "cisapride", "clarithromycin", "clozapine",
             "domperidone", "droperidol", "ondansetron", "pimozide", "risperidone", "terfenadine"]
DRUGS_LOW = ["diltiazem", "metoprolol", "mexiletine", "ranolazine", "tamoxifen", "verapamil"]
DRUGS_24 = DRUGS_HIGH + DRUGS_MID + DRUGS_LOW
SENS_6 = ["dofetilide", "sotalol", "cisapride", "ondansetron", "verapamil", "ranolazine"]  # v6 ②.11
SMOKE_DRUGS = ["dofetilide", "cisapride", "verapamil"]

# channel -> (csv IC50 column, csv h column, pars index); hERG does not go through fc (B2 replaces it with alpha), listed separately for the stat arm
CHANNELS_FC = [("LateNa", "Late_sodiumIC50", "Late_sodiumh", 2),
               ("PeakNa", "Peak_sodiumIC50", "Peak_sodiumh", 3),
               ("IKs", "IKsIC50", "IKsh", 4),
               ("IK1", "IK1IC50", "IK1h", 5),
               ("CaL", "CaLIC50", "CaLh", 6),
               ("Ito", "ItoIC50", "Itoh", 7)]
HERG_CSV = ("hERGIC50", "hERGh")


# ----------------------------------------------------------------------------
# engine loading (importlib; the B0/alpha files are used byte-for-byte unchanged)
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


# ----------------------------------------------------------------------------
# card-3 dynamic-binding parameters (v6 item 2.8 pinned: per-drug median of stratified-median regression of k_obs vs conc over provable datasets;
# if not provable -> library-level median default; k_off = k_on x IC50, IC50 from the ic50_self median or the official table)
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
# alpha four tables (16713003; tau seconds -> milliseconds x1000, temperature bridge /QFAC; tau_act arm scales only the V>=-30 anchors)
# ----------------------------------------------------------------------------
def build_alpha_tabs(eng):
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)

    def scaled(tab, arm=None):
        out = {}
        for x, ly in zip(tab.xs, tab.ly):
            f = 1000.0 / QFAC
            if arm is not None and x >= -30.0:
                f *= arm
            out[float(x)] = float(np.exp(ly)) * f
        return eng.Tab(out, log_y=True)

    return {"m_ss": tabs["m_ss"], "h_ss": tabs["h_ss"],
            "tau_m_ms": scaled(tabs["tau_m"], arm=TAU_ARM),
            "tau_h_ms": scaled(tabs["tau_h"])}


# ----------------------------------------------------------------------------
# B2 rhs: B0 port + IKr replacement (GKrfc=inf -> GKr=0.0 exact; Markov states frozen)
# ----------------------------------------------------------------------------
def build_b2(b0, P_ctrl, atabs, G_alpha, kon_per_nM_s, koff_per_s, C_nM, mode):
    P2 = list(P_ctrl)
    P2[1] = float("inf")                      # GKrfc -> GKr = 0.046/inf = 0.0 (exact)
    rhs0 = b0.build_model(P2)
    ko = P_ctrl[b0.PARS_NAMES.index("ko")]
    RT_F = 8314.0 * 310.0 / 96485.0           # physical constants at 310 K (same as B0; the pars T=37 is Celsius, only for the Markov Q10)
    # Acap/(F*vmyo) (same formula copied from B0 build_model, for the dy[3] correction)
    rad, L = 0.0011, 0.01
    vcell = 1000 * 3.14 * rad * rad * L
    Ageo = 2 * 3.14 * rad * rad + 2 * 3.14 * rad * L
    Acap_Fvmyo = 2 * Ageo / (96485.0 * 0.68 * vcell)
    m_ss, h_ss = atabs["m_ss"], atabs["h_ss"]
    tm_ms, th_ms = atabs["tau_m_ms"], atabs["tau_h_ms"]
    kon_ms = kon_per_nM_s * C_nM / 1000.0     # 1/ms (concentration already multiplied in)
    koff_ms = koff_per_s / 1000.0

    def rhs(t, y):
        dy, cur = rhs0(t, y)
        v = y[0]
        ma, ha, Da = y[50], y[51], y[52]
        for i in range(39, 49):               # official Markov hERG six states + bound states + D frozen
            dy[i] = 0.0
        dy_a = list(dy) + [0.0, 0.0, 0.0]
        dy_a[50] = (m_ss(v) - ma) / tm_ms(v)
        dy_a[51] = (h_ss(v) - ha) / th_ms(v)
        if mode == "dyn":
            dy_a[52] = kon_ms * (1.0 - ha) * (1.0 - Da) - koff_ms * Da
        # stat arm: D identically 0 (initial value 0 and dy=0)
        EK = RT_F * math.log(ko / y[3])       # host Nernst (ki dynamic, v6 item 2.7)
        IKr_a = G_alpha * ma * ha * (1.0 - Da) * (v - EK)
        dy_a[0] -= IKr_a
        dy_a[3] -= IKr_a * Acap_Fvmyo
        dy_a[49] += IKr_a
        return dy_a, (cur[0], cur[1], cur[2], cur[3], IKr_a, cur[5], cur[6])

    return rhs


def run_beats(b0, rhs, y0, nbeats, keep_last=True, label=""):
    from scipy.integrate import solve_ivp
    f = lambda t, y: rhs(t, y)[0]
    t_rec = np.arange(0.0, CL + 1e-9, DT_REC)
    y = np.array(y0, dtype=float)
    apd = np.full(nbeats, np.nan)
    apa = np.full(nbeats, np.nan)
    qnet = np.full(nbeats, np.nan)
    v_last = None
    t0 = time.time()
    for b in range(nbeats):
        q0 = y[49]
        sol = solve_ivp(f, (0.0, CL), y, method="LSODA", t_eval=t_rec,
                        rtol=1e-7, atol=1e-9)
        if not sol.success:
            break
        v = sol.y[0]
        y = sol.y[:, -1].copy()
        apd[b], apa[b], _, _ = b0.apd90(t_rec, v)
        qnet[b] = (y[49] - q0) * 1e-3
        if b == nbeats - 1 and keep_last:
            v_last = v.copy()
        if (b + 1) % 20 == 0 or b == 0:
            el = time.time() - t0
            print(f"    {label} beat {b + 1}/{nbeats}  APD90={apd[b]:7.2f}ms  "
                  f"qNet={qnet[b]:.4f}  took {el:.0f}s", flush=True)
    return apd, qnet, apa, v_last, y


def steady_flag(apd):
    ok = apd[~np.isnan(apd)]
    if len(ok) < 10:
        return float("nan"), False
    r = float(np.max(ok[-10:]) - np.min(ok[-10:]))
    return r, bool(r < 0.5)


def last_valid(x):
    ok = x[~np.isnan(x)]
    return float(ok[-1]) if len(ok) else float("nan")


def last_med(x, k=5):
    """Median of the last k valid values (calibration reading, robust to transient noise)."""
    ok = x[~np.isnan(x)]
    if len(ok) == 0:
        return float("nan")
    return float(np.median(ok[-k:]))


# ----------------------------------------------------------------------------
# G_alpha calibration (C1: control last-beat APD90 within 2% of B0; bisection in log space)
# ----------------------------------------------------------------------------
def calibrate_G(b0, atabs, P_ctrl, y0):
    """Two-phase calibration: first a fixed grid to find the healthy region (APA>=90 mV), then bisection on the APD90 target inside it.
    Physical prior: healthy G_alpha magnitude ~0.1-1 (pA/pF)/mV (repolarization needs ~1.5 pA/pF at -40 mV).
    If no bracketable healthy point on the grid -> take the healthy point with min |APD90-target|, ok=False, registered as borderline."""
    target, tol = B0_FORMAL_APD90, 0.02 * B0_FORMAL_APD90

    def probe(lg):
        rhs = build_b2(b0, P_ctrl, atabs, 10.0 ** lg, 0.0, 0.0, 0.0, "stat")
        apd, qnet, apa, _, _ = run_beats(b0, rhs, y0, CAL_BEATS, keep_last=False, label="calib")
        return last_med(apd, 10), last_med(apa, 10), last_valid(qnet)

    grid = [-1.0, -0.5, 0.0, 0.5]
    pts = []
    for lg in grid:
        a, pa, qn = probe(lg)
        pts.append((lg, a, pa, qn))
        print(f"  [calib·grid] G=10^{lg:+.1f}={10 ** lg:.3g}  APD90={a:.2f}ms  APA={pa:.1f}mV"
              f"  qNet={qn:.4f}  {'healthy' if pa >= 90 else 'pathological(APA<90)'}", flush=True)
    healthy = [(lg, a) for lg, a, pa, qn in pts if pa >= 90.0 and not math.isnan(a)]
    bracket = None
    for (l1, a1), (l2, a2) in zip(healthy, healthy[1:]):
        if (a1 - target) * (a2 - target) <= 0 and a1 != a2:
            bracket = (l1, l2)
            break
    if bracket:
        lo, hi = bracket
        for it in range(CAL_ITERS):
            mid = 0.5 * (lo + hi)
            a, pa, qn = probe(mid)
            print(f"  [calib·bisect {it + 1}/{CAL_ITERS}] G=10^{mid:.3f}={10 ** mid:.4g}"
                  f"  APD90={a:.2f}ms  APA={pa:.1f}mV  qNet={qn:.4f}", flush=True)
            if pa < 90.0 or math.isnan(a) or a > target:
                lo = mid                       # insufficient repolarization -> G too small
            else:
                hi = mid
            if pa >= 90.0 and not math.isnan(a) and abs(a - target) <= tol:
                return 10.0 ** mid, a, True
        a, pa, qn = probe(0.5 * (lo + hi))
        return 10.0 ** (0.5 * (lo + hi)), a, bool(pa >= 90 and not math.isnan(a)
                                                  and abs(a - target) <= tol)
    if healthy:
        lg, a = min(healthy, key=lambda p: abs(p[1] - target))
        print(f"  [calib] no bracketable region on the grid; taking the closest healthy point G=10^{lg:+.1f}", flush=True)
        return 10.0 ** lg, a, bool(abs(a - target) <= tol)
    lg, a, pa, qn = min(pts, key=lambda p: abs((p[1] if not math.isnan(p[1]) else 9e3) - target))
    print(f"  [calib] no healthy point (all APA <90 mV) - the alpha replacement cannot produce a normal AP in this host; registered as-is", flush=True)
    return 10.0 ** lg, a, False


# ----------------------------------------------------------------------------
# AUC (Mann-Whitney, same formula as B1)
# ----------------------------------------------------------------------------
def auc_pos(pos, neg):
    n_g, n_t = 0.0, 0
    for a in pos:
        for b in neg:
            n_g += 1.0 if a > b else (0.5 if a == b else 0.0)
            n_t += 1
    return n_g / n_t if n_t else float("nan")


def judge(results, APD_ctrl, qNet_ctrl):
    from scipy.stats import spearmanr
    risk = np.array([r["CiPA"] for r in results], dtype=float)
    qr = np.array([r["qNet_ratio"] for r in results], dtype=float)
    ar = np.array([r["APD90_ratio"] for r in results], dtype=float)
    okq, oka = ~np.isnan(qr), ~np.isnan(ar)
    rho_q, p_q = spearmanr(risk[okq], qr[okq])
    rho_a, p_a = spearmanr(risk[oka], ar[oka])
    sup = 1.0 - qr
    hi_s = sup[(risk == 2) & okq]; lo_s = sup[(risk == 0) & okq]; mid_s = sup[(risk == 1) & okq]
    hi_a = ar[(risk == 2) & oka];   lo_a = ar[(risk == 0) & oka];   mid_a = ar[(risk == 1) & oka]
    auc_q = auc_pos(hi_s, lo_s); auc_a = auc_pos(hi_a, lo_a)
    V1 = bool(rho_q <= -0.60 and p_q < 0.01)
    V2 = bool(auc_q >= 0.80)
    return {"V1_rho_qNet": float(rho_q), "V1_p": float(p_q), "V1_pass": V1,
            "V2_AUC_qNetSupp_hi_lo": float(auc_q), "V2_pass": V2,
            "reg_APD90_rho": float(rho_a), "reg_APD90_p": float(p_a),
            "reg_AUC_APD90_hi_lo": float(auc_a),
            "reg_AUC_qNetSupp_mid_lo": float(auc_pos(mid_s, lo_s)),
            "reg_AUC_APD90_mid_lo": float(auc_pos(mid_a, lo_a)),
            "reg_n_EAD_or_fail": int(sum(r["EAD_or_fail"] for r in results)),
            "reg_n_steady_fail": int(sum(not r["steady_ok"] for r in results))}


# ----------------------------------------------------------------------------
# single-drug task (shared by both arms; worker for NWORKERS parallelism)
# ----------------------------------------------------------------------------
def one_drug(task):
    (drug, mode, C_scale, G_alpha) = task
    b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")
    rows = load_cipa(CIPA_CSV)
    ka3, lib_med = ka3_params()
    row = rows[drug]
    C = float(row["therapeutic"]) * C_scale
    P_ctrl = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    P_d = list(P_ctrl)
    mults = {}
    for nm, cIC, ch, idx in CHANNELS_FC:
        IC50, h = float(row[cIC]), float(row[ch])
        if IC50 <= 0.0 or h <= 0.0:
            mults[nm] = 1.0
            continue
        mult = 1.0 / (1.0 + (C / IC50) ** h)
        mults[nm] = mult
        P_d[idx] = 1.0 / mult
    IC50, h = float(row[HERG_CSV[0]]), float(row[HERG_CSV[1]])
    mult_herg = 1.0 / (1.0 + (C / IC50) ** h) if (IC50 > 0 and h > 0) else 1.0
    ic50_dyn = ka3[drug]["ic50_self"] if IC50_SRC == "ka3" else IC50
    if ic50_dyn is None or ic50_dyn <= 0:
        ic50_dyn = IC50 if IC50 > 0 else float("inf")
    kon = ka3[drug]["k_on"]
    koff = kon * ic50_dyn

    eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")
    atabs = build_alpha_tabs(eng)
    with open(B0_SS, encoding="utf-8") as fh:
        y_ss = json.load(fh)["states50"]
    v0 = y_ss[0]
    y0 = list(y_ss) + [atabs["m_ss"](v0), atabs["h_ss"](v0), 0.0]

    G_alpha = task[3]   # passed in after calibration
    rhs = build_b2(b0, P_d, atabs, G_alpha, kon, koff, C, mode)
    if mode == "stat":
        # B2s: D=0, IKr_alpha x mult_hERG - equivalent implementation: G_alpha x mult
        rhs = build_b2(b0, P_d, atabs, G_alpha * mult_herg, 0.0, 0.0, 0.0, "stat")
    apd, qnet, _, v_last, _ = run_beats(b0, rhs, y0, N_DRUG, label=f"{drug}/{mode}")
    nd = int(np.sum(~np.isnan(apd)))
    APD_d, qNet_d = last_valid(apd), last_valid(qnet)
    rng, okd = steady_flag(apd)
    return {"drug": drug, "mode": mode, "CiPA": int(float(row["CiPA"])), "C_nM": C,
            "mult": mults, "mult_hERG": mult_herg, "k_on": kon, "k_off": koff,
            "k_on_default": ka3[drug]["k_on_default"], "ic50_used": ic50_dyn,
            "APD90_ms": APD_d, "qNet_uC_uF": qNet_d,
            "steady_range_ms": rng, "steady_ok": okd,
            "EAD_or_fail": bool(nd < N_DRUG or math.isnan(APD_d)),
            "n_beats_ok": nd, "trace": (v_last.tolist() if v_last is not None else None)}


def main():
    t_start = time.time()
    arm_tag = f"tauA{TAU_ARM:g}_Q{Q10:g}_{IC50_SRC}"
    print("=" * 74, flush=True)
    print(" alpha-model Stage B · B2 alpha-substitution arm (alpha hERG + six-channel static Hill, 24 drugs, dyn/stat two arms)", flush=True)
    print(f" mode: {'smoke SMOKE' if SMOKE else 'full'}  arm tag: {arm_tag}"
          f"  TAU_ACT_ARM={TAU_ARM} Q10={Q10}(÷{QFAC:.3f}) IC50_SRC={IC50_SRC} NWORKERS={NWORKERS}", flush=True)
    print(f" control {N_CTRL} beats + calibration ({CAL_BEATS} beats x <={CAL_ITERS}) + drug {N_DRUG} beats", flush=True)
    print("=" * 74, flush=True)

    if not os.path.exists(B0_SS):
        raise RuntimeError("StageB_B0稳态_CL1000.json missing - the B1 full run must save it first (v6 item 2.5)")
    b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")
    P_ctrl = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    with open(B0_SS, encoding="utf-8") as fh:
        y_ss = json.load(fh)["states50"]

    eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")
    atabs = build_alpha_tabs(eng)
    ka3, lib_med = ka3_params()
    n_def = sum(1 for d in DRUGS_24 if ka3[d]["k_on_default"])
    print(f"[card-3] k_on library-level median default {lib_med:.3e} 1/(nM*s); defaulted drugs {n_def}/24"
          f"（{[d for d in DRUGS_24 if ka3[d]['k_on_default']]}）", flush=True)

    rows = load_cipa(CIPA_CSV)
    # v6.1: the all-24-drug condition only checks Q10/IC50_SRC (every tau_act arm must run the full set, preregistration section 11.6.4/11)
    drug_list = SMOKE_DRUGS if SMOKE else (DRUGS_24 if Q10 == 2.5 and IC50_SRC == "ka3"
                                           else SENS_6)
    if not SMOKE and drug_list == DRUGS_24:
        n2 = sum(1 for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 2)
        n1 = sum(1 for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 1)
        n0 = sum(1 for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 0)
        assert (n2, n1, n0) == (7, 11, 6)
        print(f"[list] 24 drugs in place; labels high 7 / intermediate 11 / low 6 match card-4 section 3", flush=True)
    else:
        print(f"[list] this arm's drug list ({len(drug_list)}): {drug_list}", flush=True)

    # ---------- G_alpha calibration + control ----------
    v0 = y_ss[0]
    y0 = list(y_ss) + [atabs["m_ss"](v0), atabs["h_ss"](v0), 0.0]
    print(f"\n[G_alpha calib] initial m={y0[50]:.4f} h={y0[51]:.4f} @v={v0:.2f}mV", flush=True)
    G_alpha, apd_cal, cal_ok = calibrate_G(b0, atabs, P_ctrl, y0)
    print(f"[G_alpha calib] G_alpha={G_alpha:.4g}  calibration last-beat APD90={apd_cal:.2f}ms  "
          f"C1(<=2%) -> {'pass' if cal_ok else 'borderline/fail - registered'}", flush=True)

    print(f"\n[control] {N_CTRL} beats ...", flush=True)
    rhs_c = build_b2(b0, P_ctrl, atabs, G_alpha, 0.0, 0.0, 0.0, "stat")
    apd_c, qnet_c, _, v_c, _ = run_beats(b0, rhs_c, y0, N_CTRL, label="control")
    APD_ctrl, qNet_ctrl = last_valid(apd_c), last_valid(qnet_c)
    r_c, ok_c = steady_flag(apd_c)
    c1 = abs(APD_ctrl - B0_FORMAL_APD90) / B0_FORMAL_APD90
    C1 = bool(c1 <= 0.02)
    print(f"  control last-beat APD90={APD_ctrl:.3f}ms qNet={qNet_ctrl:.4f}uC/uF  "
          f"steady range={r_c:.3f}ms  C1 |diff|={c1 * 100:.2f}% -> {'pass' if C1 else 'fail'}", flush=True)

    # ---------- C2 / C3 self-checks (section 7, post-v6-erratum convention) ----------
    checks = {}
    for tagc, drugc, expect in [("C2", "dofetilide", "APD90 +40% and qNet decrease"),
                                ("C3", "nifedipine", "APD90 shortening")]:
        if drugc not in rows:
            print(f"  [{tagc}] {drugc} not in newCiPA.csv, skipped (registry)", flush=True)
            checks[tagc] = {"skip": True}
            continue
        rec = one_drug((drugc, "dyn", 100.0, G_alpha))
        r_apd = rec["APD90_ms"] / APD_ctrl
        r_q = rec["qNet_uC_uF"] / qNet_ctrl
        if tagc == "C2":
            ok = bool(r_apd > 1.40 and r_q < 1.0)
        else:
            ok = bool(r_apd < 1.0)
        checks[tagc] = {"drug": drugc, "C": "100xCmax", "APD90_ratio": r_apd,
                        "qNet_ratio": r_q, "pass": ok, "expect": expect}
        print(f"  [{tagc}] {drugc} 100×Cmax: APD90 {r_apd:.3f}×  qNet {r_q:.3f}×  "
              f"(required: {expect}) -> {'pass' if ok else 'fail'}", flush=True)

    # ---------- 24 drugs x two arms ----------
    tasks = [(d, mode, 1.0, G_alpha) for mode in ("dyn", "stat") for d in drug_list]
    print(f"\n[24 drugs two arms] {len(tasks)} tasks x {N_DRUG} beats, NWORKERS={NWORKERS}", flush=True)
    results = []
    if NWORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            for rec in ex.map(one_drug, tasks):
                results.append(rec)
                print(f"  [done] {rec['drug']}/{rec['mode']} APD90={rec['APD90_ms']:.2f}  "
                      f"qNet={rec['qNet_uC_uF']:.4f}", flush=True)
    else:
        for i, tk in enumerate(tasks):
            t0 = time.time()
            rec = one_drug(tk)
            results.append(rec)
            print(f"  [{i + 1}/{len(tasks)}] {rec['drug']:15s}/{rec['mode']:4s} "
                  f"CiPA={rec['CiPA']} C={rec['C_nM']:9.3f}nM  "
                  f"APD90={rec['APD90_ms']:8.2f}  qNet={rec['qNet_uC_uF']:.4f}  "
                  f"steady {rec['steady_range_ms']:.2f}ms{'v' if rec['steady_ok'] else 'x'}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    for rec in results:
        rec["APD90_ratio"] = rec["APD90_ms"] / APD_ctrl if not math.isnan(rec["APD90_ms"]) else float("nan")
        rec["qNet_ratio"] = rec["qNet_uC_uF"] / qNet_ctrl if not math.isnan(rec["qNet_uC_uF"]) else float("nan")

    # ---------- criteria (per arm; smoke = same code path rehearsal, no verdict) ----------
    verdicts = {}
    for mode in ("dyn", "stat"):
        sub = [r for r in results if r["mode"] == mode and r["drug"] in drug_list]
        if len(sub) < 4:
            continue
        vd = judge(sub, APD_ctrl, qNet_ctrl)
        verdicts[mode] = vd
        lab = "criterion-path rehearsal (smoke/sensitivity arm, not a verdict)" if (SMOKE or drug_list != DRUGS_24) else "criteria"
        arm_name = "B2 dynamic" if mode == "dyn" else "B2s static"
        print("\n" + "-" * 74, flush=True)
        print(f" [{arm_name}·{lab}] V1 rho(qNet_ratio, risk order)={vd['V1_rho_qNet']:.3f}(p={vd['V1_p']:.2e})"
              f" <=-0.60 and p<0.01 -> {'pass' if vd['V1_pass'] else 'fail'}", flush=True)
        print(f" [{arm_name}·{lab}] V2 AUC(qNet suppression, high vs low)={vd['V2_AUC_qNetSupp_hi_lo']:.3f}"
              f" >=0.80 -> {'pass' if vd['V2_pass'] else 'fail'}", flush=True)
        print(f" [{arm_name}·registry] APD90: rho={vd['reg_APD90_rho']:.3f} AUC high-low={vd['reg_AUC_APD90_hi_lo']:.3f}; "
              f"EAD/failure {vd['reg_n_EAD_or_fail']} steady-not-reached {vd['reg_n_steady_fail']}", flush=True)

    # ---------- increment criterion: B2 vs B1 (preregistration section 6, judged only when B1 result on record) ----------
    incr = None
    if os.path.exists(B1_JSON) and "dyn" in verdicts:
        b1v = json.load(open(B1_JSON, encoding="utf-8"))["verdict"]
        incr = {"B1_V1_rho": b1v["V1_rho_qNet"], "B1_V2_AUC": b1v["V2_AUC_qNetSupp_hi_lo"],
                "B2_dyn_V1_rho": verdicts["dyn"]["V1_rho_qNet"],
                "B2_dyn_V2_AUC": verdicts["dyn"]["V2_AUC_qNetSupp_hi_lo"]}
        incr["not_inferior"] = bool(verdicts["dyn"]["V1_rho_qNet"] <= b1v["V1_rho_qNet"]
                                    and verdicts["dyn"]["V2_AUC_qNetSupp_hi_lo"] >= b1v["V2_AUC_qNetSupp_hi_lo"])
        # more negative rho is better: B2 rho <= B1 rho means not inferior; larger AUC is better
        print(f"\n [increment] B2dyn vs B1: rho {incr['B2_dyn_V1_rho']:.3f} vs {incr['B1_V1_rho']:.3f}; "
              f"AUC {incr['B2_dyn_V2_AUC']:.3f} vs {incr['B1_V2_AUC']:.3f} -> "
              f"{'not inferior' if incr['not_inferior'] else 'inferior'}", flush=True)
    else:
        print("\n [increment] B1 formal result not yet saved; the increment criterion is deferred until B1 completes.", flush=True)

    overall = None
    if not SMOKE and drug_list == DRUGS_24 and "dyn" in verdicts:
        overall = bool(verdicts["dyn"]["V1_pass"] and verdicts["dyn"]["V2_pass"] and C1
                       and checks.get("C2", {}).get("pass") and checks.get("C3", {}).get("pass"))
        print(f"\n B2 overall verdict (dyn main line): {'PASS' if overall else 'FAIL'}"
              f" (joint V1/V2 + C1 + C2 + C3)", flush=True)

    # ---------- save ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(BASE, f"2026-09-14_α模型_StageB_B2_α替换_24药_{arm_tag}{tag}.json")
    out = {"meta": {"smoke": SMOKE, "arm": {"TAU_ACT_ARM": TAU_ARM, "Q10": Q10, "QFAC": QFAC,
                                             "IC50_SRC": IC50_SRC, "anchor_cell": "16713003"},
                    "G_alpha": G_alpha, "cal_apd90": apd_cal, "cal_ok": cal_ok,
                    "control_APD90_ms": APD_ctrl, "control_qNet_uC_uF": qNet_ctrl,
                    "C1_rel_diff": c1, "drug_list": drug_list,
                    "k_on_default_drugs": [d for d in drug_list if ka3[d]["k_on_default"]],
                    "runtime_s": time.time() - t_start},
           "checks": checks, "results": [{k: v for k, v in r.items() if k != "trace"} for r in results],
           "traces": {"control": v_c.tolist() if v_c is not None else None,
                      **{f"{r['drug']}/{r['mode']}": r["trace"] for r in results if r["trace"]}},
           "verdicts": verdicts, "incremental_vs_B1": incr,
           "B2_overall_pass": overall}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    fcsv = os.path.join(BASE, f"2026-09-14_α模型_StageB_B2_α替换_24药_{arm_tag}{tag}.csv")
    with open(fcsv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drug", "mode", "CiPA", "C_nM", "k_on", "k_off", "k_on_default", "ic50_used",
                    "mult_hERG", "APD90_ms", "APD90_ratio", "qNet_uC_uF", "qNet_ratio",
                    "steady_range_ms", "steady_ok", "EAD_or_fail"])
        for r in sorted(results, key=lambda x: (x["mode"], x["drug"])):
            w.writerow([r["drug"], r["mode"], r["CiPA"], r["C_nM"], r["k_on"], r["k_off"],
                        r["k_on_default"], r["ic50_used"], r["mult_hERG"], r["APD90_ms"],
                        r["APD90_ratio"], r["qNet_uC_uF"], r["qNet_ratio"],
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
    ttl = f"B2 alpha-substitution · {arm_tag} · {'smoke' if SMOKE else '24 drugs'}"
    if "dyn" in verdicts:
        ttl += (f" · dyn V1 ρ={verdicts['dyn']['V1_rho_qNet']:.3f}"
                f"{'✓' if verdicts['dyn']['V1_pass'] else '×'}"
                f" V2={verdicts['dyn']['V2_AUC_qNetSupp_hi_lo']:.3f}"
                f"{'✓' if verdicts['dyn']['V2_pass'] else '×'}")
    fig.suptitle(ttl, fontsize=13)

    for ax, key, ylab in [(axes[0, 0], "qNet_ratio", "qNet ratio"), (axes[0, 1], "APD90_ratio", "APD90 ratio")]:
        for mode, mk in [("dyn", "o"), ("stat", "s")]:
            for cls, col in [(0, "tab:green"), (1, "tab:orange"), (2, "tab:red")]:
                xs, ys = [], []
                for r in results:
                    if r["mode"] == mode and r["CiPA"] == cls and not math.isnan(r[key]):
                        xs.append(cls + (0.12 if mode == "stat" else -0.12) + (np.random.rand() - 0.5) * 0.12)
                        ys.append(r[key])
                ax.scatter(xs, ys, c=col, s=30, marker=mk, zorder=3,
                           label=f"{mode}" if cls == 0 else None, alpha=0.85)
        ax.axhline(1.0, color="k", lw=0.7, ls="--")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "intermediate", "high"])
        ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    ax = axes[1, 0]
    for mode, col in [("dyn", "tab:blue"), ("stat", "tab:gray")]:
        if mode not in verdicts:
            continue
        sub = [r for r in results if r["mode"] == mode]
        hi = [1.0 - r["qNet_ratio"] for r in sub if r["CiPA"] == 2 and not math.isnan(r["qNet_ratio"])]
        lo = [1.0 - r["qNet_ratio"] for r in sub if r["CiPA"] == 0 and not math.isnan(r["qNet_ratio"])]
        allv = sorted(set(hi + lo))
        if hi and lo and len(allv) > 1:
            tprs, fprs = [1.0], [1.0]
            for th in sorted(allv, reverse=True):
                tprs.append(float(np.mean([s >= th for s in hi])))
                fprs.append(float(np.mean([s >= th for s in lo])))
            tprs.append(0.0); fprs.append(0.0)
            ax.plot(fprs, tprs, "o-", ms=3, color=col,
                    label=f"{mode} AUC={verdicts[mode]['V2_AUC_qNetSupp_hi_lo']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC high vs low (qNet suppression)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    t_rec = np.arange(0.0, CL + 1e-9, DT_REC)
    if v_c is not None:
        ax.plot(t_rec, v_c, "k", lw=1.2, label="control")
    for r in results:
        if r["drug"] in SMOKE_DRUGS[:1] and r["trace"]:
            ax.plot(t_rec, r["trace"], lw=1.0, label=f"{r['drug']}/{r['mode']}", alpha=0.8)
    ax.set_title("last-beat AP"); ax.set_xlabel("t (ms)"); ax.set_ylabel("v (mV)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fpng = os.path.join(BASE, f"2026-09-14_α模型_StageB_B2_α替换_24药_{arm_tag}{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" figure saved: {fpng}", flush=True)

    if SMOKE:
        print("\n[smoke done] full-run commands (Spyder; main line about 3h, each sensitivity arm about 45 min):\n"
              "  主线:  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_StageB_B2_α替换_24药.py' --wdir\n"
              "  tau_act arm: set env TAU_ACT_ARM=0.3 / 0.5 / 2 first, then the same command\n"
              "  Q10 arm: Q10=2.0 / 3.0; IC50 arm: IC50_SRC=official", flush=True)


if __name__ == "__main__":
    main()
