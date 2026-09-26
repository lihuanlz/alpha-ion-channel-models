# 2026-09-14_alpha-model_drug-card4_AP-waveform_clinical-risk_crosscheck.py
# ============================================================================
# Drug card 4 · clinical-risk cross-check of drug block under the AP waveform (preregistered verdict card)
# Preregistration: 结果\预注册_α模型_药物卡4_AP波形临床风险对拍_2026-09-14.md (criteria pinned before the run)
#
# Model: I = G·m·h·(1-D)·(V-E_rev); engine four tables untouched verbatim (imports the formal engine)
# Drug module (trapping type): dD/dt = k_on·[D]·m·(1-D) - k_off·D
# Parameters: slope of the pooled linear fit of k_obs([D]) = k_on (card-3 sealed result, k_obs>=0.9 saturation excluded);
#   k_off := k_on·IC50 (consistency anchoring: sustained-depolarization steady state returns to the measured Hill balance D_ss(IC50)=0.5;
#   the fit intercept is registered for diagnosis only and does not enter the model -- Milnes mainly observes the binding onset, so the intercept is unreliable).
#   slope<=0 or valid concentration bands <2 -> static degraded D=Hill(IC50,nH) clamp, flag registered
# Anchor: newCiPA.csv (FDA/CiPA official; therapeutic = free Cmax nM; CiPA 2 high / 1 medium / 0 low)
# Criteria: V1 Spearman rho(S_eq@1x, risk order)>=0.60 and one-sided p<0.01; V2 AUC(high vs low)>=0.80;
#   V3/steady-state/sensitivity registered only. Synthetic self-check with three conditions (monotonic / trapping limit / Hill consistency) before the run.
# v2 (amended after smoke, before the full run; preregistration section 8 on record):
#   (1) k_off changed to consistency anchoring k_off=k_on·IC50 (fit intercept registered only);
#   (2) main verdict metric switched to S_eq (duty-cycle-adjusted analytic equilibrium: S_eq=[D]·<m>/([D]·<m>+IC50),
#       <m> = time average of m in the last control-AP loop) -- the finite-train S@12 loops becomes a transient registry,
#       because slow-dissociation drugs are far from equilibrium within 105.9 s (clinical dosing is chronic = equilibrium).
# Run: Spyder %runfile '...py' --wdir. SMOKE=1 smoke: 3 drugs (dof/ond/ver), protocol verbatim identical.
# ============================================================================
import os
import json
import csv
import importlib.util
import numpy as np
import scipy.io as sio
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1e-4
E_REV = -88.33
SMOKE = os.environ.get("SMOKE", "0") == "1"

F_CARD3 = os.path.join(HERE, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")
F_CIPA = os.path.join(HERE, "CiPA官方锚", "newCiPA.csv")
F_PROTO = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/"
           "wA1/代码A1_16704007_流程固定重标卡_2026-09-11/data/protocols/ap_protocol.mat")

N_LOOP = 12                      # 8.8245s x 12 ~ 105.9s ~ 84 beats
CONC_MULT = [1, 3, 10, 25]       # x free Cmax
SS_CHECK_LOOP = 6                # steady-state diagnostic: loop 6 vs loop 12
SAT_KOBS = 0.9                   # k_obs saturation exclusion threshold

DRUGS_24 = ["azimilide", "bepridil", "disopyramide", "dofetilide", "ibutilide", "sotalol",
            "vandetanib",
            "astemizole", "chlorpromazine", "cisapride", "clarithromycin", "clozapine",
            "domperidone", "droperidol", "ondansetron", "pimozide", "risperidone",
            "terfenadine",
            "diltiazem", "metoprolol", "mexiletine", "ranolazine", "tamoxifen", "verapamil"]
SMOKE_DRUGS = ["dofetilide", "ondansetron", "verapamil"]


# ---------- engine import (filename starts with a digit, use importlib) ----------
def load_engine():
    fp = os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py")
    spec = importlib.util.spec_from_file_location("alpha_engine", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_anchor_tabs(eng):
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    return eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)


# ---------- drug-module forward (verbatim extension of the engine's forward: adds the D state) ----------
def forward_drug(V, tabs, kon_nM, koff, conc_nM, hill_clamp=None):
    """kon_nM: 1/(nM·s); koff: 1/s; hill_clamp: constant D under static degradation.
    Returns a per-loop Q list (I = G·m·h·(1-D)·(V-E_rev), integral dt segmented by loop)."""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    G = tabs["G"] if tabs["G"] else 1.0
    n = len(V)
    nlp = n // N_LOOP
    m0, h0, D = ms[0], hss[0], 0.0
    Q = np.zeros(N_LOOP)
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hss[i] + (h0 - hss[i]) * eh[i]
        if hill_clamp is not None:
            D = hill_clamp
        else:
            D += DT * (kon_nM * conc_nM * m0 * (1.0 - D) - koff * D)
            if D < 0.0:
                D = 0.0
        Q[i // nlp] += G * m0 * h0 * (1.0 - D) * (v[i] - E_REV)
    return Q * DT


# ---------- card 3 -> per-drug parameters ----------
def drug_params(card3, drug):
    ic50s, nhs, ic50_lab, nh_lab = [], [], [], []
    pts = []  # (conc_nM, k_obs)
    n_sat = 0
    for key, ds in card3["datasets"].items():
        if key.split("|")[1] != drug:
            continue
        L1 = ds.get("L1") or {}
        if L1.get("ic50_self"):
            ic50s.append(L1["ic50_self"])
            nhs.append(L1["nh_self"])
        if L1.get("ic50_lab"):
            ic50_lab.append(L1["ic50_lab"])
            nh_lab.append(L1["nh_lab"])
        for u in ds.get("units") or []:
            k = u.get("k_obs")
            if k is None:
                continue
            if k >= SAT_KOBS:
                n_sat += 1
                continue
            pts.append((u["conc_nM"], k))
    ic50 = float(np.median(ic50s)) if ic50s else None
    nh = float(np.median(nhs)) if nhs else None
    kon = koff = None
    koff_fit = None
    degen = None
    if pts:
        tiers = sorted({c for c, _ in pts})
        if len(tiers) >= 2:
            x = np.array([c for c, _ in pts])
            y = np.array([k for _, k in pts])
            sl, ic, r, p, se = stats.linregress(x, y)
            koff_fit = float(ic)
            if sl > 0 and ic50:
                kon = float(sl)
                koff = float(sl * ic50)      # consistency anchoring: D_ss(IC50)=0.5
            elif sl > 0:
                degen = "静态退化(无IC50)"
            else:
                degen = "静态退化(斜率≤0)"
        else:
            degen = "静态退化(浓度档<2)"
    else:
        degen = "静态退化(无k_obs)"
    return dict(ic50=ic50, nh=nh, ic50_lab=float(np.median(ic50_lab)) if ic50_lab else None,
                nh_lab=float(np.median(nh_lab)) if nh_lab else None,
                kon=kon, koff=koff, koff_fit=koff_fit, degen=degen, n_sat=n_sat, n_pts=len(pts))


def hill_block(ic50_nM, nh, conc_nM):
    if ic50_nM is None or nh is None or ic50_nM <= 0:
        return None
    return float(conc_nM ** nh / (ic50_nM ** nh + conc_nM ** nh))


# ---------- synthetic self-check (three conditions pinned in the preregistration) ----------
def synth_check(V1, tabs):
    print("\n[synthetic self-check]", flush=True)
    kon, koff, ic50 = 1e-4, 1e-3, 10.0     # self-consistent: IC50 = koff/kon
    ok = True
    # (a) AP-waveform monotonicity: S@25x > S@1x (Cmax=10nM virtual)
    q1 = forward_drug(V1, tabs, kon, koff, 10.0)[-1]
    q25 = forward_drug(V1, tabs, kon, koff, 250.0)[-1]
    q0 = forward_drug(V1, tabs, 0.0, 0.0, 0.0)[-1]
    s1, s25 = 1 - q1 / q0, 1 - q25 / q0
    ca = s25 > s1 > 0
    print(f"  (a) monotonic: S@1x={s1:.4f} S@25x={s25:.4f} -> {'pass' if ca else 'fail'}", flush=True)
    ok &= ca
    # (b) k_off=0 trapping limit: D only increases (direct numerical check of the integral)
    v = V1.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    em = np.exp(-DT / tm)
    m0, D, dmin = ms[0], 0.0, 0.0
    for i in range(len(v)):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        dD = DT * (kon * 10.0 * m0 * (1.0 - D))
        D += dD
        dmin = min(dmin, dD)
    cb = dmin >= -1e-15
    print(f"  (b) trapping limit (k_off=0): min(dD)={dmin:.2e} -> {'pass' if cb else 'fail'}", flush=True)
    ok &= cb
    # (c) sustained +20mV (m->1) @ [D]=IC50: D_ss->0.5
    mss1 = float(np.exp(np.interp(20.0, tabs["m_ss"].xs, tabs["m_ss"].ly)))
    Dss = kon * ic50 * mss1 / (kon * ic50 * mss1 + koff)
    cc = abs(Dss - 0.5) < 0.02
    print(f"  (c) Hill consistency: m_ss(+20)={mss1:.4f} D_ss={Dss:.4f} (should be ~0.5) -> {'pass' if cc else 'fail'}",
          flush=True)
    ok &= cc
    print(f"  synthetic self-check {'pass' if ok else 'fail -- whole card downgraded to registry'}", flush=True)
    return ok


def auc_high_low(s_by_drug, risk_by_drug):
    hi = [s_by_drug[d] for d in s_by_drug if risk_by_drug[d] == 2]
    lo = [s_by_drug[d] for d in s_by_drug if risk_by_drug[d] == 0]
    if not hi or not lo:
        return None
    u, _ = stats.mannwhitneyu(hi, lo, alternative="greater")
    return float(u / (len(hi) * len(lo)))


def main():
    print("=" * 76, flush=True)
    print(" drug card 4 · AP-waveform clinical-risk cross-check (preregistered verdict card)", flush=True)
    print("=" * 76, flush=True)

    eng = load_engine()
    tabs = build_anchor_tabs(eng)
    print(f"[anchor] cell 16713003 G={tabs['G']:.4f} gflag={tabs['gflag']}", flush=True)

    V1 = sio.loadmat(F_PROTO)["T"].flatten().astype(float)
    n1 = len(V1)
    V = np.tile(V1, N_LOOP)
    print(f"[protocol] ap_protocol single loop {n1 * DT:.4f}s x {N_LOOP} = {n1 * N_LOOP * DT:.1f}s", flush=True)

    synth_ok = synth_check(V, tabs)

    card3 = json.load(open(F_CARD3, encoding="utf-8"))
    cmax, risk = {}, {}
    with open(F_CIPA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["drug"].strip()
            if d in DRUGS_24:
                cmax[d] = float(row["therapeutic"])     # nM
                risk[d] = int(row["CiPA"])
    drugs = SMOKE_DRUGS if SMOKE else DRUGS_24
    print(f"[drugs] {len(drugs)} ({'smoke' if SMOKE else 'full'})", flush=True)

    # control (no drug) computed once; <m> = time average over the last control loop (duty cycle, drug independent)
    Q0 = forward_drug(V, tabs, 0.0, 0.0, 0.0)
    q0_last, q0_mid = Q0[-1], Q0[SS_CHECK_LOOP - 1]
    m_arr, _h_arr = eng.forward(V, tabs)
    m_bar = float(m_arr[-len(m_arr) // N_LOOP:].mean())
    print(f"[duty cycle] AP last-loop <m> = {m_bar:.4f}", flush=True)

    res = {}
    for i, d in enumerate(drugs):
        p = drug_params(card3, d)
        entry = dict(risk=risk[d], cmax_nM=cmax[d], **p)
        S, S_mid, S_static, S_eq, S_eq_lab = {}, {}, {}, {}, {}
        for mult in CONC_MULT:
            conc = cmax[d] * mult
            b = hill_block(p["ic50"], p["nh"], conc)
            S_static[str(mult)] = b
            if p["ic50"]:
                S_eq[str(mult)] = float(conc * m_bar / (conc * m_bar + p["ic50"]))
            if p["ic50_lab"]:
                S_eq_lab[str(mult)] = float(conc * m_bar / (conc * m_bar + p["ic50_lab"]))
            if p["degen"]:
                if b is None:
                    S[str(mult)] = None
                    continue
                Qd = forward_drug(V, tabs, 0.0, 0.0, 0.0, hill_clamp=b)
            else:
                Qd = forward_drug(V, tabs, p["kon"], p["koff"], conc)
            S[str(mult)] = float(1.0 - Qd[-1] / q0_last)
            S_mid[str(mult)] = float(1.0 - Qd[SS_CHECK_LOOP - 1] / q0_mid)
        entry["S_transient"] = S
        entry["S_transient_loop%d" % SS_CHECK_LOOP] = S_mid
        entry["S_eq"] = S_eq
        entry["S_eq_锚版"] = S_eq_lab
        entry["S_static_Hill"] = S_static
        s1, s6 = S.get("1"), S_mid.get("1")
        entry["稳态旗"] = (s1 is not None and s6 is not None and abs(s1 - s6) > 0.05)
        res[d] = entry
        tag = p["degen"] or f"kon={p['kon']:.2e} koff={p['koff']:.2e}"
        print(f"  [{i + 1}/{len(drugs)}] {d} (risk {risk[d]} Cmax={cmax[d]}nM): {tag}"
              f"  S_eq@1x={S_eq.get('1')}  transient S@1x={S.get('1')}", flush=True)

    # ---------- criteria (main verdict metric = S_eq@1x, pinned in v2) ----------
    s1map = {d: r["S_eq"]["1"] for d, r in res.items() if r["S_eq"].get("1") is not None}
    rho, pval = (np.nan, np.nan)
    if len(s1map) >= 8:
        rho, pval = stats.spearmanr([s1map[d] for d in s1map],
                                    [risk[d] for d in s1map])
        pval = pval / 2.0 if rho > 0 else 1.0 - pval / 2.0   # one-sided
    auc = auc_high_low(s1map, risk)
    # sensitivity: anchor-IC50 version of S_eq
    s1lab = {d: r["S_eq_锚版"]["1"] for d, r in res.items() if r["S_eq_锚版"].get("1") is not None}
    rho_lab, p_lab = (np.nan, np.nan)
    if len(s1lab) >= 8:
        rho_lab, p_lab = stats.spearmanr([s1lab[d] for d in s1lab], [risk[d] for d in s1lab])
        p_lab = p_lab / 2.0 if rho_lab > 0 else 1.0 - p_lab / 2.0
    auc_lab = auc_high_low(s1lab, risk) if s1lab else None
    # sensitivity: static Hill version
    s1st = {d: r["S_static_Hill"]["1"] for d, r in res.items()
            if r["S_static_Hill"].get("1") is not None}
    rho_st, p_st = (np.nan, np.nan)
    if len(s1st) >= 8:
        rho_st, p_st = stats.spearmanr([s1st[d] for d in s1st], [risk[d] for d in s1st])
        p_st = p_st / 2.0 if rho_st > 0 else 1.0 - p_st / 2.0
    auc_st = auc_high_low(s1st, risk) if s1st else None
    # transient registry
    s1tr = {d: r["S_transient"]["1"] for d, r in res.items()
            if r["S_transient"].get("1") is not None}
    rho_tr, p_tr = (np.nan, np.nan)
    if len(s1tr) >= 8:
        rho_tr, p_tr = stats.spearmanr([s1tr[d] for d in s1tr], [risk[d] for d in s1tr])
        p_tr = p_tr / 2.0 if rho_tr > 0 else 1.0 - p_tr / 2.0
    v3 = [d for d, r in res.items() if (r["S_eq"].get("25") or 0) > 0.5]

    verdict = {
        "合成自检": bool(synth_ok),
        "主判指标": "S_eq@1×Cmax（占空比调整平衡，v2）",
        "m_bar_AP末循环": m_bar,
        "V1_Spearman": {"rho": float(rho), "p_one_side": float(pval),
                        "过": bool(synth_ok and rho >= 0.60 and pval < 0.01)},
        "V2_AUC高低": {"auc": auc, "过": bool(synth_ok and auc is not None and auc >= 0.80)},
        "V3_S_eq25x过半药数": len(v3),
        "敏感性_锚IC50版": {"rho": float(rho_lab), "p_one_side": float(p_lab), "auc": auc_lab},
        "敏感性_静态Hill版": {"rho": float(rho_st), "p_one_side": float(p_st), "auc": auc_st},
        "登记_暂态84拍版": {"rho": float(rho_tr), "p_one_side": float(p_tr)},
        "稳态旗药数": sum(1 for r in res.values() if r["稳态旗"]),
        "静态退化药": [d for d, r in res.items() if r["degen"]],
    }

    print("\n" + "=" * 76, flush=True)
    print(f" V1 (main verdict S_eq@1x): rho={rho:.3f} p={pval:.4f} -> "
          f"{'pass' if verdict['V1_Spearman']['过'] else 'fail'}", flush=True)
    print(f" V2: AUC(high vs low)={auc} -> {'pass' if verdict['V2_AUC高低']['过'] else 'fail'}", flush=True)
    print(f" V3 registry: drugs with S_eq@25x>0.5: {len(v3)}/{len(res)}; steady-state flags {verdict['稳态旗药数']}; "
          f"static degraded {len(verdict['静态退化药'])}", flush=True)
    print(f" sensitivity (anchor IC50): rho={rho_lab:.3f} p={p_lab:.4f} AUC={auc_lab}", flush=True)
    print(f" sensitivity (static Hill): rho={rho_st:.3f} p={p_st:.4f} AUC={auc_st}", flush=True)
    print(f" registry (transient 84 beats): rho={rho_tr:.3f} p={p_tr:.4f}", flush=True)

    sfx = "_冒烟" if SMOKE else ""
    fj = os.path.join(HERE, f"2026-09-14_α模型_药物卡4_AP风险对拍{sfx}_结果.json")
    json.dump({"预注册": "预注册_α模型_药物卡4_AP波形临床风险对拍_2026-09-14.md",
               "判词": verdict, "逐药": res}, open(fj, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  results saved: {fj}", flush=True)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = {2: "#c0392b", 1: "#2980b9", 0: "#27ae60"}
    labels = {2: "high", 1: "medium", 0: "low"}
    for rk in (2, 1, 0):
        ds = [d for d in s1map if risk[d] == rk]
        ys = [s1map[d] for d in ds]
        ax.scatter([rk] * len(ds), ys, c=colors[rk], s=60, alpha=0.85, label=f"{labels[rk]} risk")
        for d, y in zip(ds, ys):
            ax.annotate(d, (rk, y), fontsize=6, alpha=0.7,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_xticks([2, 1, 0])
    ax.set_xticklabels(["high risk", "medium risk", "low risk"])
    ax.set_ylabel("S_eq@1xCmax (equilibrium hERG inhibition)")
    ax.set_title(f"card-4 main verdict: rho={rho:.2f} p={pval:.4f}  AUC={auc if auc else float('nan'):.2f}")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for d in drugs:
        r = res[d]
        xs, ys = [], []
        for mult in CONC_MULT:
            v_ = r["S_eq"].get(str(mult))
            if v_ is not None:
                xs.append(mult)
                ys.append(v_)
        if xs:
            ax.plot(xs, ys, "o-", color=colors[risk[d]], alpha=0.6, lw=1, ms=3)
    ax.set_xscale("log")
    ax.set_xlabel("concentration x free Cmax")
    ax.set_ylabel("S_eq")
    ax.set_title("dose-inhibition curves (red=high blue=medium green=low)")
    ax.grid(alpha=0.3, which="both")
    fpng = os.path.join(HERE, f"2026-09-14_α模型_药物卡4_AP风险对拍{sfx}.png")
    fig.tight_layout()
    fig.savefig(fpng, dpi=140, bbox_inches="tight")
    print(f"  figure saved: {fpng}", flush=True)


if __name__ == "__main__":
    main()
