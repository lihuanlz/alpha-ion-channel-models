# -*- coding: utf-8 -*-
"""
2026-09-14 · alpha-model Stage B · B1 official static arm (24 drugs)
==================================================
B0 engine (verified port reused via importlib) + seven-channel static Hill multipliers (fc interface),
24 drugs (pinned by the card-4 preregistration section 3 list), each continued from the control steady state;
last-beat qNet/APD90 are taken, and the drug/control ratio is checked against the CiPA risk ranking.

Criteria (preregistration section 5 + section 11.7 v5 symbols pinned before the run):
  V1 main: Spearman rho(qNet_ratio, risk order 0/1/2) <= -0.60 and p < 0.01
           (official direction: high risk -> qNet decrease, Dutta 2017 Fig 3);
  V2 main: AUC(qNet suppression = 1-qNet_ratio, high vs low) >= 0.80;
  registry: APD90_ratio same-convention metrics; intermediate vs low AUC; per-drug steady/EAD flag counts;
        control reproduction check |APD90 - 252.118| < 0.5 ms (v4 section 11.6.2).

Discipline: this side runs only ast.parse + SMOKE=1 smoke (control 60 beats + 3 drugs x 60 beats);
the full run (1000 + 24x500 beats, about 95 min) is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_StageB_B1_官方静态臂_24药.py' --wdir
Output: _结果.json/.csv/.png next to this script (smoke carries the _冒烟 suffix) + StageB_B0稳态_CL1000.json (full run).
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

SMOKE = os.environ.get("SMOKE", "0") == "1"
N_CTRL = int(os.environ.get("SMOKE_BEATS", "60")) if SMOKE else int(os.environ.get("NBEATS_CTRL", "1000"))
N_DRUG = int(os.environ.get("SMOKE_BEATS", "60")) if SMOKE else int(os.environ.get("NBEATS_DRUG", "500"))
CL = 1000.0
DT_REC = 0.1
B0_FORMAL_APD90 = 252.1181387725961   # B0 full-run last beat (on record 2026-09-14 14:51)

# card-4 preregistration section 3 list (24 drugs, pinned)
DRUGS_HIGH = ["azimilide", "bepridil", "disopyramide", "dofetilide", "ibutilide", "sotalol", "vandetanib"]
DRUGS_MID = ["astemizole", "chlorpromazine", "cisapride", "clarithromycin", "clozapine",
             "domperidone", "droperidol", "ondansetron", "pimozide", "risperidone", "terfenadine"]
DRUGS_LOW = ["diltiazem", "metoprolol", "mexiletine", "ranolazine", "tamoxifen", "verapamil"]
DRUGS_24 = DRUGS_HIGH + DRUGS_MID + DRUGS_LOW
SMOKE_DRUGS = ["dofetilide", "cisapride", "verapamil"]   # v4 §11.6.1：moxifloxacin→cisapride

# channel -> (csv IC50 column, csv h column, pars index)  (pinned in v4 section 11.6.6)
CHANNELS = [("hERG", "hERGIC50", "hERGh", 1),
            ("LateNa", "Late_sodiumIC50", "Late_sodiumh", 2),
            ("PeakNa", "Peak_sodiumIC50", "Peak_sodiumh", 3),
            ("IKs", "IKsIC50", "IKsh", 4),
            ("IK1", "IK1IC50", "IK1h", 5),
            ("CaL", "CaLIC50", "CaLh", 6),
            ("Ito", "ItoIC50", "Itoh", 7)]


def load_b0():
    path = os.path.join(BASE, "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")
    spec = importlib.util.spec_from_file_location("b0_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)     # __main__ guard, main() is not triggered
    return mod


def load_cipa(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["drug"].strip()] = row
    return rows


def drug_params(P_ctrl, row):
    """Returns (P_drug, mult dict, C_nM). fc_eff = 1/mult (all control fc = 1)."""
    C = float(row["therapeutic"])
    P = list(P_ctrl)
    mults = {}
    for nm, cIC, ch, idx in CHANNELS:
        IC50 = float(row[cIC])
        h = float(row[ch])
        if IC50 <= 0.0 or h <= 0.0:
            mults[nm] = 1.0
            continue
        mult = 1.0 / (1.0 + (C / IC50) ** h)
        mults[nm] = mult
        P[idx] = 1.0 / mult          # fc_eff = fc/mult, fc=1
    return P, mults, C


def run_beats(b0, P, y0, nbeats, keep_last_trace=True, label=""):
    """Pace nbeats from y0; return beat-by-beat APD90/qNet and the last-beat trace."""
    from scipy.integrate import solve_ivp
    rhs = b0.build_model(P)
    f = lambda t, y: rhs(t, y)[0]
    t_rec = np.arange(0.0, CL + 1e-9, DT_REC)
    y = np.array(y0, dtype=float)
    apd = np.full(nbeats, np.nan)
    qnet = np.full(nbeats, np.nan)
    v_last = None
    t_run = time.time()
    for b in range(nbeats):
        q0 = y[49]
        sol = solve_ivp(f, (0.0, CL), y, method="LSODA",
                        t_eval=t_rec, rtol=1e-7, atol=1e-9)
        if not sol.success:
            break
        v = sol.y[0]
        y = sol.y[:, -1].copy()
        apd[b], _, _, _ = b0.apd90(t_rec, v)
        qnet[b] = (y[49] - q0) * 1e-3
        if b == nbeats - 1 and keep_last_trace:
            v_last = v.copy()
        if (b + 1) % 20 == 0 or b == 0:
            el = time.time() - t_run
            eta = el / (b + 1) * (nbeats - b - 1)
            print(f"    {label} beat {b + 1}/{nbeats}  APD90={apd[b]:7.2f}ms  "
                  f"qNet={qnet[b]:.4f}  took {el:.0f}s ETA {eta:.0f}s", flush=True)
    return apd, qnet, v_last, y


def steady_flag(apd):
    ok = apd[~np.isnan(apd)]
    if len(ok) < 10:
        return float("nan"), False
    r = float(np.max(ok[-10:]) - np.min(ok[-10:]))
    return r, bool(r < 0.5)


def auc_pos(gt, scores_pos, scores_neg):
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 * ties."""
    n_g = 0.0
    n_t = 0
    for a in scores_pos:
        for b in scores_neg:
            if a > b:
                n_g += 1.0
            elif a == b:
                n_g += 0.5
            n_t += 1
    return n_g / n_t if n_t else float("nan")


def main():
    t_start = time.time()
    print("=" * 74, flush=True)
    print(" alpha-model Stage B · B1 official static arm (Markov hERG + seven-channel static Hill, 24 drugs)", flush=True)
    print(f" mode: {'smoke SMOKE (control ' + str(N_CTRL) + ' beats + ' + str(len(SMOKE_DRUGS)) + ' drugs x ' + str(N_DRUG) + ' beats)' if SMOKE else 'full (control ' + str(N_CTRL) + ' beats + 24 drugs x ' + str(N_DRUG) + ' beats)'}", flush=True)
    print("=" * 74, flush=True)

    b0 = load_b0()
    P_ctrl = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    y_init = b0.load_named_values(b0.STATES_FILE, b0.STATE_NAMES) + [0.0]

    rows = load_cipa(CIPA_CSV)
    drug_list = SMOKE_DRUGS if SMOKE else DRUGS_24
    missing = [d for d in drug_list if d not in rows]
    if missing:
        raise RuntimeError(f"newCiPA.csv missing drugs: {missing}")
    if not SMOKE:
        n2 = sum(1 for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 2)
        n1 = sum(1 for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 1)
        n0 = sum(1 for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 0)
        assert (n2, n1, n0) == (7, 11, 6), f"label counts {n2}/{n1}/{n0} != 7/11/6"
        assert set(DRUGS_HIGH) == {d for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 2}
        assert set(DRUGS_LOW) == {d for d in DRUGS_24 if int(float(rows[d]["CiPA"])) == 0}
        print(f"[list] 24 drugs in place; labels high 7 / intermediate 11 / low 6 match card-4 section 3", flush=True)

    # ---------- control ----------
    print(f"\n[control] {N_CTRL} beats ...", flush=True)
    t0 = time.time()
    apd_c, qnet_c, v_c, y_ss = run_beats(b0, P_ctrl, y_init, N_CTRL, label="control")
    nd_c = int(np.sum(~np.isnan(apd_c)))
    APD_ctrl = float(apd_c[nd_c - 1])
    qNet_ctrl = float(qnet_c[nd_c - 1])
    r_c, ok_c = steady_flag(apd_c)
    repro = abs(APD_ctrl - B0_FORMAL_APD90)
    print(f"  control last-beat APD90={APD_ctrl:.3f}ms qNet={qNet_ctrl:.4f}uC/uF  "
          f"steady range={r_c:.3f}ms  B0 reproduction diff={repro:.4f}ms  took {time.time() - t0:.0f}s", flush=True)
    repro_pass = True
    if not SMOKE:
        repro_pass = bool(repro < 0.5)
        print(f"  reproduction check |diff| < 0.5ms -> {'pass' if repro_pass else 'fail - B1 stops and registers'}", flush=True)
        if not repro_pass:
            print("B1 aborted: control reproduction check failed.", flush=True)
            return
        fss = os.path.join(BASE, "StageB_B0稳态_CL1000.json")
        with open(fss, "w", encoding="utf-8") as fh:
            json.dump({"states50": [float(x) for x in y_ss],
                       "APD90_ms": APD_ctrl, "qNet_uC_uF": qNet_ctrl,
                       "note": "B1 control 1000-beat final state, reused by B2 (v4 section 11.6.3)"}, fh,
                      ensure_ascii=False, indent=1)
        print(f"  steady state saved: {fss}", flush=True)

    # ---------- per drug ----------
    results = []
    traces = {"control": v_c}
    for i, name in enumerate(drug_list):
        row = rows[name]
        P_d, mults, C = drug_params(P_ctrl, row)
        t0 = time.time()
        apd_d, qnet_d, v_d, _ = run_beats(b0, P_d, y_ss, N_DRUG, label=name)
        nd = int(np.sum(~np.isnan(apd_d)))
        if nd == 0:
            APD_d = float("nan"); qNet_d = float("nan"); rng = float("nan"); okd = False
            note = "integration failed or no repolarization"
        else:
            APD_d = float(apd_d[nd - 1])
            qNet_d = float(qnet_d[nd - 1])
            rng, okd = steady_flag(apd_d)
            note = "" if nd == N_DRUG else f"only {nd}/{N_DRUG} beats valid"
        ead = bool(nd < N_DRUG or math.isnan(APD_d))
        rec = {"drug": name, "CiPA": int(float(row["CiPA"])), "C_nM": C,
               "mult": mults, "APD90_ms": APD_d, "qNet_uC_uF": qNet_d,
               "APD90_ratio": APD_d / APD_ctrl if not math.isnan(APD_d) else float("nan"),
               "qNet_ratio": qNet_d / qNet_ctrl if not math.isnan(qNet_d) else float("nan"),
               "steady_range_ms": rng, "steady_ok": okd, "EAD_or_fail": ead, "note": note}
        results.append(rec)
        traces[name] = v_d
        m5 = "/".join(f"{k}:{v:.3f}" for k, v in mults.items() if v < 0.999)
        print(f"  [{i + 1}/{len(drug_list)}] {name:15s} CiPA={rec['CiPA']} C={C:9.3f}nM  "
              f"APD90={APD_d:8.2f}({rec['APD90_ratio']:.3f}×)  qNet={qNet_d:.4f}({rec['qNet_ratio']:.3f}×)  "
              f"steady {rng:.2f}ms{'v' if okd else 'x'}  {note}  {time.time() - t0:.0f}s  [{m5}]", flush=True)

    # ---------- criteria (v5 symbols pinned; smoke = same code path rehearsal, no verdict) ----------
    from scipy.stats import spearmanr
    risk = np.array([r["CiPA"] for r in results], dtype=float)
    qr = np.array([r["qNet_ratio"] for r in results], dtype=float)
    ar = np.array([r["APD90_ratio"] for r in results], dtype=float)
    okq = ~np.isnan(qr)
    oka = ~np.isnan(ar)
    rho_q, p_q = spearmanr(risk[okq], qr[okq])
    rho_a, p_a = spearmanr(risk[oka], ar[oka])
    sup = 1.0 - qr                                        # qNet suppression: larger = deeper block
    hi_s = sup[(risk == 2) & okq]; lo_s = sup[(risk == 0) & okq]; mid_s = sup[(risk == 1) & okq]
    hi_a = ar[(risk == 2) & oka];   lo_a = ar[(risk == 0) & oka];   mid_a = ar[(risk == 1) & oka]
    auc_q = auc_pos(None, hi_s, lo_s)                     # P(suppression_high > suppression_low)
    auc_a = auc_pos(None, hi_a, lo_a)
    auc_q_ml = auc_pos(None, mid_s, lo_s)
    auc_a_ml = auc_pos(None, mid_a, lo_a)
    V1 = bool(rho_q <= -0.60 and p_q < 0.01)
    V2 = bool(auc_q >= 0.80)
    verdict = {"V1_rho_qNet": float(rho_q), "V1_p": float(p_q), "V1_pass": V1,
               "V2_AUC_qNetSupp_hi_lo": float(auc_q), "V2_pass": V2,
               "reg_APD90_rho": float(rho_a), "reg_APD90_p": float(p_a),
               "reg_AUC_APD90_hi_lo": float(auc_a),
               "reg_AUC_qNetSupp_mid_lo": float(auc_q_ml), "reg_AUC_APD90_mid_lo": float(auc_a_ml),
               "reg_n_EAD_or_fail": int(sum(r["EAD_or_fail"] for r in results)),
               "reg_n_steady_fail": int(sum(not r["steady_ok"] for r in results)),
               "B1_overall_pass": (None if SMOKE else bool(V1 and V2)),
               "rehearsal": SMOKE}
    lab = "criterion-path rehearsal (smoke, not a verdict)" if SMOKE else "criteria"
    print("\n" + "=" * 74, flush=True)
    print(f" [{lab}] V1 Spearman rho(qNet_ratio, risk order) = {rho_q:.3f} (p={p_q:.2e}) "
          f"<= -0.60 and p<0.01 -> {'pass' if V1 else 'fail'}", flush=True)
    print(f" [{lab}] V2 AUC(qNet suppression, high vs low) = {auc_q:.3f} >=0.80 -> {'pass' if V2 else 'fail'}", flush=True)
    print(f" [{lab}·registry] APD90_ratio: rho={rho_a:.3f}(p={p_a:.2e}) AUC high-low={auc_a:.3f}; "
          f"intermediate vs low AUC qNet suppression={auc_q_ml:.3f} APD90={auc_a_ml:.3f}; "
          f"EAD/failure {verdict['reg_n_EAD_or_fail']} steady-not-reached {verdict['reg_n_steady_fail']}", flush=True)
    if not SMOKE:
        print(f" B1 overall verdict: {'PASS' if verdict['B1_overall_pass'] else 'FAIL'}", flush=True)
    print("=" * 74, flush=True)

    # ---------- save ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(BASE, f"2026-09-14_α模型_StageB_B1_官方静态臂_24药{tag}.json")
    out = {"meta": {"smoke": SMOKE, "n_ctrl": N_CTRL, "n_drug": N_DRUG,
                    "control_APD90_ms": APD_ctrl, "control_qNet_uC_uF": qNet_ctrl,
                    "control_repro_diff_ms": repro,
                    "runtime_s": time.time() - t_start},
           "results": results, "verdict": verdict}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    fcsv = os.path.join(BASE, f"2026-09-14_α模型_StageB_B1_官方静态臂_24药{tag}.csv")
    with open(fcsv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drug", "CiPA", "C_nM"] + [f"mult_{c[0]}" for c in CHANNELS]
                   + ["APD90_ms", "APD90_ratio", "qNet_uC_uF", "qNet_ratio",
                      "steady_range_ms", "steady_ok", "EAD_or_fail", "note"])
        for r in results:
            w.writerow([r["drug"], r["CiPA"], r["C_nM"]] + [r["mult"][c[0]] for c in CHANNELS]
                       + [r["APD90_ms"], r["APD90_ratio"], r["qNet_uC_uF"], r["qNet_ratio"],
                          r["steady_range_ms"], r["steady_ok"], r["EAD_or_fail"], r["note"]])
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
    ttl = f"B1 official static arm · {'smoke' if SMOKE else '24 drugs'}"
    if verdict:
        ttl += f" · V1 ρ={verdict['V1_rho_qNet']:.3f}{'✓' if verdict['V1_pass'] else '×'}"
        ttl += f" V2 AUC={verdict['V2_AUC_qNetSupp_hi_lo']:.3f}{'✓' if verdict['V2_pass'] else '×'}"
        if SMOKE:
            ttl += " (rehearsal)"
    fig.suptitle(ttl, fontsize=13)

    for ax, key, lab in [(axes[0, 0], "qNet_ratio", "qNet ratio (drug/control)"),
                         (axes[0, 1], "APD90_ratio", "APD90 ratio (drug/control)")]:
        for cls, cname, col in [(0, "low", "tab:green"), (1, "intermediate", "tab:orange"), (2, "high", "tab:red")]:
            xs, ys = [], []
            for r in results:
                if r["CiPA"] == cls and not math.isnan(r[key]):
                    xs.append(cls + (np.random.rand() - 0.5) * 0.24)
                    ys.append(r[key])
            ax.scatter(xs, ys, c=col, s=34, zorder=3, label=cname)
            if ys:
                ax.hlines(np.median(ys), cls - 0.3, cls + 0.3, color=col, lw=2)
            for x, y, r in zip(xs, ys, [r for r in results if r["CiPA"] == cls and not math.isnan(r[key])]):
                ax.annotate(r["drug"], (x, y), fontsize=6, alpha=0.75,
                            xytext=(3, 3), textcoords="offset points")
        ax.axhline(1.0, color="k", lw=0.7, ls="--")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["low", "intermediate", "high"])
        ax.set_ylabel(lab); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    hi = [1.0 - r["qNet_ratio"] for r in results if r["CiPA"] == 2 and not math.isnan(r["qNet_ratio"])]
    lo = [1.0 - r["qNet_ratio"] for r in results if r["CiPA"] == 0 and not math.isnan(r["qNet_ratio"])]
    allv = sorted(set(hi + lo))
    if hi and lo and len(allv) > 1:
        tprs, fprs = [1.0], [1.0]
        for th in sorted(allv, reverse=True):
            tprs.append(float(np.mean([s >= th for s in hi])))
            fprs.append(float(np.mean([s >= th for s in lo])))
        tprs.append(0.0); fprs.append(0.0)
        ax.plot(fprs, tprs, "o-", ms=3,
                label=f"qNet suppression AUC={verdict['V2_AUC_qNetSupp_hi_lo']:.3f}")
        ax.legend(fontsize=9)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR (low misjudged as high)"); ax.set_ylabel("TPR (high detected)")
    ax.set_title("ROC high vs low (qNet suppression = 1 - qNet_ratio)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    t_rec = np.arange(0.0, CL + 1e-9, DT_REC)
    for nm, col in [("control", "k")] + [(d, None) for d in SMOKE_DRUGS if d in traces]:
        vv = traces.get(nm)
        if vv is not None:
            ax.plot(t_rec, vv, lw=1.0, label=nm, color=col)
    ax.set_title("last-beat AP: control vs smoke three drugs")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("v (mV)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fpng = os.path.join(BASE, f"2026-09-14_α模型_StageB_B1_官方静态臂_24药{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" figure saved: {fpng}", flush=True)

    if SMOKE:
        print("\n[smoke done] full-run command (Spyder, about 95 min):\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_StageB_B1_官方静态臂_24药.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
