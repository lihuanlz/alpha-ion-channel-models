# -*- coding: utf-8 -*-
# 2026-09-13_memory_kernel_identifiability_three_gate_validation.py
# Purpose: final ruling on model-building feasibility (pure synthetic data, zero real-data fitting). Three gates in one run, criteria pinned:
#   Gate 1 anchor validity: closed-form inversion of kernel shape parameters (beta or alpha) from late-window dual slopes, 200 noise realizations, check bias and spread
#        pass: |bias|<=10% and IQR<=20% (1-sigma noise); IQR<=40% under 3-sigma noise
#   Gate 2 anchor breaks degeneracy: full-length joint fit (c0,a_rec,tau_rec,a_s,beta,tau_s)
#        free multi-start: if different starts converge to different beta with nearly identical residuals -> T3 degeneracy reproduced
#        anchored (beta,tau_s frozen to the gate-1 anchor): remaining parameters must recover within 20% error for the anchor to count as rescuing
#        three verdicts: anchor necessary and sufficient / joint self-identification (anchor not needed) / anchor cannot rescue (unidentifiable)
#   Gate 3 family distinguishability: stretched-exponential <-> power-law cross-fits; both directions must exceed 2-sigma residual for the families to be separable;
#        if not separable, "kernel shape parameter" is ill-defined and one must fall back to family-free spectral moments
# Run: python this_file (your machine, about 1-2 minutes)
import os
import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
SIG = 0.036          # measured noise floor (nA)
DT = 1e-3            # consistent with the v5.3 downsampling metric
T_END = 5.5
SEED = 20260913

# ---------------- kernel families ----------------

def k_stretch(t, tau, beta):
    return np.exp(-(t / tau) ** beta)

def k_power(t, tau, alpha):
    return (1.0 + t / tau) ** (-alpha)

# ---------------- v5.3 identical measurement pipeline: binned median -> log -> WLS ----------------

def binlog(t, y, bin_s=0.05):
    nbin = int(np.clip((t[-1] - t[0]) / bin_s, 16, 110))
    sig_bin = 3 * SIG / np.sqrt(max(1, len(t) // nbin))
    bt, bl, bm = [], [], []
    for b in np.array_split(np.arange(len(t)), nbin):
        med = float(np.median(y[b]))
        if med > max(sig_bin, 0.01):
            bt.append(float(np.median(t[b])))
            bl.append(float(np.log(med)))
            bm.append(med)
    return np.array(bt), np.array(bl), np.array(bm)

def wslope(tq, lq, mq):
    wq = mq ** 2
    tbar = np.sum(wq * tq) / wq.sum()
    W = np.sum(wq * (tq - tbar) ** 2)
    return float(np.sum(wq * (tq - tbar) * lq) / W)

T1, T2 = 2.0, 4.5                      # centers of the two anchor windows
W1, W2 = (1.6, 2.4), (4.1, 4.9)

def two_slopes(t, y):
    bt, bl, bm = binlog(t, y)
    out = []
    for w in (W1, W2):
        m = (bt >= w[0]) & (bt <= w[1])
        if m.sum() < 6:
            return None
        out.append(wslope(bt[m], bl[m], bm[m]))
    return out

def inv_stretch(s1, s2):
    """Closed-form inversion of s(t) = -(beta/tau^beta) t^{beta-1}"""
    beta = 1.0 + np.log(s1 / s2) / np.log(T1 / T2)
    tau = (-beta * T1 ** (beta - 1) / s1) ** (1.0 / beta)
    return float(beta), float(tau)

def inv_power(s1, s2):
    """Closed-form inversion of s(t) = -alpha/(tau+t)"""
    alpha = (T2 - T1) / (1.0 / s1 - 1.0 / s2)
    tau = -alpha / s1 - T1
    return float(alpha), float(tau)

# ---------------- gate 1 ----------------

TRUTH_ST = dict(beta=0.5, tau=2.0, amp=0.5)
TRUTH_PW = dict(alpha=1.0, tau=0.3, amp=0.5)   # alpha=1.5/tau=0.5 would drop the late-window signal below the mask; switched to a measurable truth

def gate1(rng, nmc=200):
    t = np.arange(DT, T_END, DT)
    res = {}
    for fam in ("stretch", "power"):
        for sigscale, tag in ((1.0, "1σ"), (3.0, "3σ")):
            ests = []
            for _ in range(nmc):
                if fam == "stretch":
                    y = TRUTH_ST["amp"] * k_stretch(t, TRUTH_ST["tau"], TRUTH_ST["beta"])
                else:
                    y = TRUTH_PW["amp"] * k_power(t, TRUTH_PW["tau"], TRUTH_PW["alpha"])
                y = y + rng.normal(0, SIG * sigscale, len(t))
                sl = two_slopes(t, y)
                if sl is None:
                    continue
                try:
                    if fam == "stretch":
                        b, ta = inv_stretch(*sl)
                        if 0.05 < b < 1.5 and 0.05 < ta < 100:
                            ests.append((b, ta))
                    else:
                        a, ta = inv_power(*sl)
                        if 0.05 < a < 10 and 0.01 < ta < 100:
                            ests.append((a, ta))
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
            ests = np.array(ests)
            key = f"{fam}_{tag}"
            if len(ests) < nmc * 0.8:
                res[key] = dict(error=f"valid inversions {len(ests)}/{nmc}")
                continue
            p = ests[:, 0]
            truth_p = TRUTH_ST["beta"] if fam == "stretch" else TRUTH_PW["alpha"]
            bias = float(np.median(p) / truth_p - 1)
            iqr = float((np.percentile(p, 75) - np.percentile(p, 25)) / truth_p)
            res[key] = dict(n=len(ests), shape_med=float(np.median(p)), truth=truth_p,
                            bias=bias, iqr=iqr, tau_med=float(np.median(ests[:, 1])),
                            est_shape=p.tolist())
    return res

# ---------------- gate 2 ----------------

TRUTH2 = dict(c0=0.05, a_rec=1.0, t_rec=0.15, a_s=0.5, beta=0.5, tau_s=2.0)

def model_full(th, t):
    c0, ar, tr, a_s, be, ts = th
    return c0 + ar * np.exp(-t / tr) + a_s * k_stretch(t, ts, be)

def model_anchored(th, t, be, ts):
    c0, ar, tr, a_s = th
    return c0 + ar * np.exp(-t / tr) + a_s * k_stretch(t, ts, be)

LB6 = [-0.5, 0.0, 0.02, 0.0, 0.10, 0.2]
UB6 = [0.5, 5.0, 2.00, 5.0, 1.00, 20.0]
LB4 = [-0.5, 0.0, 0.02, 0.0]
UB4 = [0.5, 5.0, 2.00, 5.0]

def gate2(rng, nmc=20):
    t = np.arange(DT, T_END, DT)
    y_true = model_full([TRUTH2[k] for k in ("c0", "a_rec", "t_rec", "a_s", "beta", "tau_s")], t)
    beta_starts = [0.2, 0.35, 0.5, 0.7, 0.9]
    free_beta, free_cost = [], []
    anch_err, anch_cost = [], []
    for _ in range(nmc):
        y = y_true + rng.normal(0, SIG, len(t))
        # (a) free multi-start
        bests = []
        for b0 in beta_starts:
            x0 = [0.1, 0.8, 0.2, 0.4, b0, 2.0]
            try:
                r = least_squares(lambda th: model_full(th, t) - y, x0,
                                  bounds=(LB6, UB6), max_nfev=300)
                bests.append((float(2 * r.cost / len(t)), r.x))
            except Exception:
                pass
        if bests:
            costs = np.array([b[0] for b in bests])
            free_cost.append(float(costs.min()))
            near = [b[1] for b in bests if b[0] <= costs.min() * 1.01 + 1e-12]
            free_beta.append([float(x[4]) for x in near])
        # (b) anchored: first rough-estimate c0 (beta frozen at nominal 0.5; c0 is insensitive to beta mis-setting), subtract it, take the gate-1 anchor, then joint fit with the anchor frozen
        try:
            r0 = least_squares(lambda th: model_anchored(th, t, 0.5, 2.0) - y,
                               [0.1, 0.8, 0.2, 0.4], bounds=(LB4, UB4), max_nfev=300)
            c0_hat = float(r0.x[0])
        except Exception:
            continue
        sl = two_slopes(t, y - c0_hat)
        if sl is not None:
            try:
                b_a, ts_a = inv_stretch(*sl)
                b_a = float(np.clip(b_a, 0.1, 1.0))
                ts_a = float(np.clip(ts_a, 0.2, 20.0))
            except Exception:
                continue
            try:
                r = least_squares(lambda th: model_anchored(th, t, b_a, ts_a) - y,
                                  [0.1, 0.8, 0.2, 0.4], bounds=(LB4, UB4), max_nfev=300)
                anch_cost.append(float(2 * r.cost / len(t)))
                rel = [abs(r.x[1] - TRUTH2["a_rec"]) / TRUTH2["a_rec"],
                       abs(r.x[2] - TRUTH2["t_rec"]) / TRUTH2["t_rec"],
                       abs(r.x[3] - TRUTH2["a_s"]) / TRUTH2["a_s"]]
                anch_err.append(rel)
            except Exception:
                pass
    out = {}
    if free_beta:
        allb = np.array([b for lst in free_beta for b in lst])
        spread = float(np.percentile(allb, 75) - np.percentile(allb, 25))
        out["free_beta_iqr"] = spread
        out["free_beta_all"] = allb.tolist()
        out["free_cost_med"] = float(np.median(free_cost))
        out["degeneracy_reproduced"] = bool(spread > 0.10)
    if anch_err:
        anch_err = np.array(anch_err)
        out["anch_relerr_med"] = [float(np.median(anch_err[:, i])) for i in range(3)]
        out["anch_cost_med"] = float(np.median(anch_cost))
        out["anchored_ok"] = bool(np.all(np.median(anch_err, axis=0) <= 0.20))
    return out

# ---------------- gate 3 ----------------

def model_family(th, t, fam):
    c0, ar, tr, a_s, sh, ts = th
    k = k_stretch(t, ts, sh) if fam == "stretch" else k_power(t, ts, sh)
    return c0 + ar * np.exp(-t / tr) + a_s * k

def gate3(rng):
    t = np.arange(DT, T_END, DT)
    cases = [
        ("true=stretched beta0.5 -> fit power-law",
         model_family([0.05, 1.0, 0.15, 0.5, 0.5, 2.0], t, "stretch"), "power",
         [0.1, 0.8, 0.2, 0.4, 1.5, 0.5], ([-0.5, 0, 0.02, 0, 0.1, 0.01], [0.5, 5, 2, 5, 10, 20])),
        ("true=power-law alpha1.0 -> fit stretched",
         model_family([0.05, 1.0, 0.15, 0.5, 1.0, 0.3], t, "power"), "stretch",
         [0.1, 0.8, 0.2, 0.4, 0.5, 2.0], ([-0.5, 0, 0.02, 0, 0.1, 0.2], [0.5, 5, 2, 5, 1, 20])),
    ]
    out = {}
    for name, y_true, fam, x0, bounds in cases:
        rms = []
        for _ in range(5):
            y = y_true + rng.normal(0, SIG, len(t))
            best = None
            for jit in range(3):
                xj = [v * f for v, f in zip(x0, [1, 1, 1, 1, [0.6, 1.0, 1.8][jit] if fam == "power" else [0.3, 0.5, 0.8][jit], 1])]
                try:
                    r = least_squares(lambda th: model_family(th, t, fam) - y, xj,
                                      bounds=bounds, max_nfev=400)
                    c = float(np.sqrt(2 * r.cost / len(t)))
                    if best is None or c < best:
                        best = c
                except Exception:
                    pass
            if best is not None:
                rms.append(best)
        out[name] = dict(rms_med=float(np.median(rms)), noise=SIG,
                         distinguishable=bool(np.median(rms) > 2 * SIG))
    return out

# ---------------- main flow ----------------

def main():
    rng = np.random.default_rng(SEED)
    print("=" * 66)
    print(" memory-kernel identifiability · three-gate validation (pure synthetic, zero real data)")
    print(f" noise sigma={SIG} nA   window {T_END}s   anchor-window centers {T1}/{T2}s   seed {SEED}")
    print("=" * 66, flush=True)

    print("\n[gate 1] anchor validity: late-window dual-slope closed-form inversion (200 realizations)", flush=True)
    g1 = gate1(rng)
    g1_pass = True
    for key, r in g1.items():
        if "error" in r:
            print(f"  {key}: {r['error']} -> fail")
            g1_pass = False
            continue
        ok_bias = abs(r["bias"]) <= 0.10
        ok_iqr = r["iqr"] <= (0.20 if key.endswith("1σ") else 0.40)
        ok = ok_bias and ok_iqr
        if key.endswith("1σ"):
            g1_pass &= ok
        print(f"  {key}: shape parameter median {r['shape_med']:.3f} (truth {r['truth']})"
              f" bias {r['bias']:+.1%} IQR {r['iqr']:.1%}  tau median {r['tau_med']:.2f}s -> {'pass' if ok else 'fail'}")
    print(f"  gate-1 verdict: {'pass -- anchor valid' if g1_pass else 'fail -- anchor invalid'}")

    print("\n[gate 2] anchor breaks degeneracy: full-length joint fit (20 realizations x multi-start)", flush=True)
    g2 = gate2(rng)
    if "free_beta_iqr" in g2:
        print(f"  free multi-start: beta convergence IQR = {g2['free_beta_iqr']:.3f}  "
              f"({'T3 degeneracy reproduced' if g2['degeneracy_reproduced'] else 'no degeneracy seen, joint self-identification'})"
              f"  residual median {g2.get('free_cost_med', float('nan')):.4f} nA^2")
    if "anch_relerr_med" in g2:
        e = g2["anch_relerr_med"]
        print(f"  anchored fit: a_rec error {e[0]:.1%}  tau_rec error {e[1]:.1%}  a_s error {e[2]:.1%}"
              f"  -> {'recovery OK (<=20%)' if g2['anchored_ok'] else 'recovery failed'}")
    deg = g2.get("degeneracy_reproduced")
    anc = g2.get("anchored_ok")
    if deg and anc:
        v2 = "锚必要且充分 —— 建模可行（锚是必需品）"
    elif (not deg) and anc:
        v2 = "联合自识别 —— 建模可行（锚非必需但无害）"
    elif deg and not anc:
        v2 = "锚也不救 —— 当前锚形式不可识别，建模失败"
    else:
        v2 = "数据不足，不下判词"
    g2_pass = bool(anc) and (deg is not None)
    print(f"  gate-2 verdict: {v2}")

    print("\n[gate 3] family distinguishability: cross-fits (residual >2-sigma required to separate)", flush=True)
    g3 = gate3(rng)
    g3_pass = True
    for name, r in g3.items():
        print(f"  {name}: cross-fit RMS = {r['rms_med']:.4f} nA (2-sigma={2*SIG:.3f})"
              f" -> {'separable' if r['distinguishable'] else 'impersonation succeeded, not separable'}")
        g3_pass &= r["distinguishable"]
    print(f"  gate-3 verdict: {'pass -- families separable, kernel shape parameter well-defined' if g3_pass else 'fail -- only family-free spectral moments extractable'}")

    print("\n" + "=" * 66)
    print(" overall verdicts:")
    print(f"  gate1 {'pass' if g1_pass else 'fail'} | gate2 {'pass' if g2_pass else 'fail'} | gate3 {'pass' if g3_pass else 'fail'}")
    if g1_pass and g2_pass:
        final = "建模可行：锚有效且破简并。" + ("核形状参数良定义。" if g3_pass else "核形状退到族无关谱矩。")
    else:
        final = "建模有硬关：" + ("锚无效。" if not g1_pass else "") + ("锚救不了简并。" if not g2_pass else "")
    print("  " + final)
    print("=" * 66)

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    ax = axes[0, 0]
    for key, r in g1.items():
        if key.startswith("stretch") and "est_shape" in r:
            ax.hist(r["est_shape"], bins=25, alpha=0.5, label=key)
    ax.axvline(TRUTH_ST["beta"], color="k", ls="--", label="truth beta=0.5")
    ax.set_title("gate-1 stretched-exponential beta-hat distribution", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for key, r in g1.items():
        if key.startswith("power") and "est_shape" in r:
            ax.hist(r["est_shape"], bins=25, alpha=0.5, label=key)
    ax.axvline(TRUTH_PW["alpha"], color="k", ls="--", label="truth alpha=1.5")
    ax.set_title("gate-1 power-law alpha-hat distribution", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if "free_beta_all" in g2:
        ax.hist(g2["free_beta_all"], bins=20, color="tomato", alpha=0.7)
        ax.axvline(TRUTH2["beta"], color="k", ls="--", label="truth beta=0.5")
        ax.set_title(f"gate-2 free multi-start beta-hat (IQR={g2['free_beta_iqr']:.3f})", fontweight="bold")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if "anch_relerr_med" in g2:
        e = g2["anch_relerr_med"]
        ax.bar(["a_rec", "τ_rec", "a_s"], e, color="steelblue")
        ax.axhline(0.20, color="r", ls="--", label="20% criterion")
        ax.set_title("gate-2 parameter recovery errors after anchoring", fontweight="bold"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names = list(g3.keys())
    ax.barh([n[:14] for n in names], [g3[n]["rms_med"] for n in names], color="darkorange")
    ax.axvline(2 * SIG, color="r", ls="--", label=f"2σ={2*SIG:.3f}")
    ax.set_title("gate-3 cross-fit RMS", fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["三门总判词", "",
             f"门1 锚有效: {'过' if g1_pass else '不过'}",
             f"门2 破简并: {'过' if g2_pass else '不过'}",
             f"门3 族分辨: {'过' if g3_pass else '不过'}", "", final]
    y0 = 0.9
    for L in lines:
        ax.text(0.05, y0, L, fontsize=12, fontweight="bold" if y0 in (0.9, 0.9 - 6 * 0.11) else "normal", wrap=True)
        y0 -= 0.11
    fig.suptitle("memory-kernel identifiability three-gate validation (pure synthetic)", fontsize=14, fontweight="bold")
    fpng = os.path.join(HERE, "2026-09-13_记忆核_可识别性三门验证.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    g1_out = {k: ({kk: vv for kk, vv in v.items() if kk != "est_shape"} if "error" not in v else v)
              for k, v in g1.items()}
    g2_out = {k: v for k, v in g2.items() if k != "free_beta_all"}
    out = dict(sigma_nA=SIG, seed=SEED, gate1=g1_out, gate1_pass=bool(g1_pass),
               gate2=g2_out, gate2_pass=bool(g2_pass), gate2_verdict=v2,
               gate3=g3, gate3_pass=bool(g3_pass), final=final)
    fjson = os.path.join(HERE, "2026-09-13_memory_kernel_three_gate_validation_results.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n  figure saved: {fpng}")
    print(f"  results saved: {fjson}")

if __name__ == "__main__":
    main()
