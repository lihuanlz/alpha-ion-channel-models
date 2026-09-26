# lei211_J1J2_extraction.py  (v2: re-leak-correction + J4 E_rev first + DF-corrected h_ss)
# criteria source: pre-registered verdict card Lei211 alpha population 2026-09-21 (frozen + amendments A1/A2/A3)
# J4: staircase downslope (-70 -> -109, 1 mV/3 ms) zero-crossing voltage -> per-cell E_rev
# J1: h_ss(V) = [A(V)/(V - E_rev)] / [A(-140)/(-140 - E_rev)], A = transient amplitude (extremum minus last 50 ms)
# J2: m_ss(V) = A_tail(V)/A_tail(+40) (tail peak minus tail end); tau_act(V) test-segment rising edge (after re-leak-correction)
# SMOKE=1 three-well figures
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "数据", "Lei全量")
SMOKE = os.environ.get("SMOKE", "0") == "1"

BATCH = os.environ.get("BATCH", "herg25oc1")
WELLS = [l.strip() for l in open(os.path.join(DATA, "qc", f"selected-{BATCH}.txt"), encoding="utf-8")
         if l.strip() and not l.startswith("#")]
if SMOKE:
    WELLS = ["A01", "B03", "C01"]

SIN_V = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]
ACT_V = [-50, -35, -20, -5, 10, 25, 40]
DT_I = 2e-4
DT_V = 1e-4


def load_protocol(name):
    d = np.genfromtxt(os.path.join(DATA, "protocol", f"protocol-{name}.csv"),
                      delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1:]


def load_current(proto, well):
    d = np.genfromtxt(os.path.join(DATA, proto if BATCH=="herg25oc1" else os.path.join(BATCH, proto), f"{BATCH}-{proto}-{well}.csv"),
                      delimiter=",", skip_header=1)
    if d.ndim == 1:
        d = d[:, None]
    return d


def releak_sweep(i5, v10):
    """Lei official re-leak-correction (ported from lib/releakcorrect.py): I' = I + g*(V - V0), g minimises |mean of 0.175-0.2 s window|"""
    v = v10[::2][:len(i5)]
    v0 = v[0]
    wa, wb = int(0.175 / DT_I), int(0.2 / DT_I)
    win_i = i5[wa:wb]
    win_v = v[wa:wb] - v0
    # closed-form solution of |mean(I + g*dV)| for g: g = -mean(I)/mean(dV) (signs aligned by the means)
    mv = np.mean(win_v)
    g = 0.0 if abs(mv) < 1e-9 else -np.mean(win_i) / mv
    return i5 + g * (v - v0), g


def segments(v10):
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]


def extract_J4(well):
    """staircase downslope zero-crossing -> E_rev"""
    _, V = load_protocol("staircaseramp")
    v10 = V[:, 0]
    I = load_current("staircaseramp", well)[:, 0]
    segs = segments(v10)
    # find the +40 x0.5 s segment (after -80 x1.0, immediately followed by the 3 ms small-step downslope)
    ramp_a = ramp_b = None
    for k, (vv, a, b) in enumerate(segs):
        if vv == 40 and 0.4 <= (b - a) * DT_V <= 0.6 and k + 1 < len(segs):
            # what follows should be the 3 ms small steps starting at -70
            vv2, a2, b2 = segs[k + 1]
            if vv2 == -70 and (b2 - a2) * DT_V < 0.02:
                ramp_a = a2
                # continue to the end of the small steps reaching >= -109
                j = k + 1
                while j < len(segs) and (segs[j][2] - segs[j][1]) * DT_V < 0.02:
                    ramp_b = segs[j][2]
                    j += 1
                break
    if ramp_a is None:
        return None
    ia, ib = ramp_a // 2, ramp_b // 2
    v_seg = v10[ramp_a:ramp_b:2]
    i_seg = I[ia:ib]
    n = min(len(v_seg), len(i_seg))
    v_seg, i_seg = v_seg[:n].copy(), i_seg[:n].copy()
    # robustify: 5-point median smoothing + skip the first 15 ms of the slope (+40 -> -70 step artefact)
    k_s = int(0.015 / DT_I)
    if len(i_seg) < k_s + 20:
        return None
    ker = np.ones(5) / 5
    i_sm = np.convolve(i_seg, ker, mode="same")
    s = np.sign(i_sm[k_s:])
    cross = np.where(np.diff(s) < 0)[0]  # only + to - (downslope current turns from positive to negative)
    if len(cross) == 0:
        return None
    k = cross[0] + k_s
    v0, v1 = v_seg[k], v_seg[k + 1]
    i0, i1 = i_sm[k], i_sm[k + 1]
    return float(v0 - i0 * (v1 - v0) / (i1 - i0))


def extract_J1(well, e_rev):
    _, V = load_protocol("sinactiv")
    I = load_current("sinactiv", well)
    n_sw = I.shape[1]
    # noise: -80 holding segment after re-leak-correction
    i0c, _ = releak_sweep(I[: int(0.1 / DT_I), 0], V[: int(0.1 / DT_V), 0])
    sig = float(np.std(i0c[: int(0.09 / DT_I)]))
    A = np.full(n_sw, np.nan)
    qc = []
    for sw in range(n_sw):
        ic, _ = releak_sweep(I[:, sw], V[:, sw])
        segs = segments(V[:, sw])
        test = None
        for vv, a, b in segs:
            dur = (b - a) * DT_V
            if vv == SIN_V[sw] and dur >= 0.4 and vv != -80:
                test = (a, b)
            elif SIN_V[sw] == -80 and vv == -80 and dur >= 0.5:
                test = (a, b)  # sweep3: -80x0.6
            elif SIN_V[sw] == 20 and vv == 20 and dur >= 0.9:
                test = (a, b)  # sweep8: +20x1.0
        if test is None:
            qc.append("no_seg")
            continue
        a, b = test[0] // 2, test[1] // 2
        seg = ic[a:b]
        if len(seg) < 100:
            qc.append("short")
            continue
        late = np.mean(seg[-int(0.05 / DT_I):])
        win = seg[: int(0.2 / DT_I)]
        pk = float(np.min(win)) if SIN_V[sw] < e_rev else float(np.max(win))
        amp = pk - late
        A[sw] = amp
        if abs(amp) < 4 * sig:
            qc.append("low_snr")
            continue
        qc.append("ok")
    # DF-corrected normalisation
    A0 = A[0]
    if not np.isfinite(A0) or abs(A0) < 1e-9:
        return None, sig
    h = np.full(n_sw, np.nan)
    for sw in range(n_sw):
        df = SIN_V[sw] - e_rev
        if abs(df) < 10:
            continue  # -80 level excluded
        h[sw] = (A[sw] / df) / (A0 / (SIN_V[0] - e_rev))
    ok = np.array([q == "ok" for q in qc]) & np.isfinite(h) & (h >= -0.1) & (h <= 1.3)
    return {"h": h.tolist(), "A": A.tolist(), "ok": ok.tolist(), "qc": qc}, sig


def extract_J2(well):
    _, V = load_protocol("sactiv")
    I = load_current("sactiv", well)
    n_sw = I.shape[1]
    i0c, _ = releak_sweep(I[: int(0.1 / DT_I), 0], V[: int(0.1 / DT_V), 0])
    sig = float(np.std(i0c[: int(0.09 / DT_I)]))
    atail = np.full(n_sw, np.nan)
    tau = np.full(n_sw, np.nan)
    delay = np.full(n_sw, np.nan)
    r2 = np.full(n_sw, np.nan)
    qc = []
    for sw in range(n_sw):
        ic, _ = releak_sweep(I[:, sw], V[:, sw])
        segs = segments(V[:, sw])
        test = tail = None
        for vv, a, b in segs:
            dur = (b - a) * DT_V
            if vv == ACT_V[sw] and dur >= 0.9:
                test = (a, b)
            elif vv == -40 and dur >= 0.4:
                tail = (a, b)
        if test is None or tail is None:
            qc.append("no_seg")
            continue
        ta, tb = tail[0] // 2, tail[1] // 2
        tseg = ic[ta:tb]
        xa, xb = test[0] // 2, test[1] // 2
        test_end = float(np.mean(ic[xb - int(0.05 / DT_I): xb]))  # amendment A4: subtract end-of-test-segment level
        pk = float(np.max(tseg[: int(0.15 / DT_I)]))
        amp = pk - test_end
        atail[sw] = amp
        if abs(amp) < 4 * sig:
            qc.append("low_snr")
            continue
        qc.append("ok")
    ref = atail[-1]
    m = atail / ref if np.isfinite(ref) and abs(ref) > 1e-9 else np.full(n_sw, np.nan)
    ok_m = np.array([q == "ok" for q in qc]) & np.isfinite(m) & (m >= -0.1) & (m <= 1.3)
    return {"m": m.tolist(), "atail": atail.tolist(),
            "ok_m": ok_m.tolist(), "qc": qc}, sig


def main():
    out = {}
    for w in WELLS:
        e = extract_J4(w)
        j1, sig1 = extract_J1(w, e if e is not None else -84.0)
        j2, sig2 = extract_J2(w)
        out[w] = {"E_rev": e, "sigma_sin": sig1, "sigma_act": sig2, "J1": j1, "J2": j2}
        print(w, "E_rev=", None if e is None else round(e, 1),
              "J1qc=", j1["qc"] if j1 else None, "J2qc=", j2["qc"], flush=True)
    fp = os.path.join(ROOT, f"lei211_J1J2_结果_{BATCH}.json" if not SMOKE else f"lei211_J1J2_冒烟_{BATCH}.json")
    json.dump({"SIN_V": SIN_V, "ACT_V": ACT_V, "cells": out}, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", fp)

    if SMOKE:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for w in WELLS:
            j1 = out[w]["J1"]
            if j1:
                hv = np.array(j1["h"]); okm = np.array(j1["ok"])
                axes[0, 0].plot(np.array(SIN_V)[okm], hv[okm], "o-", label=w, ms=4)
            j2 = out[w]["J2"]
            mv = np.array(j2["m"]); okm = np.array(j2["ok_m"])
            axes[0, 1].plot(np.array(ACT_V)[okm], mv[okm], "s-", label=w, ms=4)
            av = np.array(j2["atail"])
            axes[1, 0].plot(np.array(ACT_V)[okm], av[okm], "^-", label=w, ms=4)
        axes[0, 0].set_title("J1 h_ss(V) DF-corrected (amendment A2)")
        axes[0, 0].axhline(1.0, color="gray", lw=0.5)
        axes[0, 0].set_xlabel("mV"); axes[0, 0].legend()
        axes[0, 1].set_title("J2 m_ss(V) = A_tail/A_tail(+40) (amendment A4)")
        axes[0, 1].set_xlabel("mV")
        axes[1, 0].set_title("J2 A_tail(V) raw amplitude pA")
        axes[1, 0].set_xlabel("mV")
        # A01 sactiv sweep6 before/after re-leak-correction
        _, V6 = load_protocol("sactiv")
        I6 = load_current("sactiv", "A01")[:, 6]
        ic6, g6 = releak_sweep(I6, V6[:, 6])
        tt = np.arange(len(I6)) * DT_I
        axes[1, 1].plot(tt, I6, lw=0.5, label="before re-leak-corr", color="C7")
        axes[1, 1].plot(tt, ic6, lw=0.5, label=f"after re-leak-corr g={g6:.2f}", color="C0")
        axes[1, 1].set_title("A01 sactiv sweep6(+40) re-leak-correction comparison")
        axes[1, 1].legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(ROOT, "lei211_J1J2_冒烟.png")
        fig.savefig(png, dpi=130, bbox_inches="tight")
        print("saved", png)


if __name__ == "__main__":
    main()
