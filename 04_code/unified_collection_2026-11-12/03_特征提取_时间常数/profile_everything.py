# -*- coding: utf-8 -*-
"""轮廓似然·全数据完整重验（2026-11-08）
范围：corpus v5r 全部 270,827 行 + recovery v5 全部 5,212 行。
每行：严格复现冻结提取器逻辑重拟合（v2: Kv/K2P/Kir/other；v5_hcn: HCN；v5: Recovery），
随后对该参数做剖面似然 95% CI（固定其余结构，只优化 nuisance 参数）。
断点续跑：已完成行写 features/profile_all_rows.csv，重跑自动跳过。
用法: python profile_everything.py [预算秒数, 默认250]
"""
import sys, re, time
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd, h5py
from scipy.optimize import curve_fit
from multiprocessing import Pool

R = Path(__file__).parent
sys.path.insert(0, str(R))
from extract_features_v2 import (fit_tau_rise_v2, fit_tau_decay_v2, fit_boltz3,
                                 _rise, boltz3_inc, boltz3_dec, ACT_V, DEA_V, INA_V, E_REV)
from extract_features_v5_hcn import (fit_tau_rise_signed, fit_tau_decay_signed,
                                     fit_iv_boltz, read_meta,
                                     ACT_V as HCN_ACT_V, DEA_V as HCN_DEA_V)

CKPT = R / "features" / "profile_all_rows.csv"

# ---------------- 通用剖面 ----------------
def profile_generic(rss_of, hat, lo_b, hi_b, n, n_grid=60):
    try:
        rss0 = rss_of(hat)
    except Exception:
        return np.nan, np.nan
    if not np.isfinite(rss0): return np.nan, np.nan
    thr = rss0 * np.exp(2.0 * 1.92 / max(n, 5))
    lo = hi = np.nan
    for x in np.linspace(lo_b, hat, n_grid // 2)[::-1]:
        try: r = rss_of(x)
        except Exception: r = np.nan
        if np.isfinite(r) and r > thr: lo = x; break
    for x in np.linspace(hat, hi_b, n_grid // 2):
        try: r = rss_of(x)
        except Exception: r = np.nan
        if np.isfinite(r) and r > thr: hi = x; break
    return lo, hi

def prof_rise(tt, yy, p_fix, tau_hat, t0, tau_hi_bound):
    def rss_of(tv):
        f = lambda t_, Iss, c: _rise(t_, Iss, tv, p_fix, c, t0)
        pb, _ = curve_fit(f, tt, yy, p0=[max(yy.max(), 1e-9), yy[0]], maxfev=8000)
        return float(np.sum((yy - f(tt, *pb)) ** 2))
    return profile_generic(rss_of, tau_hat, max(tau_hat * 0.4, 0.05),
                           min(tau_hat * 2.5, tau_hi_bound), len(yy))

def prof_decay(tt, ii, t0, model, which, tau_hat, tau_hi_bound):
    """which=1 剖面快分量, 2 剖面慢分量(bi); mono 只 which=1。"""
    if model == "mono" or which == 1 and model != "bi":
        def rss_of(tv):
            f = lambda t_, A, c: A * np.exp(-(t_ - t0) / tv) + c
            pb, _ = curve_fit(f, tt, ii, p0=[max(ii.max() - ii.min(), 1e-9), ii[-1]],
                              maxfev=8000)
            return float(np.sum((ii - f(tt, *pb)) ** 2))
    elif which == 1:
        def rss_of(tv):
            f = lambda t_, A1, A2, t2, c: A1*np.exp(-(t_-t0)/tv) + A2*np.exp(-(t_-t0)/t2) + c
            pb, _ = curve_fit(f, tt, ii, p0=[ii.max()*0.6, ii.max()*0.4, tau_hat*5, ii[-1]],
                              bounds=([0, 0, 0.05, -np.inf], [np.inf, np.inf, tau_hi_bound, np.inf]),
                              maxfev=12000)
            return float(np.sum((ii - f(tt, *pb)) ** 2))
    else:
        def rss_of(tv):
            f = lambda t_, A1, t1, A2, c: A1*np.exp(-(t_-t0)/t1) + A2*np.exp(-(t_-t0)/tv) + c
            pb, _ = curve_fit(f, tt, ii, p0=[ii.max()*0.6, max(tau_hat/5, 0.1), ii.max()*0.4, ii[-1]],
                              bounds=([0, 0.05, 0, -np.inf], [np.inf, tau_hi_bound, np.inf, np.inf]),
                              maxfev=12000)
            return float(np.sum((ii - f(tt, *pb)) ** 2))
    return profile_generic(rss_of, tau_hat, max(tau_hat * 0.4, 0.05),
                           min(tau_hat * 2.5, tau_hi_bound), len(ii))

def prof_boltz_vh(V, y, inc, vh_hat, se_hint):
    f3 = boltz3_inc if inc else boltz3_dec
    def rss_of(vv):
        if inc:
            f = lambda V_, A, k, c: A/(1.0+np.exp(-(V_-vv)/k)) + c
        else:
            f = lambda V_, A, k, c: A/(1.0+np.exp((V_-vv)/k)) + c
        pb, _ = curve_fit(f, V, y, p0=[max(y.max()-y.min(), 1e-12), 8.0, y.min()],
                          bounds=([0, 0.5, -np.inf], [np.inf, 100, np.inf]), maxfev=12000)
        return float(np.sum((y - f(V, *pb)) ** 2))
    span = max(8.0, 6.0 * se_hint) if np.isfinite(se_hint) else 8.0
    return profile_generic(rss_of, vh_hat, vh_hat - span, vh_hat + span, len(y))

def prof_boltz_k(V, y, inc, k_hat):
    def rss_of(kv):
        if inc:
            f = lambda V_, A, vh, c: A/(1.0+np.exp(-(V_-vh)/kv)) + c
        else:
            f = lambda V_, A, vh, c: A/(1.0+np.exp((V_-vh)/kv)) + c
        pb, _ = curve_fit(f, V, y, p0=[max(y.max()-y.min(), 1e-12), np.median(V), y.min()],
                          maxfev=12000)
        return float(np.sum((y - f(V, *pb)) ** 2))
    return profile_generic(rss_of, k_hat, max(0.5, k_hat * 0.4), min(100.0, k_hat * 3.0), len(y))

def prof_hcn_boltz2(V, Gn, which, hat):
    """HCN 归一化两参数 Boltzmann: which='vh' 剖 Vh 优化 k; which='k' 反之。"""
    if which == "vh":
        def rss_of(vv):
            pb, _ = curve_fit(lambda V_, k: 1/(1+np.exp((V_-vv)/k)), V, Gn,
                              p0=[8.0], bounds=([2], [25]), maxfev=8000)
            return float(np.sum((Gn - 1/(1+np.exp((V-vv)/pb[0]))) ** 2))
        return profile_generic(rss_of, hat, max(-150, hat - 10), min(-50, hat + 10), len(Gn))
    else:
        def rss_of(kv):
            pb, _ = curve_fit(lambda V_, vh: 1/(1+np.exp((V_-vh)/kv)), V, Gn,
                              p0=[-90.0], bounds=([-150], [-50]), maxfev=8000)
            return float(np.sum((Gn - 1/(1+np.exp((V-pb[0])/kv))) ** 2))
        return profile_generic(rss_of, hat, max(2, hat*0.4), min(25, hat*2.5), len(Gn))

def prof_rec(dts, fracs, tau_hat):
    def rss_of(tv):
        f = lambda t_, c0, A: c0 + A*(1-np.exp(-np.clip(t_, 0, None)/tv))
        pb, _ = curve_fit(f, dts, fracs, p0=[fracs.min(), max(fracs.max()-fracs.min(), 0.15)],
                          bounds=([-0.5, 0.1], [1.5, 2.0]), maxfev=8000)
        return float(np.sum((fracs - f(dts, *pb)) ** 2))
    return profile_generic(rss_of, tau_hat, max(1.0, tau_hat*0.4), min(5000.0, tau_hat*2.5), len(dts))

# ---------------- 行构造辅助 ----------------
def _row(src, cid, chan, temp, par, V, val, se, model, p, r2, lo, hi):
    return dict(source=src, cell_id=int(cid), channel=chan, temp=temp, parameter=par,
                voltage_mV=V, value=val, se_pcov=se, model=model, p=p, r2=r2,
                prof_lo=lo, prof_hi=hi)

# ---------------- v2 管线细胞 (Kv/K2P/Kir/other) ----------------
def do_v2_cell(cid, chan, temp, targets):
    out = []
    fp = R / "nwb_full" / f"c{int(cid)}.nwb"
    need_act = any(p in ("tau_act", "Vhalf_act", "k_act") for p, _ in targets)
    need_dea = any(p in ("tau_deact", "tau_deact_slow2") for p, _ in targets)
    need_ina = any(p in ("tau_inact", "Vhalf_inact") for p, _ in targets)
    with h5py.File(fp, "r") as f:
        ts = f.get("acquisition/timeseries")
        if ts is None: return out
        if need_act and "Activation" in ts:
            d = ts["Activation/repetitions/repetition2/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T
            n = d.shape[1]; t = np.linspace(0, 700, n)
            G, Gv = [], []
            for si, V in enumerate(ACT_V):
                if si >= d.shape[0]: break
                i0 = d[si].astype(float) - d[si][t < 100].mean()
                seg_end = i0[(t > 550) & (t <= 600)].mean()
                seg_pk = i0[(t >= 100) & (t <= 600)].max()
                amp = seg_pk if abs(seg_pk) > abs(seg_end)*1.2 else seg_end
                if abs(V - E_REV) >= 1:
                    G.append(amp/(V-E_REV)); Gv.append(V)
                if ("tau_act", V) in targets and V >= -40:
                    ft = fit_tau_rise_v2(t, i0, 100, 600)
                    if ft:
                        m = (t >= 100) & (t <= 600); tt = t[m]; ii = i0[m]
                        pk = int(np.argmax(ii)); tt = tt[:pk+1]; yy = ii[:pk+1]
                        lo, hi = prof_rise(tt, yy, ft["p"], ft["tau"], 100, 2000)
                        out.append(_row("v2", cid, chan, temp, "tau_act", V, ft["tau"],
                                        ft["tau_se"], "", ft["p"], ft["r2"], lo, hi))
            if ("Vhalf_act", None) in targets or ("k_act", None) in targets:
                fh = fit_boltz3(Gv, G, inc=True)
                if fh:
                    if ("Vhalf_act", None) in targets:
                        lo, hi = prof_boltz_vh(np.asarray(Gv), np.asarray(G), True, fh["Vh"], fh["Vh_se"])
                        out.append(_row("v2", cid, chan, temp, "Vhalf_act", None, fh["Vh"],
                                        fh["Vh_se"], "", np.nan, fh["r2"], lo, hi))
                    if ("k_act", None) in targets:
                        lo, hi = prof_boltz_k(np.asarray(Gv), np.asarray(G), True, fh["k"])
                        out.append(_row("v2", cid, chan, temp, "k_act", None, fh["k"],
                                        fh["k_se"], "", np.nan, fh["r2"], lo, hi))
        if need_dea and "Deactivation" in ts:
            d = ts["Deactivation/repetitions/repetition1/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T
            n = d.shape[1]; t = np.linspace(0, 700, n)
            for si, V in enumerate(DEA_V):
                if si >= d.shape[0]: break
                if ("tau_deact", V) not in targets and ("tau_deact_slow2", V) not in targets: continue
                i0 = d[si].astype(float) - d[si][t < 100].mean()
                ft = fit_tau_decay_v2(t, i0, 402, 600)
                if not ft: continue
                m = (t >= 402) & (t <= 600); tt = t[m]; ii = i0[m]
                if ("tau_deact", V) in targets:
                    lo, hi = prof_decay(tt, ii, 402, ft.get("model"), 1, ft["tau"], 5000)
                    out.append(_row("v2", cid, chan, temp, "tau_deact", V, ft["tau"],
                                    ft["tau_se"], ft.get("model"), np.nan, ft["r2"], lo, hi))
                if ft.get("model") == "bi" and ("tau_deact_slow2", V) in targets:
                    lo, hi = prof_decay(tt, ii, 402, "bi", 2, ft["tau2"], 5000)
                    out.append(_row("v2", cid, chan, temp, "tau_deact_slow2", V, ft["tau2"],
                                    ft["tau2_se"], "bi", np.nan, ft["r2"], lo, hi))
        if need_ina and "Inactivation" in ts:
            d = ts["Inactivation/repetitions/repetition1/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T
            n = d.shape[1]; t = np.linspace(0, 1750, n)
            peaks = []
            for si, V in enumerate(INA_V):
                if si >= d.shape[0]: break
                i0 = d[si].astype(float) - d[si][t < 100].mean()
                test = i0[(t >= 1600) & (t <= 1700)]
                peaks.append(test.max() if len(test) else np.nan)
                if ("tau_inact", V) in targets:
                    cond_t = (t >= 200) & (t <= 1500)
                    cond = i0[cond_t]
                    if len(cond) > 50 and cond.max() > 0 and (cond.max()-cond[-1])/(cond.max()+1e-12) > 0.3:
                        ft = fit_tau_decay_v2(t[cond_t], cond, 200, 1500)
                        if ft:
                            tt = t[cond_t]
                            lo, hi = prof_decay(tt, cond, 200, ft.get("model"), 1, ft["tau"], 5000)
                            out.append(_row("v2", cid, chan, temp, "tau_inact", V, ft["tau"],
                                            ft["tau_se"], ft.get("model"), np.nan, ft["r2"], lo, hi))
            if ("Vhalf_inact", None) in targets:
                fh = fit_boltz3(INA_V, peaks, inc=False)
                if fh:
                    lo, hi = prof_boltz_vh(np.asarray(INA_V, float), np.asarray(peaks, float),
                                           False, fh["Vh"], fh["Vh_se"])
                    out.append(_row("v2", cid, chan, temp, "Vhalf_inact", None, fh["Vh"],
                                    fh["Vh_se"], "", np.nan, fh["r2"], lo, hi))
    return out

# ---------------- HCN 管线细胞 ----------------
def do_hcn_cell(cid, chan, temp, targets):
    out = []
    fp = R / "nwb_full" / f"c{int(cid)}.nwb"
    meta = read_meta(cid) or {}
    repid = meta.get("repid")
    with h5py.File(fp, "r") as f:
        ts = f.get("acquisition/timeseries")
        if ts is None or "Activation" not in ts: return out
        reps = ts["Activation/repetitions"]
        rep = f"repetition{repid}" if repid and f"repetition{repid}" in reps else \
              ("repetition2" if "repetition2" in reps else list(reps.keys())[0])
        d = ts[f"Activation/repetitions/{rep}/data"][:]
        if d.ndim == 1: d = d[:, None]
        d = d.T
        xi = 1e-4
        npts = d.shape[1]
        if npts < 30000: return out
        t = np.arange(npts)*xi*1000.0
        base = d[:, int(0.05/xi):int(0.09/xi)].mean(axis=1)
        noise = d[:, int(0.05/xi):int(0.09/xi)].std(axis=1)
        d0 = d - base[:, None]
        Iss = d0[:, int(2.9/xi):int(3.05/xi)].mean(axis=1)
        V = np.array(HCN_ACT_V[:d.shape[0]], float)
        # Vhalf/k: 重建归一化 G
        if ("Vhalf_act", None) in targets or ("k_act", None) in targets:
            lm = (V >= -40) & (V <= -10)
            if lm.sum() >= 3:
                lk = np.polyfit(V[lm], Iss[lm], 1)
                Ic = Iss - np.polyval(lk, V)
                if abs(Ic[-1]) >= max(0.02, 5*float(noise[-1])):
                    gm = V <= -60
                    if gm.sum() >= 6:
                        Vg = V[gm]; G = Ic[gm]/(Vg-(-45.0))
                        if G[-1] > 0:
                            Gn = G/G[-1]
                            fh = fit_iv_boltz(V, Iss, noise)
                            if fh:
                                if ("Vhalf_act", None) in targets:
                                    lo, hi = prof_hcn_boltz2(Vg, Gn, "vh", fh["Vh"])
                                    out.append(_row("hcn", cid, chan, temp, "Vhalf_act", None,
                                                    fh["Vh"], fh["Vh_se"], "", np.nan, fh["r2"], lo, hi))
                                if ("k_act", None) in targets:
                                    lo, hi = prof_hcn_boltz2(Vg, Gn, "k", fh["k"])
                                    out.append(_row("hcn", cid, chan, temp, "k_act", None,
                                                    fh["k"], fh["k_se"], "", np.nan, fh["r2"], lo, hi))
        Iref = np.median(Iss[:3])
        for si in range(d.shape[0]):
            if ("tau_act", float(V[si])) not in targets: continue
            if abs(Iss[si]-Iref) < max(0.02, 5*noise[si]): continue
            ft = fit_tau_rise_signed(t, d0[si], 120.0, 3050.0)
            if ft:
                m = (t >= 120.0) & (t <= 3050.0)
                tt = t[m]; ii = d0[si][m].astype(float)
                if (ii[-int(len(ii)*0.1):].mean() - ii[:int(len(ii)*0.05)].mean()) < 0:
                    ii = -ii
                lo, hi = prof_rise(tt, ii, ft["p"], ft["tau"], 120.0, 20000)
                out.append(_row("hcn", cid, chan, temp, "tau_act", float(V[si]), ft["tau"],
                                ft["tau_se"], "", ft["p"], ft["r2"], lo, hi))
        # Deactivation
        if any(p in ("tau_deact", "tau_deact_slow2") for p, _ in targets) and "Deactivation" in ts:
            dd = ts["Deactivation/repetitions/repetition1/data"][:]
            if dd.ndim == 1: dd = dd[:, None]
            dd = dd.T
            if dd.shape[1] >= 26000:
                td = np.arange(dd.shape[1])*xi*1000.0
                based = dd[:, int(0.05/xi):int(0.09/xi)].mean(axis=1)
                dd0 = dd - based[:, None]
                for si in range(min(dd.shape[0], len(HCN_DEA_V))):
                    Vd = float(HCN_DEA_V[si])
                    if ("tau_deact", Vd) not in targets and ("tau_deact_slow2", Vd) not in targets: continue
                    ft = fit_tau_decay_signed(td, dd0[si], 1615.0, 2650.0)
                    if not ft: continue
                    m = (td >= 1615.0) & (td <= 2650.0)
                    tt = td[m]; ii = dd0[si][m].astype(float)
                    if ii[:5].mean() < ii[-5:].mean(): ii = -ii
                    if ("tau_deact", Vd) in targets:
                        lo, hi = prof_decay(tt, ii, 1615.0, ft.get("model"), 1, ft["tau"], 20000)
                        out.append(_row("hcn", cid, chan, temp, "tau_deact", Vd, ft["tau"],
                                        ft["tau_se"], ft.get("model"), np.nan, ft["r2"], lo, hi))
                    if ft.get("model") == "bi" and ("tau_deact_slow2", Vd) in targets:
                        lo, hi = prof_decay(tt, ii, 1615.0, "bi", 2, ft["tau2"], 20000)
                        out.append(_row("hcn", cid, chan, temp, "tau_deact_slow2", Vd, ft["tau2"],
                                        ft["tau2_se"], "bi", np.nan, ft["r2"], lo, hi))
    return out

# ---------------- Recovery 细胞 ----------------
def do_rec_cell(cid, chan, temp):
    out = []
    fp = R / "nwb_full" / f"c{int(cid)}.nwb"
    XI = 1e-4
    with h5py.File(fp, "r") as f:
        ts = f.get("acquisition/timeseries")
        if ts is None or "Recovery" not in ts: return out
        reps = ts["Recovery/repetitions"]
        rep = "repetition1" if "repetition1" in reps else list(reps.keys())[0]
        dg = ts[f"Recovery/repetitions/{rep}/data"]
        keys = [k for k in dg.keys() if re.match(r"data\d+$", k)]
        if len(keys) < 8: return out
        keys = sorted(keys, key=lambda x: int(x[4:]))
        dts, fracs = [], []
        for k in keys:
            seg = dg[k]["data"][:].ravel()
            n = len(seg)
            if n < 15000: continue
            dur = n*XI
            base = seg[int(0.02/XI):int(0.08/XI)].mean()
            pp = seg[int(0.105/XI):int(0.4/XI)].max() - base
            rec_lvl = seg[int(1.55/XI):int(1.65/XI)].mean() - base if n > int(1.7/XI) else 0.0
            tp = seg[max(0, n-int(0.26/XI)):n-int(0.02/XI)].max() - base - rec_lvl
            if pp < 0.05: continue
            dts.append(dur - 1.75); fracs.append(tp/pp)
        if len(dts) < 8: return out
        dts = np.array(dts)*1000.0; fracs = np.array(fracs)
        if fracs.max() - fracs.min() < 0.15: return out
        def frec(t_, c0, A, tau): return c0 + A*(1-np.exp(-np.clip(t_, 0, None)/tau))
        p0 = [fracs.min(), max(fracs.max()-fracs.min(), 0.15), 100.0]
        p, pcov = curve_fit(frec, dts, fracs, p0=p0,
                            bounds=([-0.5, 0.1, 1], [1.5, 2.0, 5000]), maxfev=20000)
        pred = frec(dts, *p)
        r2 = 1 - np.sum((fracs-pred)**2)/max(np.sum((fracs-fracs.mean())**2), 1e-30)
        if r2 < 0.7 or not (1 <= p[2] <= 5000): return out
        try: se = float(np.sqrt(pcov[2][2]))
        except Exception: se = np.nan
        lo, hi = prof_rec(dts, fracs, float(p[2]))
        out.append(_row("rec", cid, chan, temp, "tau_rec", None, float(p[2]), se,
                        "mono", np.nan, float(r2), lo, hi))
    return out

def do_cell(task):
    cid, fam, chan, temp, targets, want_rec = task
    try:
        rows = []
        if fam == "HCN":
            rows += do_hcn_cell(cid, chan, temp, targets)
        else:
            rows += do_v2_cell(cid, chan, temp, targets)
        if want_rec:
            rows += do_rec_cell(cid, chan, temp)
        return rows
    except Exception as e:
        return [dict(source="ERROR", cell_id=int(cid), channel=chan, temp=temp,
                     parameter="ERROR", voltage_mV=np.nan, value=np.nan, se_pcov=np.nan,
                     model=str(e)[:100], p=np.nan, r2=np.nan, prof_lo=np.nan, prof_hi=np.nan)]

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 250.0
    t_start = time.time()
    c = pd.read_csv(R / "features" / "features_corpus_v5r.csv", low_memory=False)
    rec = pd.read_csv(R / "features" / "features_recovery_v5.csv")
    fam_map = c.drop_duplicates("cell_id").set_index("cell_id").family.to_dict()

    done = set()
    if CKPT.exists():
        dk = pd.read_csv(CKPT, usecols=["source", "cell_id", "parameter", "voltage_mV"])
        for _, r in dk.iterrows():
            v = r.voltage_mV
            done.add(f"{r.source}|{r.cell_id}|{r.parameter}|{'nan' if pd.isna(v) else v}")

    # 每细胞任务: targets = {(param, V or None)}
    cell_tasks = {}
    for r in c.itertuples(index=False):
        V = None if pd.isna(r.voltage_mV) else float(r.voltage_mV)
        src = "hcn" if fam_map.get(r.cell_id) == "HCN" else "v2"
        if f"{src}|{r.cell_id}|{r.parameter}|{'nan' if V is None else V}" in done: continue
        ent = cell_tasks.setdefault(r.cell_id, set())
        ent.add((r.parameter, V))
    rec_cells = {}
    for r in rec.itertuples(index=False):
        if f"rec|{r.cell_id}|tau_rec|nan" in done: continue
        rec_cells[r.cell_id] = (r.channel, r.temp_nominal)

    chan_map = c.drop_duplicates("cell_id").set_index("cell_id").channel.to_dict()
    temp_map = c.drop_duplicates("cell_id").set_index("cell_id").temp_nominal.to_dict()
    all_ids = set(cell_tasks) | set(rec_cells)
    todo = []
    for cid in sorted(all_ids):
        chan = chan_map.get(cid) or (rec_cells.get(cid) or ("?",))[0]
        temp = temp_map.get(cid) or (rec_cells.get(cid) or (None, None))[1]
        todo.append((cid, fam_map.get(cid, "other"), chan, temp,
                     cell_tasks.get(cid, set()), cid in rec_cells))
    print(f"待处理细胞 {len(todo)} (已完成行 {len(done)})", flush=True)
    if not todo:
        print("ALL DONE"); return

    n_rows = 0
    with Pool(processes=24) as pool:
        for res in pool.imap_unordered(do_cell, todo, chunksize=4):
            if res:
                good = [r for r in res if r["parameter"] != "ERROR"]
                bad = [r for r in res if r["parameter"] == "ERROR"]
                if good:
                    pd.DataFrame(good).to_csv(CKPT, mode="a", header=not CKPT.exists(), index=False)
                    n_rows += len(good)
                if bad:
                    for b in bad: print("  ERROR", b["cell_id"], b["model"], flush=True)
            if time.time() - t_start > budget:
                pool.terminate(); break
        else:
            pool.close()
    el = time.time() - t_start
    print(f"本轮 {el:.0f}s, 新增 {n_rows} 行", flush=True)

if __name__ == "__main__":
    main()
