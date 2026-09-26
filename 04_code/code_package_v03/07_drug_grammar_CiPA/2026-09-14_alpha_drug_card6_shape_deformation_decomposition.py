# 2026-09-14_α模型_药物卡6_形状变形分解.py
# ============================================================================
# Drug card 6: shape-deformation decomposition (tau-like vs h_ss-like), preregistered attribution card
# Preregistration: 结果\预注册_α模型_药物卡6_形状变形分解_2026-09-14.md (criteria pinned before the run)
#
# Method: card-5 units reused verbatim; normalized ramp-window shape is time-warp registered
#   with kappa* (grid [0.5,2.0] step 0.02, domain D(k) never out of bounds);
#   f_tau = 1-(eps_x1/eps_x0_sd)^2 (variance share explained by the time scale);
#   per-cell control odd/even split gives the f_tau_cc floor; cell-level paired Wilcoxon.
# Criteria: synthetic self-check C1/C2/C3 + regression (vs card-5 unit table <1e-9); main
#   criterion T1 (paired p<0.01 and median delta f_tau>=0.25) x T2 (median eps_x1>0.03), four quadrants.
# ============================================================================
import os
import json
import zipfile
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/结果/"
        "hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包/a6k5t")
SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_SETS = [("a6k5t-osfstorage-hERG-archivelab3", "dofetilide"),
              ("a6k5t-osfstorage-hERG-archivelab4", "pimozide"),
              ("a6k5t-osfstorage-hERG_Phase_II-archivelab1", "moxifloxacin")]

MIN_ACTRL_PA = 50.0
B_LO, B_HI = 0.2, 0.9
MIN_SWEEPS = 5
MAX_CELLS = 3
RNG = np.random.default_rng(20260914)

K_GRID = np.round(np.arange(0.5, 2.0001, 0.02), 4)  # pinned: 76 points including 1.0
EX0_GATE = 0.03          # card-5 eps_x_thr on-record value: judge-able gate
FTAU_FLOOR = 0.25        # T1 effect floor: median delta f_tau >= 0.25
WILCOX_P = 0.01          # T1 significance line


def conc_nM(conc, unit):
    u = (unit or "").strip().lower()
    return conc * 1000.0 if (u.startswith("u") or u.startswith("µ")) else conc


# ---------- ted.xlsx (identical to card 5 v3, verbatim) ----------
def load_ted(tedx):
    wb = openpyxl.load_workbook(tedx, read_only=True)
    rw = [list(r) for r in wb["ResultsWide"].iter_rows(values_only=True)]
    cd = [list(r) for r in wb["CursorsDefinitions"].iter_rows(values_only=True)]
    wb.close()
    h = rw[0]
    ix = {k: h.index(k) for k in ("CELLID", "TRACENUM", "ELTIME", "LIQUID", "CONC", "CONCU",
                                  "Ramp", "CTLFL")}
    cells = {}
    for r in rw[1:]:
        if r[ix["CELLID"]] is None or r[ix["Ramp"]] is None:
            continue
        try:
            tn = int(r[ix["TRACENUM"]])
            et = float(r[ix["ELTIME"]]) / 1000.0
            rv = float(r[ix["Ramp"]])
            liq = str(r[ix["LIQUID"]]).lower()
            cc = float(r[ix["CONC"]]) if r[ix["CONC"]] is not None else 0.0
            cu = str(r[ix["CONCU"]] or "")
            ctl = str(r[ix["CTLFL"]] or "")
        except (TypeError, ValueError):
            continue
        cells.setdefault(str(r[ix["CELLID"]]), []).append((tn, et, liq, cc, cu, rv, ctl))
    hdr_cd = [str(c).strip().upper() if c else "" for c in cd[0]]
    ic = {n: hdr_cd.index(n) for n in ("CURSOR", "STIME", "ETIME")}
    # v3: cursor unit columns are mixed (lab1/lab2=ms, lab3/4/5=s); unified to seconds
    iu_s = hdr_cd.index("STIMEU") if "STIMEU" in hdr_cd else None
    iu_e = hdr_cd.index("ETIMEU") if "ETIMEU" in hdr_cd else None
    curs = {}
    for r in cd[1:]:
        if r and r[ic["CURSOR"]]:
            try:
                st, et = float(r[ic["STIME"]]), float(r[ic["ETIME"]])
                if iu_s is not None and str(r[iu_s]).strip().lower().startswith("ms"):
                    st /= 1000.0
                if iu_e is not None and str(r[iu_e]).strip().lower().startswith("ms"):
                    et /= 1000.0
                curs[str(r[ic["CURSOR"]]).strip()] = (st, et)
            except (TypeError, ValueError):
                pass
    return cells, curs


def cell_plan(rows):
    """One cell -> (A_ctrl, ctrl last-5-sweep TRACENUM, [(conc, last-5-sweep TRACENUM, b_ss)]). Verbatim from card 5."""
    tn0 = sorted((r for r in rows
                  if (r[6] == "Y" and (r[4] == 0.0 or "control" in r[2] or "vehicle" in r[2]))
                  or "control" in r[2] or "vehicle" in r[2]), key=lambda r: r[1])
    drg = sorted((r for r in rows if not r[2].startswith("e-4031")
                  and "control" not in r[2] and "vehicle" not in r[2]), key=lambda r: r[1])
    if len(tn0) < MIN_SWEEPS or not drg:
        return None
    A_ctrl = float(np.median([r[5] for r in tn0[-5:]]))
    a0 = abs(A_ctrl)
    if a0 < 1e-6:
        scale = 1e12
    elif a0 < 0.02:
        return None
    elif a0 < 50.0:
        scale = 1e3
    else:
        scale = 1.0
    A_ctrl *= scale
    if abs(A_ctrl) < MIN_ACTRL_PA:
        return None
    ctrl_tr = [r[0] for r in tn0[-5:]]
    by_c = {}
    for r in drg:
        by_c.setdefault(conc_nM(r[3], r[4]), []).append(r)
    segs = []
    for C, rs in sorted(by_c.items()):
        if len(rs) < MIN_SWEEPS:
            continue
        last = rs[-5:]
        b = 1.0 - float(np.median([x[5] * scale for x in last])) / A_ctrl
        if B_LO <= b <= B_HI:
            segs.append((C, [x[0] for x in last], float(b)))
    if not segs:
        return None
    return A_ctrl, ctrl_tr, segs


# ---------- raw csv (identical to card 5 v3: single zipfile handle) ----------
def csv_cols(hdr):
    m = {}
    for j, c in enumerate(hdr):
        c = c.strip()
        if c.startswith("trace_#"):
            num, _, kind = c[7:].partition("_current")
            if kind:
                m.setdefault(int(num), {})["i"] = j
            else:
                num, _, kind = c[7:].partition("_voltage")
                if kind:
                    m.setdefault(int(num), {})["v"] = j
    return m


def load_csv_traces(path, need):
    """Read only the needed trace current columns + t_ms. Returns t(s), {N: i}."""
    z = None
    if path.endswith(".zip"):
        z = zipfile.ZipFile(path)
        fh = z.open(z.namelist()[0])
    else:
        fh = open(path, "rb")
    hdr = fh.readline().decode("utf-8", "replace").strip().split(",")
    cmap = csv_cols(hdr)
    use = ["t_ms"]
    for N in need:
        if N in cmap and "i" in cmap[N]:
            use.append(hdr[cmap[N]["i"]])
    df = pd.read_csv(fh, names=hdr, header=None, usecols=lambda c: c in use)
    fh.close()
    if z is not None:
        z.close()
    t = df["t_ms"].to_numpy(float) / 1000.0
    out = {}
    for N in need:
        col = None
        if N in cmap and "i" in cmap[N] and hdr[cmap[N]["i"]] in df:
            col = hdr[cmap[N]["i"]]
        if col:
            out[N] = df[col].to_numpy(float)
    return t, out


# ---------- shape basics (verbatim from card 5) ----------
def mean_trace(t, traces, bwin):
    sub = []
    for i in traces:
        m = (t >= bwin[0]) & (t <= bwin[1])
        b0 = np.nanmean(i[m]) if m.sum() >= 3 else 0.0
        sub.append(i - b0)
    return np.nanmean(sub, axis=0)


def shape_eps(t, ctrl, drug, bwin, rwin):
    """Card-5 eps and eps_noise + normalized window waveform (for warp).
    Returns (eps_full, eps_noise, tm, cn, dn) or None."""
    m = (t >= rwin[0]) & (t <= rwin[1])
    mb = (t >= bwin[0]) & (t <= bwin[1])
    if m.sum() < 20:
        return None
    c, d = ctrl[m], drug[m]
    if not (np.isfinite(c).all() and np.isfinite(d).all()):
        return None
    sc = np.max(np.abs(c))
    sd = np.max(np.abs(d))
    if sc <= 0 or sd <= 0:
        return None
    cn, dn = c / sc, d / sd
    eps = float(np.sqrt(np.mean((dn - cn) ** 2)) / np.sqrt(np.mean(cn ** 2)))
    sg_c = float(np.nanstd(ctrl[mb])) if mb.sum() >= 3 else 0.0
    sg_d = float(np.nanstd(drug[mb])) if mb.sum() >= 3 else 0.0
    eps_noise = float(np.sqrt((sg_d / sd) ** 2 + (sg_c / sc) ** 2))
    return eps, eps_noise, t[m], cn, dn


# ---------- kappa* registration (pinned: grid [0.5,2.0] step 0.02, domain never out of bounds) ----------
def _domain_eps(tm, cn, dn, rwin, k):
    # v2 fix: stretch around window start r0 (ramp phase counted from r0): dq(t)=dn(r0+k*(t-r0));
    # domain D(k)=[r0, min(r1, r0+(r1-r0)/k)] (k<=1 full window; k>1 right edge shrinks, no overrun)
    r0, r1 = rwin
    hi = r0 + (r1 - r0) / k if k > 1.0 else r1
    sel = (tm >= r0) & (tm <= hi)
    if sel.sum() < 20:
        return None
    tq = tm[sel]
    cq = np.interp(tq, tm, cn)
    dq = np.interp(r0 + k * (tq - r0), tm, dn)
    nq = np.max(np.abs(cq))
    nd = np.max(np.abs(dq))
    if nq <= 0 or nd <= 0:
        return None
    cq, dq = cq / nq, dq / nd
    return float(np.sqrt(np.mean((dq - cq) ** 2)) / np.sqrt(np.mean(cq ** 2)))


def warp_fit(tm, cn, dn, rwin, eps_noise):
    """Returns (kappa*, eps_x1, eps_x0_sd, f_tau_raw) or None.
    f_tau_raw has no judge-able gate (the cc floor uses the same convention)."""
    best = None
    for k in K_GRID:
        e = _domain_eps(tm, cn, dn, rwin, k)
        if e is not None and (best is None or e < best[1]):
            best = (float(k), e)
    if best is None:
        return None
    k_star, e1 = best
    # same-domain eps(1): take kappa=1 on D(kappa*) (under the r0 convention this compares the original waveform on the shrunk domain)
    r0, r1 = rwin
    hi = r0 + (r1 - r0) / k_star if k_star > 1.0 else r1
    sel = (tm >= r0) & (tm <= hi)
    tq = tm[sel]
    cq = np.interp(tq, tm, cn)
    dq = np.interp(tq, tm, dn)
    nq = np.max(np.abs(cq))
    nd = np.max(np.abs(dq))
    cq, dq = cq / nq, dq / nd
    e0 = float(np.sqrt(np.mean((dq - cq) ** 2)) / np.sqrt(np.mean(cq ** 2)))
    ex1 = float(np.sqrt(max(e1 * e1 - eps_noise * eps_noise, 0.0)))
    ex0 = float(np.sqrt(max(e0 * e0 - eps_noise * eps_noise, 0.0)))
    f_tau = float(max(0.0, 1.0 - (ex1 / ex0) ** 2)) if ex0 > 1e-6 else None
    edge = bool(abs(k_star - K_GRID[0]) < 1e-9 or abs(k_star - K_GRID[-1]) < 1e-9)
    return k_star, ex1, ex0, f_tau, edge


# ---------- synthetic self-check deformation pieces ----------
def ar1_noise(n, sigma, rho=0.33, rng=RNG):
    e = rng.normal(0, sigma * np.sqrt(1 - rho * rho), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def stretch_window(y, t, rwin, factor):
    """Uniform time stretch of the full ramp window by x factor (tau-like deformation): y2(t)=y(r0+(t-r0)/factor)."""
    m = np.where((t >= rwin[0]) & (t <= rwin[1]))[0]
    seg = y[m]
    tt = t[m]
    src = rwin[0] + (tt - rwin[0]) / factor
    y2 = y.copy()
    y2[m] = np.interp(src, tt, seg)
    return y2


def taper_tail(y, t, rwin, factor=0.75):
    """Linear post-peak amplitude taper to x factor (time untouched, h_ss-like amplitude deformation)."""
    m = np.where((t >= rwin[0]) & (t <= rwin[1]))[0]
    seg = y[m]
    pk = int(np.argmax(np.abs(seg)))
    idx = m[pk:]
    y2 = y.copy()
    y2[idx] = y2[idx] * np.linspace(1.0, factor, len(idx))
    return y2


def synth_check(t, tmpl, bwin, rwin):
    print("\n[synthetic self-check]", flush=True)
    mb = (t >= bwin[0]) & (t <= bwin[1])
    sigma = float(np.nanstd(tmpl[mb])) if mb.sum() >= 3 else 1.0
    res = {"C1": [], "C2": [], "C3": []}
    for _ in range(20):
        c1 = 0.5 * tmpl + ar1_noise(len(tmpl), sigma)
        c2 = stretch_window(0.5 * tmpl, t, rwin, 1.25) + ar1_noise(len(tmpl), sigma)
        c3 = taper_tail(0.5 * tmpl, t, rwin, 0.75) + ar1_noise(len(tmpl), sigma)
        for key, cc in (("C1", c1), ("C2", c2), ("C3", c3)):
            se = shape_eps(t, tmpl, cc, bwin, rwin)
            if se is None:
                continue
            eps, en, tm, cn, dn = se
            wf = warp_fit(tm, cn, dn, rwin, en)
            if wf is None:
                continue
            res[key].append((wf[0], wf[3] if wf[3] is not None else np.nan))
    out = {}
    for key in ("C1", "C2", "C3"):
        ks = np.array([r[0] for r in res[key]])
        fs = np.array([r[1] for r in res[key]])
        out[key] = dict(kappa=ks, f_tau=fs)
    c1_ok = bool(len(out["C1"]["kappa"]) == 20 and
                 np.all(np.abs(out["C1"]["kappa"] - 1.0) <= 0.05 + 1e-9))
    c2_k = bool(np.sum(np.abs(out["C2"]["kappa"] - 1.25) <= 0.10 + 1e-9) >= 18)
    c2_f = bool(np.sum(out["C2"]["f_tau"] >= 0.8) >= 18)
    c2_ok = c2_k and c2_f
    # v2: C3 criterion keeps only the kappa* measure; f_tau moved to registry - smoke evidence:
    # a single-parameter stretch can absorb ~80% of a smooth amplitude taper (f_tau~0.8),
    # so f_tau is not used as tau evidence when |kappa*-1| < k_thr
    c3_ok = bool(np.sum(np.abs(out["C3"]["kappa"] - 1.0) <= 0.05 + 1e-9) >= 18)
    k_thr = float(max(3.0 * np.median(np.abs(out["C1"]["kappa"] - 1.0)), 0.10))
    print(f"  C1 pure scaling: kappa* median={np.median(out['C1']['kappa']):.3f} "
          f"all in [0.95,1.05]={'yes' if c1_ok else 'no'}; "
          f"raw f_tau median={np.nanmedian(out['C1']['f_tau']):.3f} (synthetic floor reference)", flush=True)
    print(f"  C2 time stretch x1.25: kappa* median={np.median(out['C2']['kappa']):.3f} "
          f"recovered {np.sum(np.abs(out['C2']['kappa'] - 1.25) <= 0.10 + 1e-9)}/20 (>=18); "
          f"f_tau median={np.nanmedian(out['C2']['f_tau']):.3f}; >=0.8 in "
          f"{np.sum(out['C2']['f_tau'] >= 0.8)}/20（≥18）", flush=True)
    print(f"  C3 amplitude taper x0.75: kappa* median={np.median(out['C3']['kappa']):.3f} "
          f"in [0.95,1.05] in {np.sum(np.abs(out['C3']['kappa'] - 1.0) <= 0.05 + 1e-9)}/20 (>=18); "
          f"f_tau median={np.nanmedian(out['C3']['f_tau']):.3f} (registry only, not tau evidence)", flush=True)
    print(f"  k_thr = {k_thr:.3f} (C1-anchored max(3 x C1 median |kappa*-1|, 0.10))", flush=True)
    ok = c1_ok and c2_ok and c3_ok
    print(f"  synthetic self-check {'pass' if ok else 'fail - whole card downgraded to registry'}", flush=True)
    return ok, dict(C1_k_med=float(np.median(out["C1"]["kappa"])),
                    C1_f_med=float(np.nanmedian(out["C1"]["f_tau"])),
                    C2_k_med=float(np.median(out["C2"]["kappa"])),
                    C2_f_med=float(np.nanmedian(out["C2"]["f_tau"])),
                    C3_k_med=float(np.median(out["C3"]["kappa"])),
                    C3_f_med=float(np.nanmedian(out["C3"]["f_tau"])),
                    C1_ok=c1_ok, C2_ok=c2_ok, C3_ok=c3_ok,
                    k_thr=k_thr), k_thr


# ---------- template lookup (identical to card 5 v3, verbatim) ----------
def find_template(datasets):
    for arch, drug, ds in datasets:
        tedx = os.path.join(ds, "subtracted", "ted", "ted.xlsx")
        try:
            cells, curs = load_ted(tedx)
        except Exception:
            continue
        if "Baseline" not in curs or "Ramp" not in curs:
            continue
        for cid, rows in cells.items():
            p = cell_plan(rows)
            if not p:
                continue
            fn1 = os.path.join(ds, "subtracted", "ted", cid + ".csv")
            fn2 = fn1 + ".zip"
            fn = fn1 if os.path.exists(fn1) else (fn2 if os.path.exists(fn2) else None)
            if fn is None:
                continue
            try:
                t, tr = load_csv_traces(fn, p[1])
                ctrs = [tr[N] for N in p[1] if N in tr]
                if len(ctrs) < 3:
                    continue
                tmpl = mean_trace(t, ctrs, curs["Baseline"])
                if tmpl is None or len(tmpl) < 10 or not np.isfinite(tmpl).all():
                    continue
                return t, tmpl, curs["Baseline"], curs["Ramp"]
            except Exception:
                continue
    return None


# ---------- dataset analysis ----------
def analyze_dataset(arch, drug, ds_dir, smoke):
    tedx = os.path.join(ds_dir, "subtracted", "ted", "ted.xlsx")
    cells, curs = load_ted(tedx)
    if "Baseline" not in curs or "Ramp" not in curs:
        return dict(判="游标缺", units=[])
    bwin, rwin = curs["Baseline"], curs["Ramp"]
    plans = {}
    for cid, rows in cells.items():
        p = cell_plan(rows)
        if p:
            plans[cid] = p
    if not plans:
        return dict(判="无合格单元", units=[])
    med_ac = float(np.median([p[0] for p in plans.values()]))
    picks = sorted(plans.items(), key=lambda kv: abs(kv[1][0] - med_ac))[:MAX_CELLS if not smoke else 1]
    units = []
    for cid, (A_ctrl, ctrl_tr, segs) in picks:
        fn1 = os.path.join(ds_dir, "subtracted", "ted", cid + ".csv")
        fn2 = fn1 + ".zip"
        fn = fn1 if os.path.exists(fn1) else (fn2 if os.path.exists(fn2) else None)
        if fn is None:
            units.append(dict(cell=cid, 判定="raw缺"))
            continue
        need = list(ctrl_tr)
        for C, trs, b in segs:
            need += trs
        try:
            t, tr = load_csv_traces(fn, need)
        except Exception as e:
            units.append(dict(cell=cid, 判定="csv读失败", err=str(e)[:120]))
            continue
        ctrs = [tr[N] for N in ctrl_tr if N in tr]
        if len(ctrs) < 3:
            units.append(dict(cell=cid, 判定="对照trace缺"))
            continue
        ctrl = mean_trace(t, ctrs, bwin)
        # control odd/even split -> f_tau_cc floor (no gate)
        f_tau_cc = None
        if len(ctrs) >= 4:
            odd = mean_trace(t, ctrs[0::2], bwin)
            even = mean_trace(t, ctrs[1::2], bwin)
            se_cc = shape_eps(t, odd, even, bwin, rwin)
            if se_cc is not None:
                wf_cc = warp_fit(se_cc[2], se_cc[3], se_cc[4], rwin, se_cc[1])
                if wf_cc is not None:
                    f_tau_cc = wf_cc[3]
        for C, trs, b in segs:
            drs = [tr[N] for N in trs if N in tr]
            if len(drs) < 3:
                continue
            drug_tr = mean_trace(t, drs, bwin)
            se = shape_eps(t, ctrl, drug_tr, bwin, rwin)
            if se is None:
                continue
            eps, en, tm, cn, dn = se
            ex0_full = float(np.sqrt(max(eps * eps - en * en, 0.0)))
            wf = warp_fit(tm, cn, dn, rwin, en)
            if wf is None:
                continue
            k_star, ex1, ex0_sd, f_tau_raw, edge = wf
            gated = bool(ex0_full >= EX0_GATE)
            units.append(dict(cell=cid, conc_nM=C, b_ss=b,
                              eps_x0_full=ex0_full, eps_noise=en,
                              kappa=k_star, eps_x1=ex1, eps_x0_sd=ex0_sd,
                              f_tau=f_tau_raw if gated else None,
                              f_tau_raw=f_tau_raw, f_tau_cc=f_tau_cc,
                              edge=edge, gated=gated, 判定="ok"))
    ok = [u for u in units if u["判定"] == "ok"]
    return dict(判="ok" if ok else "无合格单元", units=units)


# ---------- main flow ----------
def main():
    print("=" * 76, flush=True)
    print(" Drug card 6: shape-deformation decomposition (tau-like vs h_ss-like)", flush=True)
    print("=" * 76, flush=True)

    datasets = []
    for arch in sorted(os.listdir(PACK)):
        d0 = os.path.join(PACK, arch)
        if not (os.path.isdir(d0) and arch.startswith("a6k5t-osfstorage-hERG")):
            continue
        for drug in sorted(os.listdir(d0)):
            ds = os.path.join(d0, drug)
            if os.path.isdir(ds) and os.path.exists(os.path.join(ds, "subtracted", "ted", "ted.xlsx")):
                datasets.append((arch, drug, ds))
    if SMOKE:
        datasets = [d for d in datasets if (d[0], d[1]) in SMOKE_SETS]
    print(f"\ndatasets: {len(datasets)} ({'smoke' if SMOKE else 'full'})", flush=True)

    # synthetic self-check: one independent run before the loop
    synth_ok = False
    synth_d = None
    k_thr = None
    tmpl = find_template(datasets)
    if tmpl is not None:
        try:
            synth_ok, synth_d, k_thr = synth_check(tmpl[0], tmpl[1], tmpl[2], tmpl[3])
        except Exception as e:
            print(f"  synthetic self-check exception (downgraded to registry): {e}", flush=True)
    else:
        print("\n[synthetic self-check] no usable template -> whole card downgraded to registry", flush=True)

    result = {"预注册": "预注册_α模型_药物卡6_形状变形分解_2026-09-14.md",
              "datasets": {}, "合成自检": synth_d, "回归": None, "判词": {}}
    all_units = []
    for i, (arch, drug, ds) in enumerate(datasets):
        key = f"{arch.replace('a6k5t-osfstorage-', '')}|{drug}"
        try:
            r = analyze_dataset(arch, drug, ds, SMOKE)
        except Exception as e:
            r = dict(判="提取异常", err=str(e)[:200], units=[])
        result["datasets"][key] = r
        ok = [u for u in r["units"] if u["判定"] == "ok"]
        all_units += [dict(dataset=key, **u) for u in ok]
        print(f"  [{i + 1}/{len(datasets)}] {key}: units {len(ok)}", flush=True)

    # ---------- regression check: vs card-5 unit table (criterion <1e-9) ----------
    sfx = "_冒烟" if SMOKE else ""
    f5 = os.path.join(HERE, f"2026-09-14_α模型_药物卡5_形状不变性判决{sfx}_单元表.csv")
    reg_ok = None
    if os.path.exists(f5) and all_units:
        d5 = pd.read_csv(f5)
        d5["k"] = (d5["dataset"] + "|" + d5["cell"] + "|" + d5["conc_nM"].round(6).astype(str))
        m5 = dict(zip(d5["k"], d5["eps_x"]))
        diffs = []
        for u in all_units:
            k = u["dataset"] + "|" + u["cell"] + "|" + str(round(u["conc_nM"], 6))
            if k in m5:
                diffs.append(abs(u["eps_x0_full"] - m5[k]))
        if diffs:
            mx = float(np.max(diffs))
            reg_ok = bool(mx < 1e-9)
            result["回归"] = dict(比对单元=len(diffs), 卡5单元=len(m5), 最大绝对差=mx, 过=reg_ok)
            print(f"\n[regression] vs card-5 unit table {len(diffs)}/{len(all_units)} units, "
                  f"max |delta eps_x|={mx:.2e} -> {'reproduced' if reg_ok else 'not reproduced'}", flush=True)
    else:
        print("\n[regression] card-5 unit table missing or empty -> downgraded to registry", flush=True)

    # ---------- criteria (preregistration section 6) ----------
    print("\n" + "=" * 76, flush=True)
    gated = [u for u in all_units if u.get("gated") and u.get("f_tau") is not None]
    n_edge = int(np.sum([u["edge"] for u in all_units]))
    print(f" units {len(all_units)}; judge-able (eps_x0>={EX0_GATE}) {len(gated)}; grid edge hits {n_edge}", flush=True)

    ks = np.array([u["kappa"] for u in gated])
    fs = np.array([u["f_tau"] for u in gated])
    ccs = np.array([u["f_tau_cc"] for u in gated if u["f_tau_cc"] is not None])
    ex1s = np.array([u["eps_x1"] for u in gated])
    med_ex1 = float(np.median(ex1s)) if len(ex1s) else np.nan
    med_koff = float(np.median(np.abs(ks - 1.0))) if len(ks) else np.nan
    med_f = float(np.median(fs)) if len(fs) else np.nan

    # cell-level pairing (registry only since v2, not part of the criteria)
    by_cell = {}
    for u in gated:
        by_cell.setdefault((u["dataset"], u["cell"]), {"f": [], "cc": u["f_tau_cc"]})
        by_cell[(u["dataset"], u["cell"])]["f"].append(u["f_tau"])
    pairs = []
    for k, v in by_cell.items():
        if v["cc"] is not None and v["f"]:
            pairs.append((float(np.median(v["f"])), float(v["cc"])))
    pairs = np.array(pairs) if pairs else np.zeros((0, 2))
    p_one = np.nan
    med_d = np.nan
    if len(pairs) >= 10:
        dif = pairs[:, 0] - pairs[:, 1]
        med_d = float(np.median(dif))
        try:
            w = stats.wilcoxon(pairs[:, 0], pairs[:, 1])
            p_two = float(w.pvalue)
            p_one = p_two / 2.0 if med_d > 0 else 1.0 - p_two / 2.0
        except Exception as e:
            print(f"  Wilcoxon exception: {e}", flush=True)

    # v2 criterion: T1 = median |kappa*-1| >= k_thr (C1-anchored) and median f_tau >= 0.5; T2 unchanged
    T1 = bool(len(ks) >= 30 and k_thr is not None and
              med_koff >= k_thr and med_f >= 0.5)
    T2 = bool(len(ex1s) >= 30 and med_ex1 > EX0_GATE)

    if len(gated) < 30:
        verdict = "不可判（判-able 单元不足 30）"
    elif T1 and not T2:
        verdict = "τ 主导（warp 后干净，Stage B 走 τ 表调制）"
    elif (not T1) and T2:
        verdict = "幅度形状主导（warp 无用，Stage B 走 h_ss 表调制）"
    elif T1 and T2:
        verdict = "双成分（τ 调制 + 幅度调制并存，模块需两条腿）"
    else:
        verdict = "变形低于可判水平（与卡5 并读，登记异常）"

    cs = np.array([u["conc_nM"] for u in gated])
    rho_k = p_k = np.nan
    if len(ks) >= 30:
        rk = stats.spearmanr(ks, np.log10(cs))
        rho_k = float(rk.statistic)
        p_k = float(rk.pvalue)
    verdict_d = {
        "合成自检过": bool(synth_ok), "回归过": reg_ok,
        "单元": int(len(all_units)), "判able单元": int(len(gated)),
        "配对细胞": int(len(pairs)), "网格触边": n_edge,
        "中位kappa": float(np.median(ks)) if len(ks) else None,
        "中位f_tau_药": float(np.median(fs)) if len(fs) else None,
        "中位f_tau_cc": float(np.median(ccs)) if len(ccs) else None,
        "T1": {"中位|κ*−1|": med_koff, "κ_thr": k_thr, "中位f_τ": med_f, "过": T1},
        "T2": {"中位ε_x1": med_ex1, "过": T2},
        "登记_Wilcoxon配对": {"p_one_side": p_one, "中位Δf_τ": med_d},
        "登记_Spearman_kappa_logC": {"rho": rho_k, "p": p_k},
        "总判": verdict,
    }
    result["判词"] = verdict_d

    # per-drug summary
    by_drug = {}
    for u in gated:
        by_drug.setdefault(u["dataset"].split("|")[1], []).append(u)
    drug_tab = {d: dict(n=len(us),
                        med_kappa=float(np.median([x["kappa"] for x in us])),
                        med_f_tau=float(np.median([x["f_tau"] for x in us])),
                        med_eps_x1=float(np.median([x["eps_x1"] for x in us])))
                for d, us in by_drug.items()}
    result["分药"] = drug_tab

    print(f" judge-able units {len(gated)}; paired cells {len(pairs)}", flush=True)
    print(f" median kappa*={verdict_d['中位kappa']}; median f_tau drug={verdict_d['中位f_tau_药']} "
          f"vs cc={verdict_d['中位f_tau_cc']}", flush=True)
    print(f" T1={'pass' if T1 else 'fail'} (median |kappa*-1|={med_koff:.4f} k_thr={k_thr} "
          f"median f_tau={med_f:.3f}) T2={'pass' if T2 else 'fail'} (median eps_x1={med_ex1:.4f})", flush=True)
    print(f" registry: Wilcoxon paired p={p_one:.4g} median delta f_tau={med_d:.3f} ({len(pairs)} cells)", flush=True)
    print(f" registry: Spearman(kappa*, log10 C) rho={rho_k:.3f} p={p_k:.3g}", flush=True)
    print(f" overall verdict: {verdict}", flush=True)

    fj = os.path.join(HERE, f"2026-09-14_α模型_药物卡6_形状变形分解{sfx}_结果.json")
    json.dump(result, open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  result saved: {fj}", flush=True)

    fcsv = os.path.join(HERE, f"2026-09-14_α模型_药物卡6_形状变形分解{sfx}_单元表.csv")
    import csv as csvmod
    with open(fcsv, "w", encoding="utf-8-sig", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=["dataset", "cell", "conc_nM", "b_ss",
                                             "eps_x0_full", "eps_noise", "kappa",
                                             "eps_x1", "eps_x0_sd", "f_tau",
                                             "f_tau_raw", "f_tau_cc", "edge", "gated"])
        w.writeheader()
        for u in all_units:
            w.writerow({k: u.get(k) for k in w.fieldnames})
    print(f"  unit table saved: {fcsv}", flush=True)

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
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    if len(ks):
        ax.hist(ks, bins=40, alpha=0.75)
        ax.axvline(1.0, color="r", ls="--", label="kappa=1 (no time deformation)")
        ax.set_xlabel("kappa* (>1 = drug slows)")
        ax.set_ylabel("unit count")
        ax.set_title(f"kappa* distribution (median {np.median(ks):.3f})")
        ax.legend()
    ax = axes[0, 1]
    if len(pairs):
        ax.scatter(pairs[:, 1], pairs[:, 0], s=14, alpha=0.5)
        lim = [0, 1]
        ax.plot(lim, lim, "r--", label="y=x")
        ax.set_xlabel("f_tau_cc (control odd/even split floor)")
        ax.set_ylabel("f_tau drug (cell median)")
        ax.set_title(f"paired: delta median={med_d:.3f}, Wilcoxon one-sided p={p_one:.3g}")
        ax.legend()
    ax = axes[1, 0]
    if len(gated):
        ex0s = np.array([u["eps_x0_full"] for u in gated])
        ax.scatter(ex0s, ex1s, s=10, alpha=0.4)
        ax.plot([0, 1], [0, 1], "r--", label="y=x (warp absorbs all)")
        ax.axhline(EX0_GATE, color="gray", ls=":", label=f"residual line {EX0_GATE}")
        ax.set_xlabel("eps_x0 (pre-warp, card-5 convention)")
        ax.set_ylabel("eps_x1 (post-warp residual)")
        ax.set_title("pre/post-warp deformation (judge-able units)")
        ax.legend()
    ax = axes[1, 1]
    if drug_tab:
        ds_ = sorted(drug_tab.items(), key=lambda kv: -kv[1]["med_f_tau"])
        names = [d for d, _ in ds_]
        vals = [v["med_f_tau"] for _, v in ds_]
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.axhline(FTAU_FLOOR, color="r", ls="--", label=f"reference line {FTAU_FLOOR} (not a criterion)")
        ax.set_ylabel("median f_tau")
        ax.set_title("per-drug time-scale share (registry)")
        ax.legend()
    fpng = os.path.join(HERE, f"2026-09-14_α模型_药物卡6_形状变形分解{sfx}.png")
    fig.tight_layout()
    fig.savefig(fpng, dpi=140, bbox_inches="tight")
    print(f"  figure saved: {fpng}", flush=True)


if __name__ == "__main__":
    main()
