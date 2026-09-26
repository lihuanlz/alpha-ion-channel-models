# 2026-09-14_alpha-model_drug-card5_shape-invariance_verdict.py
# ============================================================================
# Drug card 5 · shape-invariance verdict (preregistered verdict card)
# Preregistration: 结果\预注册_α模型_药物卡5_形状不变性判决_2026-09-14.md (criteria pinned before the run)
#
# Hypothesis: pure block -- the drug only scales the amplitude and does not touch gating time constants (current shape invariant).
# Method: same cell, last 5 sweeps of the control segment vs last 5 sweeps of each concentration segment; baseline subtraction -> sweep average -> Ramp-window peak-amplitude normalization
#   -> r_shape (Pearson) and eps (relative shape residual).
# Gates: |A_ctrl|>=50pA (same unit normalization as card-3 v2), 0.2<=b_ss<=0.9, segment sweeps >=5; at most 3 cells per dataset.
# Criteria: G1 pooled median eps_x<=eps_x_thr (v2 main metric, after noise-floor subtraction; r/eps registered);
#   G2 Spearman(eps_x, log10 C) drift test;
#   synthetic self-check C1 pure scaling all pass / C2 time stretch all caught, eps_x_thr=max(3x C1 median eps_x, 0.03).
# v3 fix: cursor STIMEU/ETIMEU units mixed (lab1/lab2=ms, lab3/4/5=s) unified to seconds;
#   zipfile single handle; synthetic self-check pulled out of the dataset loop to run once.
# Run: Spyder %runfile '...py' --wdir. SMOKE=1 smoke: 3 datasets x 1 cell, metric verbatim identical.
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
R_LINE = 0.98
RNG = np.random.default_rng(20260914)


def conc_nM(conc, unit):
    u = (unit or "").strip().lower()
    return conc * 1000.0 if (u.startswith("u") or u.startswith("µ")) else conc


# ---------- ted.xlsx ----------
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
    # v3 fix: cursors carry STIMEU/ETIMEU unit columns; lab1/lab2 are ms, lab3/4/5 are s (proven on record)
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
    """One cell -> (A_ctrl, control last-5-sweep TRACENUMs, [(conc, last-5-sweep TRACENUMs, b_ss)]). Unit normalization same as card-3 v2."""
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


# ---------- raw csv ----------
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
        # v3 fix: single handle (the original code opened ZipFile twice and leaked a handle)
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


# ---------- shape metrics ----------
def mean_trace(t, traces, bwin):
    sub = []
    for i in traces:
        m = (t >= bwin[0]) & (t <= bwin[1])
        b0 = np.nanmean(i[m]) if m.sum() >= 3 else 0.0
        sub.append(i - b0)
    return np.nanmean(sub, axis=0)


def shape_metrics(t, ctrl, drug, bwin, rwin):
    """Return (r_shape, eps, eps_noise, eps_x). v2: noise floor subtracted.
    ε_noise = sqrt(σ_d²/A_d² + σ_c²/A_c²)；ε_x = sqrt(max(ε²−ε_noise²,0))。"""
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
    r = float(np.corrcoef(cn, dn)[0, 1])
    eps = float(np.sqrt(np.mean((dn - cn) ** 2)) / np.sqrt(np.mean(cn ** 2)))
    sg_c = float(np.nanstd(ctrl[mb])) if mb.sum() >= 3 else 0.0
    sg_d = float(np.nanstd(drug[mb])) if mb.sum() >= 3 else 0.0
    eps_noise = float(np.sqrt((sg_d / sd) ** 2 + (sg_c / sc) ** 2))
    eps_x = float(np.sqrt(max(eps ** 2 - eps_noise ** 2, 0.0)))
    return r, eps, eps_noise, eps_x


# ---------- synthetic self-check (control-anchored threshold) ----------
def ar1_noise(n, sigma, rho=0.33, rng=RNG):
    e = rng.normal(0, sigma * np.sqrt(1 - rho * rho), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def warp_tail(y, t, rwin, factor=1.2):
    """Stretch the post-peak time axis within the Ramp window by xfactor (simulating a slowed-deactivation type warp)."""
    m = np.where((t >= rwin[0]) & (t <= rwin[1]))[0]
    seg = y[m]
    pk = int(np.argmax(np.abs(seg)))
    out = seg[:pk + 1].tolist()
    src = np.arange(pk, len(seg)) / factor
    out += list(np.interp(src, np.arange(len(seg)), seg)[1:])
    y2 = y.copy()
    y2[m[:len(out)]] = out
    return y2


def synth_check(t, tmpl, bwin, rwin):
    print("\n[synthetic self-check]", flush=True)
    mb = (t >= bwin[0]) & (t <= bwin[1])
    sigma = float(np.nanstd(tmpl[mb])) if mb.sum() >= 3 else 1.0
    e1, e2 = [], []
    for _ in range(20):
        c1 = 0.5 * tmpl + ar1_noise(len(tmpl), sigma)
        r1 = shape_metrics(t, tmpl, c1, bwin, rwin)
        c2 = warp_tail(0.5 * tmpl, t, rwin) + ar1_noise(len(tmpl), sigma)
        r2 = shape_metrics(t, tmpl, c2, bwin, rwin)
        if r1:
            e1.append(r1)
        if r2:
            e2.append(r2)
    e1 = np.array(e1)
    e2 = np.array(e2)
    ex_thr = max(3.0 * float(np.median(e1[:, 3])), 0.03)
    c1_ok = bool(np.all(e1[:, 3] <= ex_thr))
    c2_ok = bool(np.all(e2[:, 3] > ex_thr))
    print(f"  C1 pure scaling: r median={np.median(e1[:, 0]):.4f} eps_x median={np.median(e1[:, 3]):.4f}"
          f" -> {'all pass' if c1_ok else 'false verdicts present'}", flush=True)
    print(f"  C2 gating warp (post-peak x1.2): r median={np.median(e2[:, 0]):.4f} "
          f"eps_x median={np.median(e2[:, 3]):.4f} -> {'all caught' if c2_ok else 'some escaped'}", flush=True)
    print(f"  eps_x_thr = {ex_thr:.4f} (control anchored)", flush=True)
    ok = c1_ok and c2_ok
    print(f"  synthetic self-check {'pass' if ok else 'fail -- whole card downgraded to registry'}", flush=True)
    return ok, ex_thr


def analyze_dataset(arch, drug, ds_dir, eps_thr, smoke):
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
        need = sorted(set(ctrl_tr) | {tn for _, trs, _ in segs for tn in trs})
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
        # noise-floor diagnostic: control odd/even split
        r_cc = None
        if len(ctrs) >= 4:
            odd = mean_trace(t, ctrs[0::2], bwin)
            even = mean_trace(t, ctrs[1::2], bwin)
            rcc = shape_metrics(t, odd, even, bwin, rwin)
            if rcc:
                r_cc = rcc[0]
        for C, trs, b in segs:
            drs = [tr[N] for N in trs if N in tr]
            if len(drs) < 3:
                continue
            drug_tr = mean_trace(t, drs, bwin)
            sm = shape_metrics(t, ctrl, drug_tr, bwin, rwin)
            if sm is None:
                continue
            units.append(dict(cell=cid, conc_nM=C, b_ss=b, r_shape=sm[0], eps=sm[1],
                              eps_noise=sm[2], eps_x=sm[3], r_cc=r_cc, 判定="ok"))
    ok = [u for u in units if u["判定"] == "ok"]
    return dict(判="ok" if ok else "无合格单元", units=units,
                med_r=float(np.median([u["r_shape"] for u in ok])) if ok else None)


def find_template(datasets):
    """v3: the synthetic self-check template lookup is factored into an independent function, run once before the dataset loop.
    Scan datasets/cells in order and take the control mean waveform of the first dataset with a valid unit.
    Returns (t, tmpl, bwin, rwin) or None."""
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


def main():
    print("=" * 76, flush=True)
    print(" drug card 5 · shape-invariance verdict (sealing the pure-block hypothesis)", flush=True)
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

    # v3: synthetic self-check pulled out of the dataset loop, run once independently before the loop.
    # the original implementation was nested in the loop with exceptions swallowed by except -> retried and spammed for every dataset;
    # under the cursor-unit bug, lab1/lab2 coming up empty was amplified by exactly this path.
    eps_thr = None
    synth_ok = False
    tmpl = find_template(datasets)
    if tmpl is not None:
        try:
            synth_ok, eps_thr = synth_check(tmpl[0], tmpl[1], tmpl[2], tmpl[3])
        except Exception as e:
            print(f"  synthetic self-check exception (downgraded to registry): {e}", flush=True)
            synth_ok, eps_thr = False, None
    else:
        print("\n[synthetic self-check] no usable template (no valid unit at all) -> whole card downgraded to registry", flush=True)
    result = {"预注册": "预注册_α模型_药物卡5_形状不变性判决_2026-09-14.md",
              "datasets": {}, "eps_thr": eps_thr, "合成自检": bool(synth_ok), "判词": {}}
    all_units = []
    for i, (arch, drug, ds) in enumerate(datasets):
        key = f"{arch.replace('a6k5t-osfstorage-', '')}|{drug}"
        try:
            r = analyze_dataset(arch, drug, ds, eps_thr, SMOKE)
        except Exception as e:
            r = dict(判="提取异常", err=str(e)[:200], units=[])
        result["datasets"][key] = r
        ok = [u for u in r["units"] if u["判定"] == "ok"]
        all_units += [dict(dataset=key, **u) for u in ok]
        med = f"r median={r['med_r']:.4f}" if r.get("med_r") else r["判"]
        print(f"  [{i + 1}/{len(datasets)}] {key}: units {len(ok)} {med}", flush=True)

    # ---------- criteria (v2: eps_x main metric, r/eps registered) ----------
    rs = np.array([u["r_shape"] for u in all_units])
    es = np.array([u["eps"] for u in all_units])
    exs = np.array([u["eps_x"] for u in all_units])
    ens = np.array([u["eps_noise"] for u in all_units])
    cs = np.array([u["conc_nM"] for u in all_units])
    rccs = np.array([u["r_cc"] for u in all_units if u["r_cc"] is not None])
    g1 = bool(synth_ok and len(exs) >= 30 and np.median(exs) <= (eps_thr or np.inf))
    rho_c, p_c = (np.nan, np.nan)
    if len(exs) >= 30:
        rho_c, p_c = stats.spearmanr(exs, np.log10(cs))
        p_c = p_c / 2.0 if rho_c > 0 else 1.0 - p_c / 2.0
    g2_no_drift = bool(len(exs) >= 30 and not (rho_c > 0 and p_c < 0.01))
    if g1 and g2_no_drift:
        verdict = "纯阻断假设封卷（形状不变且无浓度漂移）"
    elif g1:
        verdict = "部分成立：形状总体不变但有浓度相关变形（定位登记）"
    else:
        verdict = "纯阻断否掉或证据不足（按药/段定位）"
    verdict_d = {
        "合成自检": bool(synth_ok), "eps_x_thr": eps_thr,
        "合格单元": int(len(exs)),
        "G1": {"med_eps_x": float(np.median(exs)) if len(exs) else None,
               "med_eps_noise": float(np.median(ens)) if len(ens) else None,
               "过": g1},
        "G2_浓度漂移": {"rho": float(rho_c), "p_one_side": float(p_c),
                       "判": "无漂移" if g2_no_drift else "有漂移"},
        "登记_中位r_shape": float(np.median(rs)) if len(rs) else None,
        "登记_中位eps_raw": float(np.median(es)) if len(es) else None,
        "噪声地板r_cc中位": float(np.median(rccs)) if len(rccs) else None,
        "总判": verdict,
    }
    result["判词"] = verdict_d

    # per-drug summary
    by_drug = {}
    for u in all_units:
        by_drug.setdefault(u["dataset"].split("|")[1], []).append(u)
    drug_tab = {d: dict(n=len(us), med_r=float(np.median([x["r_shape"] for x in us])),
                        med_eps_x=float(np.median([x["eps_x"] for x in us])))
                for d, us in by_drug.items()}
    result["分药"] = drug_tab

    print("\n" + "=" * 76, flush=True)
    print(f" valid units {len(exs)}; median eps_x={verdict_d['G1']['med_eps_x']}"
          f"(eps_x_thr={eps_thr}; noise-floor eps_noise median={verdict_d['G1']['med_eps_noise']})", flush=True)
    print(f" registry: median r_shape={verdict_d['登记_中位r_shape']} "
          f"median eps_raw={verdict_d['登记_中位eps_raw']} r_cc median={verdict_d['噪声地板r_cc中位']}", flush=True)
    print(f" G1={'pass' if g1 else 'fail'}  G2={verdict_d['G2_浓度漂移']['判']}"
          f"（ρ={rho_c:.3f} p={p_c:.4f}）", flush=True)
    print(f" overall verdict: {verdict}", flush=True)

    sfx = "_冒烟" if SMOKE else ""
    fj = os.path.join(HERE, f"2026-09-14_α模型_药物卡5_形状不变性判决{sfx}_结果.json")
    json.dump(result, open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  results saved: {fj}", flush=True)

    # CSV
    fcsv = os.path.join(HERE, f"2026-09-14_α模型_药物卡5_形状不变性判决{sfx}_单元表.csv")
    with open(fcsv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv_writer = None
        import csv as csvmod
        w = csvmod.DictWriter(f, fieldnames=["dataset", "cell", "conc_nM", "b_ss",
                                             "r_shape", "eps", "eps_noise", "eps_x", "r_cc"])
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    if len(rs):
        ax.hist(rs, bins=40, alpha=0.7, label="drug vs control r_shape")
    if len(rccs):
        ax.hist(rccs, bins=40, alpha=0.7, label="control odd/even split r_cc (noise floor)")
    ax.axvline(R_LINE, color="r", ls="--", label="criterion 0.98")
    ax.set_xlabel("shape correlation r")
    ax.set_ylabel("unit count")
    ax.legend()
    ax.set_title(f"shape-invariance distribution (median {np.median(rs):.4f})" if len(rs) else "no valid units")
    ax = axes[1]
    if len(exs):
        ax.scatter(cs, exs, s=10, alpha=0.4, label="eps_x (noise subtracted)")
        ax.scatter(cs, ens, s=10, alpha=0.3, label="eps_noise (noise floor)")
        ax.set_xscale("log")
        thr_v = eps_thr if eps_thr is not None else 0.0
        ax.axhline(thr_v, color="r", ls="--", label=f"ε_x_thr={thr_v:.3f}")
        ax.set_xlabel("concentration (nM)")
        ax.set_ylabel("shape residual")
        ax.set_title(f"eps_x ~ concentration (Spearman rho={rho_c:.2f} p={p_c:.4f})")
        ax.legend()
    fpng = os.path.join(HERE, f"2026-09-14_α模型_药物卡5_形状不变性判决{sfx}.png")
    fig.tight_layout()
    fig.savefig(fpng, dpi=140, bbox_inches="tight")
    print(f"  figure saved: {fpng}", flush=True)


if __name__ == "__main__":
    main()
