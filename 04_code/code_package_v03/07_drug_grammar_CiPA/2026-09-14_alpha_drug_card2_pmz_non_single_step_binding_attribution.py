# 2026-09-14_alpha-model_drug-card2_pmz-non-single-step-binding_attribution.py
# ============================================================================
# Drug card 2: pimozide non-single-step binding attribution (criteria pinned before the run, see
#   预注册_α模型_药物卡2_pmz非单步结合归因_2026-09-14.md, v2 amendment on record)
#
# Mechanism candidates: M1 multi-step binding / M2 perfusion-limited / M3 true double-exponential wash-in
# Data: ted.xlsx ResultsWide per-sweep Ramp (the large csv is not read)
# Gate A trajectory shape: null model = anchored-bss single exponential + linear drift, parametric-bootstrap envelope whitening; alternative = double exponential
# Gate B concentration dependence: endpoint-group Welch t (p<0.01 provable); Gate D same-cell E-4031 rate ratio R_E
# Controls C1 single-step / C2 two-step pre-equilibration (Km=0.01) / C3 perfusion / C4 double exponential, 20 MC each
#
# Run: Spyder %runfile '...py' --wdir. SMOKE=1 smoke: 4 cells/drug + 5 MC.
# ============================================================================
import os, json
import numpy as np
import openpyxl
from scipy.optimize import curve_fit
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/结果/"
        "hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包")
SMOKE = os.environ.get("SMOKE", "0") == "1"

POOLS = {
    "pimozide": dict(dir="a6k5t-osfstorage-pimozide-archivelab4",
                     drug="pimozide", ctrl="control", conc_to_nM=1.0,
                     exclude_prefix=(), ref_min_conc=1.0),
    "dofetilide": dict(dir="a6k5t-osfstorage-dofetilide-archivelab3",
                       drug="dofetilide", ctrl="vehicle-control", conc_to_nM=1000.0,
                       exclude_prefix=("57_A", "61_E"), ref_min_conc=3.0),
}

# ---- criteria (pinned) ----
A_WHITE_HI, A_WHITE_LO = 2 / 3, 1 / 3
A_IMPROVE = 0.5
B_P = 0.01
D_HI, D_LO = 3.0, 2.0
MIN_SWEEPS, MIN_BSS_E = 10, 0.5
N_ENV = 50 if SMOKE else 200       # gate-A parametric-bootstrap count
N_MC = 5 if SMOKE else 20          # control MC
CTRL_REC = 0.8                     # control recovery-rate threshold
rng = np.random.default_rng(20260914)


# ------------------------------------------------------------ data loading
def load_rw(tedx):
    wb = openpyxl.load_workbook(tedx, read_only=True)
    rw = [list(r) for r in wb["ResultsWide"].iter_rows(values_only=True)]
    wb.close()
    h = rw[0]
    ix = {k: h.index(k) for k in ("CELLID", "TRACENUM", "ELTIME", "LIQUID", "CONC", "Ramp")}
    cells = {}
    for r in rw[1:]:
        if r[ix["CELLID"]] is None or r[ix["Ramp"]] is None:
            continue
        try:
            tn = int(r[ix["TRACENUM"]]); et = float(r[ix["ELTIME"]]) / 1000.0
            rv = float(r[ix["Ramp"]]); liq = str(r[ix["LIQUID"]]).lower()
            conc = float(r[ix["CONC"]]) if r[ix["CONC"]] is not None else 0.0
        except (TypeError, ValueError):
            continue
        cells.setdefault(str(r[ix["CELLID"]]), []).append((tn, et, liq, conc, rv))
    return cells


def cell_traj(rows, cfg):
    tn0 = sorted(r for r in rows if r[2] == cfg["ctrl"])
    drg = sorted(r for r in rows if r[2] == cfg["drug"].lower())
    e40 = sorted(r for r in rows if r[2].startswith("e-4031"))
    if not tn0 or not drg:
        return None
    A_ctrl = float(np.median([r[4] for r in tn0[-5:]]))
    if abs(A_ctrl) < 50.0:
        return None
    tc = np.array([r[1] for r in tn0]); bc = np.array([r[4] for r in tn0]) / A_ctrl
    if len(tc) >= 5:
        cf = np.polyfit(tc - tc[0], bc, 1)
        rc = bc - np.polyval(cf, tc - tc[0])
        sig_b = float(np.std(rc))
        d0 = float(np.sum(rc * rc))
        rho_b = float(np.clip(np.sum(rc[:-1] * rc[1:]) / d0, 0.0, 0.95)) if d0 > 0 else 0.0
    else:
        sig_b = float(np.std(bc)); rho_b = 0.0
    t = np.array([r[1] - drg[0][1] for r in drg])
    b = 1.0 - np.array([r[4] for r in drg]) / A_ctrl
    conc_nM = float(np.median([r[3] for r in drg])) * cfg["conc_to_nM"]
    out = dict(t=t, b=b, conc_nM=conc_nM, sig_b=max(sig_b, 1e-4), rho_b=rho_b)
    if len(e40) >= MIN_SWEEPS:
        A_ss = float(np.median([r[4] for r in drg[-5:]]))
        if abs(A_ss) > 20:
            te = np.array([r[1] - e40[0][1] for r in e40])
            be = 1.0 - np.array([r[4] for r in e40]) / A_ss
            out["e4031"] = (te, be)
    return out


# ------------------------------------------------------------ models and fast fitting
KGRID = np.logspace(-4, 0, 30)  # k grid (/s)


def null_fit(t, b, refine=True):
    """Null model: b_ss anchored to the median of the last 5 sweeps; k grid + d linear LS (refine=True adds a curve_fit polish)."""
    bss = float(np.median(b[-5:]))
    tt2 = float(np.sum(t * t))
    E = 1.0 - np.exp(-np.outer(KGRID, t))          # (30, n)
    U = bss * E
    d_hat = (float(np.sum(b * t)) - U @ t) / tt2
    R = b - U - np.outer(d_hat, t)
    rss = np.sum(R * R, axis=1)
    j = int(np.argmin(rss))
    k0, d0 = float(KGRID[j]), float(d_hat[j])
    if refine:
        try:
            def f(tt, k, d):
                return bss * (1.0 - np.exp(-k * tt)) + d * tt
            p, _ = curve_fit(f, t, b, p0=[k0, d0],
                             bounds=([1e-5, -0.1], [2.0, 0.1]), maxfev=10000)
            m = f(t, *p)
            return float(p[0]), float(p[1]), bss, b - m, m
        except Exception:
            pass
    m = bss * E[j] + d0 * t
    return k0, d0, bss, b - m, m


def biexp_fit(t, b):
    def f(tt, b1, k1, b2, k2):
        return b1 * (1 - np.exp(-k1 * tt)) + b2 * (1 - np.exp(-k2 * tt))
    try:
        bm = max(b.max(), 0.05)
        p, _ = curve_fit(f, t, b, p0=[0.5 * bm, 0.02, 0.5 * bm, 0.002],
                         bounds=([0, 1e-4, 0, 1e-4], [1.5, 1.0, 1.5, 1.0]), maxfev=30000)
        m = f(t, *p)
        return p, b - m, m
    except Exception:
        return None, None, None


def acf5(x):
    x = x - x.mean()
    d = float(np.sum(x * x))
    if d <= 0:
        return np.zeros(5)
    return np.array([np.sum(x[:-k] * x[k:]) / d for k in range(1, 6)])


_env_cache = {}


def noise_env(n, rho=0.0):
    """AR(1) noise envelope (95% ACF of linearly detrended residuals), for the double-exponential whiteness test."""
    key = (n, round(rho, 2))
    if key not in _env_cache:
        sims = np.zeros((2000, 5))
        tt = np.arange(n, dtype=float)
        for i in range(2000):
            e = ar1_noise(n, 1.0, rho, rng)
            cf = np.polyfit(tt, e, 1)
            sims[i] = np.abs(acf5(e - np.polyval(cf, tt)))
        _env_cache[key] = np.percentile(sims, 95, axis=0)
    return _env_cache[key]


def ar1_noise(n, sig, rho, rng_):
    """AR(1) noise with marginal std=sig."""
    e = np.zeros(n)
    si = sig * np.sqrt(max(1e-8, 1 - rho * rho))
    x = 0.0
    for i in range(n):
        x = rho * x + rng_.normal(0, si)
        e[i] = x
    return e


def white_boot(t, b, sig, rho):
    """Gate-A null-model parametric-bootstrap whitening (AR(1) noise). Returns (k, bss, white, viol, r1)."""
    k, d, bss, res, m = null_fit(t, b, refine=True)
    n = len(t)
    sims = np.zeros((N_ENV, 5))
    for i in range(N_ENV):
        y = m + ar1_noise(n, sig, rho, rng)
        _, _, _, rs, _ = null_fit(t, y, refine=False)
        sims[i] = np.abs(acf5(rs))
    env = np.percentile(sims, 95, axis=0)
    a = np.abs(acf5(res))
    viol = int(np.sum(a > env))
    return k, bss, viol <= 2, viol, float(a[0])


# ------------------------------------------------------------ gates
def gate_A(cells):
    per = []
    r1_pairs = []  # (r1_drug, r1_e4031) for gate A2
    for c in cells:
        k, bss, white, viol, r1 = white_boot(c["t"], c["b"], c["sig_b"], c.get("rho_b", 0.0))
        rec = dict(k=k, bss=bss, white=bool(white), viol=viol, r1=r1, conc_nM=c["conc_nM"])
        if not white:
            _, res2, _ = biexp_fit(c["t"], c["b"])
            if res2 is not None:
                v2 = int(np.sum(np.abs(acf5(res2)) > noise_env(len(c["t"]), c.get("rho_b", 0.0))))
                rec["bi_white"] = bool(v2 <= 2)
            else:
                rec["bi_white"] = False
        if "e4031" in c:
            te, be = c["e4031"]
            _, _, _, _, m_e = null_fit(te, be)
            r1_e = float(np.abs(acf5(be - m_e)[0:1])[0])
            r1_pairs.append((r1, r1_e))
            rec["r1_e4031"] = r1_e
        per.append(rec)
    n = len(per)
    wf = sum(p["white"] for p in per) / n if n else np.nan
    nw = [p for p in per if not p["white"]]
    imp = (sum(1 for p in nw if p.get("bi_white")) / len(nw)) if nw else np.nan
    if wf >= A_WHITE_HI:
        verdict = "单指数兼容"
    elif wf <= A_WHITE_LO and np.isfinite(imp) and imp >= A_IMPROVE:
        verdict = "多指数信号(M3)"
    elif wf <= A_WHITE_LO:
        verdict = "非指数结构(待门A2归因)"
    else:
        verdict = "中间登记"
    # Gate A2: drug segment vs same-cell E-4031 segment r1 control
    a2 = None
    if r1_pairs:
        md = float(np.median([p[0] for p in r1_pairs]))
        me = float(np.median([p[1] for p in r1_pairs]))
        if md > 1e-9:
            rat = me / md
            if rat >= 0.8:
                a2 = dict(r1_drug=md, r1_e=me, 比=rat, 判="装置级结构(轨迹门降级)")
            elif rat <= 0.5:
                a2 = dict(r1_drug=md, r1_e=me, 比=rat, 判="药特异结构(M3方向)")
            else:
                a2 = dict(r1_drug=md, r1_e=me, 比=rat, 判="中间登记")
    return dict(n=n, white_frac=float(wf), improve_frac=float(imp) if np.isfinite(imp) else None,
                判=verdict, A2=a2, per=per)


def gate_B(pts):
    C = np.array([p[0] for p in pts]); k = np.array([p[1] for p in pts])
    grids = sorted(set(np.round(C, 3)))
    if len(grids) < 3:
        return dict(判="档不足", n=len(C))
    lo = C <= grids[0] + 1e-9
    hi = C >= grids[-2] - 1e-9
    if lo.sum() < 2 or hi.sum() < 2:
        return dict(判="组不足", n=len(C))
    tstat, p2 = sst.ttest_ind(k[hi], k[lo], equal_var=False)
    p1 = float(p2 / 2) if tstat > 0 else 1.0
    verdict = "浓度依赖可证" if (p1 < B_P and k[hi].mean() > k[lo].mean()) else "不可证(饱和方向)"
    # reference observation: linear/saturated R2 and dBIC (not part of the verdict)
    n = len(C)
    A = np.vstack([C, np.ones_like(C)]).T
    coef, *_ = np.linalg.lstsq(A, k, rcond=None)
    rss_l = float(np.sum((k - A @ coef) ** 2))
    r2_l = 1.0 - rss_l / float(np.sum((k - k.mean()) ** 2))
    return dict(判=verdict, n=n, p_one_side=p1,
                k_lo=float(k[lo].mean()), k_hi=float(k[hi].mean()),
                n_lo=int(lo.sum()), n_hi=int(hi.sum()), R2_lin=r2_l)


# ------------------------------------------------------------ synthetic controls
def sim_cells(fam, concs, n_per, nsweeps, dt, sig, bss_fn, rho):
    out = []
    for C in concs:
        for _ in range(n_per):
            t = np.arange(nsweeps) * dt
            bss = bss_fn(C)
            if fam == "C1":
                k = 4.73e-4 * C + 4.03e-3; bb = bss * (1 - np.exp(-k * t))
            elif fam == "C2":
                k = 1e-3 + 4e-3 * C / (C + 0.001); bb = bss * (1 - np.exp(-k * t))
            elif fam == "C3":
                bb = bss * (1 - np.exp(-5e-3 * t))
            else:
                bb = 0.5 * bss * (1 - np.exp(-0.02 * t)) + 0.5 * bss * (1 - np.exp(-0.002 * t))
            out.append(dict(t=t, b=bb + ar1_noise(len(t), sig, rho, rng),
                            conc_nM=C, sig_b=sig, rho_b=rho))
    return out


def run_controls(concs, n_per, nsweeps, dt, sig, bss_fn, rho):
    out = {}
    for fam in ("C1", "C2", "C3", "C4"):
        oks = []
        lastA = lastB = None
        for _ in range(N_MC):
            cells = sim_cells(fam, concs, n_per, nsweeps, dt, sig, bss_fn, rho)
            gA = gate_A(cells)
            pts = [(c["conc_nM"], p["k"]) for c, p in zip(cells, gA["per"])]
            gB = gate_B(pts)
            lastA, lastB = gA, gB
            if fam == "C1":
                ok = gA["white_frac"] >= A_WHITE_HI and gB["判"] == "浓度依赖可证"
            elif fam in ("C2", "C3"):
                ok = gA["white_frac"] >= A_WHITE_HI and gB["判"] == "不可证(饱和方向)"
            else:
                ok = (gA["white_frac"] <= A_WHITE_LO and gA["improve_frac"] is not None
                      and gA["improve_frac"] >= A_IMPROVE)
            oks.append(ok)
        rec = float(np.mean(oks))
        out[fam] = dict(找回率=rec, 末轮白化=lastA["white_frac"], 末轮门B=lastB["判"],
                        归位=bool(rec >= CTRL_REC))
        print(f"  [control {fam}] recovery rate {rec * 100:.0f}%  final-round gate-A whitening {lastA['white_frac']:.2f} "
              f"gate-B {lastB['判']} -> {'landed' if out[fam]['归位'] else 'not landed'}", flush=True)
    return out


# ------------------------------------------------------------ main run
print("=" * 76, flush=True)
print(" drug card 2 · pmz non-single-step binding attribution v3 (AR(1) envelope / gate-A2 structural attribution / endpoint t / E-4031 reference)", flush=True)
print("=" * 76, flush=True)

result = {"预注册": "预注册_α模型_药物卡2_pmz非单步结合归因_2026-09-14.md",
          "判线": {"A白化": [A_WHITE_LO, A_WHITE_HI], "A改善": A_IMPROVE,
                   "B_p": B_P, "D比": [D_LO, D_HI]},
          "controls": {}, "pools": {}, "判词": []}

pools_data = {}
pools_full = {}
for name, cfg in POOLS.items():
    tedx = os.path.join(PACK, cfg["dir"], "subtracted", "ted", "ted.xlsx")
    raw = load_rw(tedx)
    cells = []
    for cid, rows in sorted(raw.items()):
        if any(cid.startswith(p) for p in cfg["exclude_prefix"]):
            continue
        ct = cell_traj(rows, cfg)
        if ct and len(ct["t"]) >= MIN_SWEEPS:
            ct["cell"] = cid
            cells.append(ct)
    print(f"\n=== pool {name}: cells {len(cells)}", flush=True)
    pools_full[name] = cells
    if SMOKE:
        by_c = {}
        for c in cells:
            by_c.setdefault(round(c["conc_nM"], 3), []).append(c)
        pools_data[name] = [c for v in by_c.values() for c in v[:1]]
    else:
        pools_data[name] = cells

# control anchors use the full pmz real grid (not truncated even under SMOKE, so control criteria are not contaminated by the smoke metric)
pmz = pools_full["pimozide"]
concs = sorted(set(round(c["conc_nM"], 3) for c in pmz))
nsw = int(np.median([len(c["t"]) for c in pmz]))
dts = float(np.median([np.median(np.diff(c["t"])) for c in pmz]))
sig_med = float(np.median([c["sig_b"] for c in pmz]))
rho_med = float(np.median([c["rho_b"] for c in pmz]))
bss_fn = lambda C: C / (C + 1.3)
n_per = max(2 if SMOKE else 1, int(round(len(pmz) / max(1, len(concs)))))
print(f"\n[controls] anchors: concentrations {concs} per band {n_per} sweeps {nsw} dt={dts:.1f}s sigma_b={sig_med:.4f} rho={rho_med:.2f}", flush=True)
result["controls"] = run_controls(concs, n_per, nsw, dts, sig_med, bss_fn, rho_med)
ctrl_ok = all(v["归位"] for v in result["controls"].values())
print(f"[controls] all four families landed = {ctrl_ok}", flush=True)

# real pools
for name, cells in pools_data.items():
    gA = gate_A(cells)
    pts = [(c["conc_nM"], p["k"]) for c, p in zip(cells, gA["per"])]
    gB = gate_B(pts)
    # gate D
    kE = []
    for c in cells:
        if "e4031" in c:
            te, be = c["e4031"]
            bss_e = float(np.median(be[-5:]))
            if bss_e >= MIN_BSS_E:
                k_e, _, _, _, _ = null_fit(te, be)
                kE.append(k_e)
    kref = [p["k"] for c, p in zip(cells, gA["per"]) if c["conc_nM"] >= POOLS[name]["ref_min_conc"]]
    gD = dict(n_E=len(kE))
    if kE and kref:
        kE_med = float(np.median(kE)); kref_med = float(np.median(kref))
        R_E = kE_med / kref_med
        gD.update(kE_med=kE_med, kref_med=kref_med, R_E=float(R_E),
                  判=("灌注否(M1方向)" if R_E >= D_HI else
                      ("灌注限速坐实(M2)" if R_E <= D_LO else "不可分登记")))
    else:
        gD["判"] = "缺席登记"
    result["pools"][name] = dict(A=gA, B=gB, D=gD)
    print(f"\n[{name}] gate A whitening {gA['white_frac']:.2f} improvement "
          f"{gA['improve_frac'] if gA['improve_frac'] is not None else float('nan'):.2f} -> {gA['判']}"
          f"\n      gate B {gB.get('判')} (p={gB.get('p_one_side', float('nan')):.4f} "
          f"lo={gB.get('k_lo', float('nan')):.4f} hi={gB.get('k_hi', float('nan')):.4f}）"
          f"\n      gate D {gD.get('判')} (kE={gD.get('kE_med', float('nan')):.4f}/s "
          f"kref={gD.get('kref_med', float('nan')):.4f}/s R_E={gD.get('R_E', float('nan')):.2f}）",
          flush=True)

# overall verdict (logic tree pinned)
for name in POOLS:
    p = result["pools"][name]
    if not ctrl_ok:
        verdict = "对照未全归位 -> 统计量作废，判词降级登记"
    elif "M3" in p["A"]["判"]:
        verdict = f"门A {p['A']['判']} -> M3 真双指数方向"
    elif p["A"]["判"] == "非指数结构(待门A2归因)":
        a2 = (p["A"].get("A2") or {}).get("判", "A2缺席")
        if a2 == "装置级结构(轨迹门降级)":
            verdict = (f"门A 非指数结构 + 门A2 装置级 -> 轨迹门降级；"
                       f"主判词由门B/门D 给出: 门B {p['B'].get('判')} / 门D {p['D'].get('判')}")
        elif a2 == "药特异结构(M3方向)":
            verdict = "门A 非指数结构 + 门A2 药特异 -> M3 方向（非双指数可描，慢蠕变登记）"
        else:
            verdict = f"门A 非指数结构 + 门A2 {a2} -> 登记"
    elif p["A"]["判"] == "单指数兼容" and p["B"].get("判") == "不可证(饱和方向)":
        verdict = f"单指数+浓度依赖不可证 -> 门D 裁决: {p['D'].get('判')}"
    elif p["A"]["判"] == "单指数兼容" and p["B"].get("判") == "浓度依赖可证":
        verdict = "单指数+浓度依赖可证 -> 单步兼容方向（pmz 出此则卡1 L2 复查登记）"
    else:
        verdict = f"登记: 门A {p['A']['判']} / 门B {p['B'].get('判')} / 门D {p['D'].get('判')}"
    result["判词"].append(f"{name}: {verdict}")

print("\n" + "=" * 76, flush=True)
for s in result["判词"]:
    print(" ", s, flush=True)

sfx = "_冒烟" if SMOKE else ""
fj = os.path.join(HERE, f"2026-09-14_α模型_药物卡2_pmz非单步结合归因{sfx}_结果.json")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


json.dump(_clean(result), open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("  results saved:", fj, flush=True)

# ------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for pi, (name, cells) in enumerate(pools_data.items()):
    a = ax[0][pi]
    for c in cells:
        a.plot(c["t"], c["b"], lw=0.5, alpha=0.35)
    a.set_title(f"{name} wash-in trajectory family"); a.set_xlabel("t (s)"); a.set_ylabel("block b")
    a.grid(alpha=0.3)
    b2 = ax[1][pi]
    p = result["pools"][name]
    per = p["A"]["per"]
    kk = [q["k"] for q in per]; cc = [q["conc_nM"] for q in per]
    b2.plot(cc, kk, "o", ms=5)
    b2.set_xscale("log")
    b2.set_title(f"{name} gateB:{p['B'].get('判')} | gateA whitening {p['A']['white_frac']:.2f} | gateD:{p['D'].get('判')}")
    b2.set_xlabel("C (nM)"); b2.set_ylabel("k_obs (/s)"); b2.grid(alpha=0.3)
fig.tight_layout()
fp = os.path.join(HERE, f"2026-09-14_α模型_药物卡2_pmz非单步结合归因{sfx}.png")
fig.savefig(fp, dpi=130, bbox_inches="tight")
print("  figure saved:", fp, flush=True)
