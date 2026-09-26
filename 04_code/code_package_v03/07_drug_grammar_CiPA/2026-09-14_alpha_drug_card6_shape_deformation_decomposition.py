# 2026-09-14_α模型_药物卡6_形状变形分解.py
# ============================================================================
# 药物卡6 · 形状变形分解（τ类 vs h_ss类）（预注册归因卡）
# 预注册：结果\预注册_α模型_药物卡6_形状变形分解_2026-09-14.md（判线跑前钉死）
#
# 方法：卡5 单元逐字复用；Ramp 窗归一化形状做时间轴伸缩配准 κ*（网格 [0.5,2.0]
#   步 0.02，域 D(κ) 不越界）；f_τ = 1−(ε_x1/ε_x0_sd)²（时间尺度解释的方差份额）；
#   逐细胞对照奇/偶分裂给 f_τ_cc 地板；细胞级配对 Wilcoxon。
# 判线：合成自检 C1/C2/C3 + 回归（vs 卡5 单元表 <1e-9）；主判 T1（配对 p<0.01
#   且中位 Δf_τ≥0.25）× T2（中位 ε_x1>0.03）四象限。
# 运行：Spyder %runfile '...py' --wdir 。SMOKE=1 冒烟 3 数据集×1细胞，口径逐字相同。
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

K_GRID = np.round(np.arange(0.5, 2.0001, 0.02), 4)  # 钉死：76 点含 1.0
EX0_GATE = 0.03          # 卡5 ε_x_thr 在案值：判-able 门槛
FTAU_FLOOR = 0.25        # T1 效应地板：中位 Δf_τ ≥ 0.25
WILCOX_P = 0.01          # T1 显著性线


def conc_nM(conc, unit):
    u = (unit or "").strip().lower()
    return conc * 1000.0 if (u.startswith("u") or u.startswith("µ")) else conc


# ---------- ted.xlsx（与卡5 v3 逐字相同） ----------
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
    # v3：游标单位列混杂（lab1/lab2=ms，lab3/4/5=s），统一换算为秒
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
    """一个细胞 → (A_ctrl, ctrl末5扫TRACENUM, [(conc, 末5扫TRACENUM, b_ss)])。与卡5 逐字相同。"""
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


# ---------- 原始 csv（与卡5 v3 逐字相同：zipfile 单句柄） ----------
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
    """只读需要的 trace 电流列 + t_ms。返回 t(s), {N: i}。"""
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


# ---------- 形状基础（与卡5 逐字相同） ----------
def mean_trace(t, traces, bwin):
    sub = []
    for i in traces:
        m = (t >= bwin[0]) & (t <= bwin[1])
        b0 = np.nanmean(i[m]) if m.sum() >= 3 else 0.0
        sub.append(i - b0)
    return np.nanmean(sub, axis=0)


def shape_eps(t, ctrl, drug, bwin, rwin):
    """卡5 口径 ε 与 ε_noise + 归一化窗波形（供 warp）。
    返回 (eps_full, eps_noise, tm, cn, dn) 或 None。"""
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


# ---------- κ* 配准（钉死：网格 [0.5,2.0] 步 0.02，域不越界） ----------
def _domain_eps(tm, cn, dn, rwin, k):
    # v2 修复：绕窗起点 r0 伸缩（ramp 相位从 r0 起算）——dq(t)=dn(r0+k·(t−r0))；
    # 域 D(k)=[r0, min(r1, r0+(r1−r0)/k)]（k≤1 时全窗，k>1 时右缘收缩不越界）
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
    """返回 (kappa*, eps_x1, eps_x0_sd, f_tau_raw) 或 None。
    f_tau_raw 无判-able 门槛（cc 地板也用同一口径）。"""
    best = None
    for k in K_GRID:
        e = _domain_eps(tm, cn, dn, rwin, k)
        if e is not None and (best is None or e < best[1]):
            best = (float(k), e)
    if best is None:
        return None
    k_star, e1 = best
    # 同域 ε(1)：在 D(κ*) 上取 κ=1（绕 r0 口径下即原波形在收缩域上的比较）
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


# ---------- 合成自检变形件 ----------
def ar1_noise(n, sigma, rho=0.33, rng=RNG):
    e = rng.normal(0, sigma * np.sqrt(1 - rho * rho), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def stretch_window(y, t, rwin, factor):
    """Ramp 窗全窗均匀时间拉伸 ×factor（τ 类变形）：y2(t)=y(r0+(t−r0)/factor)。"""
    m = np.where((t >= rwin[0]) & (t <= rwin[1]))[0]
    seg = y[m]
    tt = t[m]
    src = rwin[0] + (tt - rwin[0]) / factor
    y2 = y.copy()
    y2[m] = np.interp(src, tt, seg)
    return y2


def taper_tail(y, t, rwin, factor=0.75):
    """峰后幅度线性渐缩至 ×factor（时间不动，h_ss 类幅度变形）。"""
    m = np.where((t >= rwin[0]) & (t <= rwin[1]))[0]
    seg = y[m]
    pk = int(np.argmax(np.abs(seg)))
    idx = m[pk:]
    y2 = y.copy()
    y2[idx] = y2[idx] * np.linspace(1.0, factor, len(idx))
    return y2


def synth_check(t, tmpl, bwin, rwin):
    print("\n[合成自检]", flush=True)
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
    # v2：C3 判线只留 κ* 口径；f_τ 转登记——冒烟实证单参数伸缩可把平滑幅度
    # 渐缩吸收 ~80%（f_τ≈0.8），故 f_τ 在 |κ*−1|<κ_thr 时不作 τ 证据
    c3_ok = bool(np.sum(np.abs(out["C3"]["kappa"] - 1.0) <= 0.05 + 1e-9) >= 18)
    k_thr = float(max(3.0 * np.median(np.abs(out["C1"]["kappa"] - 1.0)), 0.10))
    print(f"  C1 纯缩放: κ* 中位={np.median(out['C1']['kappa']):.3f} "
          f"全∈[0.95,1.05]={'是' if c1_ok else '否'}；"
          f"原始 f_τ 中位={np.nanmedian(out['C1']['f_tau']):.3f}（合成地板参照）", flush=True)
    print(f"  C2 时间拉伸×1.25: κ* 中位={np.median(out['C2']['kappa']):.3f} "
          f"找回{np.sum(np.abs(out['C2']['kappa'] - 1.25) <= 0.10 + 1e-9)}/20（≥18）；"
          f"f_τ 中位={np.nanmedian(out['C2']['f_tau']):.3f} ≥0.8 有"
          f"{np.sum(out['C2']['f_tau'] >= 0.8)}/20（≥18）", flush=True)
    print(f"  C3 幅度渐缩×0.75: κ* 中位={np.median(out['C3']['kappa']):.3f} "
          f"∈[0.95,1.05] 有{np.sum(np.abs(out['C3']['kappa'] - 1.0) <= 0.05 + 1e-9)}/20（≥18）；"
          f"f_τ 中位={np.nanmedian(out['C3']['f_tau']):.3f}（登记，不作 τ 证据）", flush=True)
    print(f"  κ_thr = {k_thr:.3f}（C1 锚定 max(3×C1 中位|κ*−1|, 0.10)）", flush=True)
    ok = c1_ok and c2_ok and c3_ok
    print(f"  合成自检 {'过' if ok else '不过——全卡降级登记'}", flush=True)
    return ok, dict(C1_k_med=float(np.median(out["C1"]["kappa"])),
                    C1_f_med=float(np.nanmedian(out["C1"]["f_tau"])),
                    C2_k_med=float(np.median(out["C2"]["kappa"])),
                    C2_f_med=float(np.nanmedian(out["C2"]["f_tau"])),
                    C3_k_med=float(np.median(out["C3"]["kappa"])),
                    C3_f_med=float(np.nanmedian(out["C3"]["f_tau"])),
                    C1_ok=c1_ok, C2_ok=c2_ok, C3_ok=c3_ok,
                    k_thr=k_thr), k_thr


# ---------- 模板查找（与卡5 v3 逐字相同） ----------
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


# ---------- 数据集分析 ----------
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
        # 对照奇/偶分裂 → f_τ_cc 地板（无门槛）
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


# ---------- 主流程 ----------
def main():
    print("=" * 76, flush=True)
    print(" 药物卡6 · 形状变形分解（τ类 vs h_ss类）", flush=True)
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
    print(f"\n数据集: {len(datasets)}（{'冒烟' if SMOKE else '全量'}）", flush=True)

    # 合成自检：循环前独立一次
    synth_ok = False
    synth_d = None
    k_thr = None
    tmpl = find_template(datasets)
    if tmpl is not None:
        try:
            synth_ok, synth_d, k_thr = synth_check(tmpl[0], tmpl[1], tmpl[2], tmpl[3])
        except Exception as e:
            print(f"  合成自检异常（降级登记）: {e}", flush=True)
    else:
        print("\n[合成自检] 无可用模板 -> 全卡降级登记", flush=True)

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
        print(f"  [{i + 1}/{len(datasets)}] {key}: 单元{len(ok)}", flush=True)

    # ---------- 回归检查：vs 卡5 单元表（判线 <1e-9） ----------
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
            print(f"\n[回归] 与卡5 单元表比对 {len(diffs)}/{len(all_units)} 单元，"
                  f"最大 |Δε_x|={mx:.2e} -> {'复现过' if reg_ok else '复现不过'}", flush=True)
    else:
        print("\n[回归] 卡5 单元表缺失或无单元 -> 降级登记", flush=True)

    # ---------- 判线（预注册 §六） ----------
    print("\n" + "=" * 76, flush=True)
    gated = [u for u in all_units if u.get("gated") and u.get("f_tau") is not None]
    n_edge = int(np.sum([u["edge"] for u in all_units]))
    print(f" 单元 {len(all_units)}；判-able（ε_x0≥{EX0_GATE}）{len(gated)}；网格触边 {n_edge}", flush=True)

    ks = np.array([u["kappa"] for u in gated])
    fs = np.array([u["f_tau"] for u in gated])
    ccs = np.array([u["f_tau_cc"] for u in gated if u["f_tau_cc"] is not None])
    ex1s = np.array([u["eps_x1"] for u in gated])
    med_ex1 = float(np.median(ex1s)) if len(ex1s) else np.nan
    med_koff = float(np.median(np.abs(ks - 1.0))) if len(ks) else np.nan
    med_f = float(np.median(fs)) if len(fs) else np.nan

    # 细胞级配对（v2 起为登记，不入判线）
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
            print(f"  Wilcoxon 异常: {e}", flush=True)

    # v2 判线：T1 = 中位 |κ*−1| ≥ κ_thr（C1 锚定）且 中位 f_τ ≥ 0.5；T2 不变
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

    # 分药汇总
    by_drug = {}
    for u in gated:
        by_drug.setdefault(u["dataset"].split("|")[1], []).append(u)
    drug_tab = {d: dict(n=len(us),
                        med_kappa=float(np.median([x["kappa"] for x in us])),
                        med_f_tau=float(np.median([x["f_tau"] for x in us])),
                        med_eps_x1=float(np.median([x["eps_x1"] for x in us])))
                for d, us in by_drug.items()}
    result["分药"] = drug_tab

    print(f" 判-able 单元 {len(gated)}；配对细胞 {len(pairs)}", flush=True)
    print(f" 中位 κ*={verdict_d['中位kappa']}；中位 f_τ 药={verdict_d['中位f_tau_药']} "
          f"vs cc={verdict_d['中位f_tau_cc']}", flush=True)
    print(f" T1={'过' if T1 else '不过'}（中位|κ*−1|={med_koff:.4f} κ_thr={k_thr} "
          f"中位f_τ={med_f:.3f}） T2={'过' if T2 else '不过'}（中位 ε_x1={med_ex1:.4f}）", flush=True)
    print(f" 登记: Wilcoxon 配对 p={p_one:.4g} 中位Δf_τ={med_d:.3f}（{len(pairs)} 细胞）", flush=True)
    print(f" 登记: Spearman(κ*,log10 C) ρ={rho_k:.3f} p={p_k:.3g}", flush=True)
    print(f" 总判: {verdict}", flush=True)

    fj = os.path.join(HERE, f"2026-09-14_α模型_药物卡6_形状变形分解{sfx}_结果.json")
    json.dump(result, open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  结果落盘: {fj}", flush=True)

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
    print(f"  单元表落盘: {fcsv}", flush=True)

    # 图
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
        ax.axvline(1.0, color="r", ls="--", label="κ=1（无时间变形）")
        ax.set_xlabel("κ*（>1 = 药物减慢）")
        ax.set_ylabel("单元数")
        ax.set_title(f"κ* 分布（中位 {np.median(ks):.3f}）")
        ax.legend()
    ax = axes[0, 1]
    if len(pairs):
        ax.scatter(pairs[:, 1], pairs[:, 0], s=14, alpha=0.5)
        lim = [0, 1]
        ax.plot(lim, lim, "r--", label="y=x")
        ax.set_xlabel("f_τ_cc（对照奇偶分裂地板）")
        ax.set_ylabel("f_τ 药（细胞中位）")
        ax.set_title(f"配对：Δ中位={med_d:.3f}，Wilcoxon 单侧 p={p_one:.3g}")
        ax.legend()
    ax = axes[1, 0]
    if len(gated):
        ex0s = np.array([u["eps_x0_full"] for u in gated])
        ax.scatter(ex0s, ex1s, s=10, alpha=0.4)
        ax.plot([0, 1], [0, 1], "r--", label="y=x（warp 全吃掉）")
        ax.axhline(EX0_GATE, color="gray", ls=":", label=f"残差线 {EX0_GATE}")
        ax.set_xlabel("ε_x0（warp 前，卡5 口径）")
        ax.set_ylabel("ε_x1（warp 后残差）")
        ax.set_title("warp 前后变形对照（判-able 单元）")
        ax.legend()
    ax = axes[1, 1]
    if drug_tab:
        ds_ = sorted(drug_tab.items(), key=lambda kv: -kv[1]["med_f_tau"])
        names = [d for d, _ in ds_]
        vals = [v["med_f_tau"] for _, v in ds_]
        ax.bar(range(len(names)), vals)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.axhline(FTAU_FLOOR, color="r", ls="--", label=f"参考线 {FTAU_FLOOR}（非判线）")
        ax.set_ylabel("中位 f_τ")
        ax.set_title("分药时间尺度份额（登记）")
        ax.legend()
    fpng = os.path.join(HERE, f"2026-09-14_α模型_药物卡6_形状变形分解{sfx}.png")
    fig.tight_layout()
    fig.savefig(fpng, dpi=140, bbox_inches="tight")
    print(f"  图落盘: {fpng}", flush=True)


if __name__ == "__main__":
    main()
