# 2026-09-14_α模型_药物卡1_阻断动力学与状态偏好.py
# ============================================================================
# 药物卡1：dofetilide/pimozide 阻断动力学与状态偏好（判线跑前钉死，见
#   预注册_α模型_药物卡1_阻断动力学与状态偏好_2026-09-14.md）
#
# 数据：a6k5t ted.xlsx（ResultsWide 逐扫 Ramp 游标 + LiquidAdditions 加药轴）
#   池A dofetilide lab3（15 细胞，排除 v6 定罪 57_A/61_E），浓度 3/10/30/100 nM
#   池B pimozide lab4（16 细胞），浓度 0.2/1/2.4 nM
#
# 判线（钉死）：
#   L1  Hill 自提 IC50、n 与实验室 ic50nh 比 ∈ [0.7,1.4]
#   L1b 自有游标 vs ResultsWide Ramp 逐扫相关，子样本中位 r ≥ 0.98
#   L2  k_obs~C 线性 R² ≥ 0.7 且 k_on>0
#   L3  b(-60..-40)/b(+20..+40) 中位 ≥0.7 捕获型 / ≤0.3 缓解型 / 中间登记
#
# 运行：Spyder %runfile '...py' --wdir 。SMOKE=1 冒烟只跑 csv 子样本 1 细胞/药。
# ============================================================================
import os, io, json, zipfile
import numpy as np
import openpyxl
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/结果/"
        "hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包")
SMOKE = os.environ.get("SMOKE", "0") == "1"

POOLS = {
    "dofetilide": dict(dir="a6k5t-osfstorage-dofetilide-archivelab3",
                       drug_liquid="Dofetilide", conc_to_nM=1000.0,
                       exclude_prefix=("57_A", "61_E"),
                       lab_ic50_nM=14.375898159, lab_nh=1.036667369173,
                       zipped=True),
    "pimozide": dict(dir="a6k5t-osfstorage-pimozide-archivelab4",
                     drug_liquid="pimozide", conc_to_nM=1.0,
                     exclude_prefix=(),
                     lab_ic50_nM=1.304384580932, lab_nh=0.969856648120,
                     zipped=False),
}

RATIO_LINE = (0.7, 1.4)
R_LINE = 0.98
R2_LINE = 0.7
TRAP_LINE, RELIEVE_LINE = 0.7, 0.3
MIN_ACTRL_PA = 50.0
MIN_BSS = 0.05


def load_ted(tedx):
    wb = openpyxl.load_workbook(tedx, read_only=True)
    def rows(sn):
        return [list(r) for r in wb[sn].iter_rows(values_only=True)]
    la = rows("LiquidAdditions")
    rw = rows("ResultsWide")
    cd = rows("CursorsDefinitions")
    wb.close()
    return la, rw, cd


def parse_anl(s):
    if s is None:
        return []
    return [int(x) for x in str(s).strip().strip(";").split(";") if x.strip().isdigit()]


def cell_records(la, rw, drug_liquid, conc_to_nM, exclude_prefix):
    hdr = la[0]
    ic = {k: hdr.index(k) for k in ("CELLID", "LIQUID", "CONC", "CONCU", "FIRST", "LAST", "ANL", "CTLFL")}
    rhdr = rw[0]
    rc = {k: rhdr.index(k) for k in ("CELLID", "ELTIME", "TRACENUM", "LIQUID", "Ramp")}
    # ResultsWide 按细胞分组
    ramp = {}
    for r in rw[1:]:
        cid = str(r[rc["CELLID"]])
        try:
            tn = int(r[rc["TRACENUM"]])
            ramp_v = float(r[rc["Ramp"]])
            et = float(r[rc["ELTIME"]])
        except (TypeError, ValueError):
            continue
        ramp.setdefault(cid, {})[tn] = (et, ramp_v)
    cells = {}
    for r in la[1:]:
        cid = str(r[ic["CELLID"]])
        if any(cid.startswith(p) for p in exclude_prefix):
            continue
        liquid = str(r[ic["LIQUID"]])
        ent = cells.setdefault(cid, {})
        if r[ic["CTLFL"]] == "Y" or liquid.lower() in ("vehicle-control", "control"):
            ent["ctrl_anl"] = parse_anl(r[ic["ANL"]])
        elif liquid.lower() == drug_liquid.lower():
            ent["conc_nM"] = float(r[ic["CONC"]]) * conc_to_nM
            ent["drug_first"] = int(r[ic["FIRST"]])
            ent["drug_last"] = int(r[ic["LAST"]])
            ent["drug_anl"] = parse_anl(r[ic["ANL"]])
    out = {}
    for cid, ent in cells.items():
        if cid not in ramp or "ctrl_anl" not in ent or "drug_anl" not in ent:
            continue
        ent["ramp"] = ramp[cid]
        out[cid] = ent
    return out


def analyze_pool(name, cfg):
    base = os.path.join(PACK, cfg["dir"], "subtracted", "ted")
    la, rw, cd = load_ted(os.path.join(base, "ted.xlsx"))
    cells = cell_records(la, rw, cfg["drug_liquid"], cfg["conc_to_nM"], cfg["exclude_prefix"])
    print(f"\n=== 池 {name}（{cfg['dir']}）：候选细胞 {len(cells)}", flush=True)
    recs = []
    for cid, ent in sorted(cells.items()):
        ramp = ent["ramp"]
        ctrl_vals = [ramp[t][1] for t in ent["ctrl_anl"] if t in ramp and np.isfinite(ramp[t][1])]
        drug_vals = [ramp[t][1] for t in ent["drug_anl"] if t in ramp and np.isfinite(ramp[t][1])]
        if not ctrl_vals or not drug_vals:
            recs.append(dict(cell=cid, 判定="excluded_ANL缺档"))
            continue
        A_ctrl = float(np.median(ctrl_vals))
        if abs(A_ctrl) < MIN_ACTRL_PA:
            recs.append(dict(cell=cid, 判定="excluded_信号不足", A_ctrl=A_ctrl))
            continue
        A_drug = float(np.median(drug_vals))
        b_ss = 1.0 - A_drug / A_ctrl
        # wash-in
        t0 = None
        ts, bs = [], []
        for tn in range(ent["drug_first"], ent["drug_last"] + 1):
            if tn not in ramp:
                continue
            et, rv = ramp[tn]
            if t0 is None:
                t0 = et
            ts.append((et - t0) / 1000.0)
            bs.append(1.0 - rv / A_ctrl)
        k_obs = None
        if b_ss > MIN_BSS and len(ts) >= 10:
            tt = np.array(ts)
            bb = np.array(bs)
            y = -np.log(np.clip(1.0 - bb / b_ss, 0.02, 1.0))
            ok = (bb > 0) & (bb < b_ss * 1.5)
            if ok.sum() >= 5:
                k_obs = float(np.sum(tt[ok] * y[ok]) / np.sum(tt[ok] ** 2))
        recs.append(dict(cell=cid, 判定="ok", conc_nM=ent["conc_nM"],
                         A_ctrl=A_ctrl, A_drug=A_drug, b_ss=b_ss,
                         k_obs=k_obs, n_washin=len(ts)))
        print(f"  {cid}: C={ent['conc_nM']:7.3f}nM A_ctrl={A_ctrl:8.1f}pA "
              f"b_ss={b_ss:.3f} k_obs={k_obs if k_obs else float('nan'):.5f}/s", flush=True)
    return recs, base, cd


def hill_fit(Cs, bs):
    from scipy.optimize import curve_fit
    def hill(C, ic50, n):
        return C ** n / (ic50 ** n + C ** n)
    C = np.array(Cs, float)
    b = np.array(bs, float)
    p0 = [float(np.exp(np.mean(np.log(C)))), 1.0]
    p, _ = curve_fit(hill, C, b, p0=p0, bounds=([1e-2, 0.2], [1e4, 4.0]), maxfev=20000)
    return float(p[0]), float(p[1])


def lin_fit(x, y):
    x = np.array(x, float)
    y = np.array(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ coef
    ss_res = float(np.sum((y - yh) ** 2))
    ss_tot = float(np.sum((y - yh.mean()) ** 2)) if len(y) else 0.0
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(coef[0]), float(coef[1]), r2


# ------------------------------------------------------------ csv 子样本（L1b+L3）
def csv_cols(hdr):
    m = {}
    for j, c in enumerate(hdr):
        c = c.strip()
        if c.startswith("trace_#"):
            body = c[7:]
            num, _, kind = body.partition("_current")
            if kind:
                m.setdefault(int(num), {})["i"] = j
            else:
                num, _, kind = body.partition("_voltage")
                if kind:
                    m.setdefault(int(num), {})["v"] = j
    return m


def load_csv_subset(path, zipped, need_traces):
    """只读需要的 trace 列 + t_ms。返回 t(s), {N: (i_pA, v_mV)}"""
    if zipped:
        z = zipfile.ZipFile(path)
        fh = z.open(z.namelist()[0])
    else:
        fh = open(path, "rb")
    hdr = fh.readline().decode("utf-8", "replace").strip().split(",")
    cmap = csv_cols(hdr)
    use = ["t_ms"]
    keep = {}
    for N in need_traces:
        if N in cmap:
            if "i" in cmap[N]:
                use.append(hdr[cmap[N]["i"]])
            if "v" in cmap[N]:
                use.append(hdr[cmap[N]["v"]])
            keep[N] = (hdr[cmap[N]["i"]], (hdr[cmap[N]["v"]] if cmap[N].get("v") is not None else None))
    df = pd.read_csv(fh, names=hdr, header=None, usecols=lambda c: c in use)
    fh.close()
    t = df["t_ms"].to_numpy(float) / 1000.0
    out = {}
    for N, (ci, cv) in keep.items():
        i = df[ci].to_numpy(float)
        v = df[cv].to_numpy(float) if cv and cv in df else None
        out[N] = (i, v)
    return t, out


def subset_cells(recs, per_conc):
    ok = [r for r in recs if r["判定"] == "ok"]
    by = {}
    for r in ok:
        by.setdefault(r["conc_nM"], []).append(r)
    picks = []
    for C, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["A_ctrl"])
        med = np.median([r["A_ctrl"] for r in rs])
        rs.sort(key=lambda r: abs(r["A_ctrl"] - med))
        picks += rs[:per_conc]
    return picks


def csv_audit(name, cfg, base, cd, picks):
    """L1b 逐扫相关 + L3 block(V)。"""
    hdr_cd = [str(c).strip().upper() if c else "" for c in cd[0]]
    ix = {n: hdr_cd.index(n) for n in ("CURSOR", "STIME", "ETIME")}
    curs = {}
    for r in cd[1:]:
        if r and r[ix["CURSOR"]]:
            try:
                curs[str(r[ix["CURSOR"]]).strip()] = (float(r[ix["STIME"]]), float(r[ix["ETIME"]]))
            except (TypeError, ValueError):
                pass
    bw = curs["Baseline"]
    rw_ = curs["Ramp"]
    out = []
    for r in picks:
        cid = r["cell"]
        fn = os.path.join(base, cid + (".csv.zip" if cfg["zipped"] else ".csv"))
        if not os.path.exists(fn):
            out.append(dict(cell=cid, 判定="csv缺"));
            continue
        ent = r["_ent"]
        # L1b: ANL 对照5 + ANL 药5 + wash-in 每10扫
        wash = list(range(ent["drug_first"], ent["drug_last"] + 1, 10))
        need = sorted(set(ent["ctrl_anl"]) | set(ent["drug_anl"]) | set(wash))
        try:
            t, tr = load_csv_subset(fn, cfg["zipped"], need)
        except Exception as e:
            out.append(dict(cell=cid, 判定="csv读失败", err=str(e)[:120]))
            continue
        def amp(iarr):
            mb = np.nanmean(iarr[(t >= bw[0]) & (t <= bw[1])])
            mr = np.nanmean(iarr[(t >= rw_[0]) & (t <= rw_[1])])
            return mr - mb
        ours, labs = [], []
        for N in need:
            if N in tr and N in ent["ramp"]:
                ours.append(amp(tr[N][0]))
                labs.append(ent["ramp"][N][1])
        r_corr = float(np.corrcoef(ours, labs)[0, 1]) if len(ours) >= 8 else np.nan
        # L3: 复合对照/药扫 → b(V)
        ratio = np.nan
        bV = None
        ci_tr = [tr[N] for N in ent["ctrl_anl"] if N in tr and tr[N][1] is not None]
        di_tr = [tr[N] for N in ent["drug_anl"] if N in tr and tr[N][1] is not None]
        if ci_tr and di_tr:
            base_sub = lambda iarr: iarr - np.nanmean(iarr[(t >= bw[0]) & (t <= bw[1])])
            ic = np.nanmean([base_sub(x[0]) for x in ci_tr], axis=0)
            idc = np.nanmean([base_sub(x[0]) for x in di_tr], axis=0)
            vv = ci_tr[0][1]
            bins = np.arange(-80, 45.1, 5.0)
            bc = 0.5 * (bins[:-1] + bins[1:])
            icm = np.full(len(bc), np.nan)
            idm = np.full(len(bc), np.nan)
            for k in range(len(bc)):
                m = (vv >= bins[k]) & (vv < bins[k + 1])
                if m.sum() >= 2:
                    icm[k] = np.nanmean(ic[m])
                    idm[k] = np.nanmean(idc[m])
            good = np.isfinite(icm) & np.isfinite(idm) & (np.abs(icm) >= 20.0)
            bVv = 1.0 - idm[good] / icm[good]
            bV = dict(V=[float(x) for x in bc[good]], b=[float(x) for x in bVv])
            m_neg = (bc[good] >= -60) & (bc[good] <= -40)
            m_pos = (bc[good] >= 20) & (bc[good] <= 40)
            if m_neg.any() and m_pos.any():
                num_ = float(np.median(bVv[m_neg]))
                den_ = float(np.median(bVv[m_pos]))
                # 守门（跑前钉死）：去极化档阻断 <0.10 时比值被噪声撑爆，判不可判
                if den_ >= 0.10:
                    ratio = num_ / den_
        out.append(dict(cell=cid, conc_nM=r["conc_nM"], 判定="ok",
                        L1b_r=r_corr, L3_ratio=ratio, bV=bV))
        print(f"  [csv] {cid}: L1b r={r_corr:.4f}  L3 比={ratio if np.isfinite(ratio) else float('nan'):.3f}",
              flush=True)
    return out


# ------------------------------------------------------------ 主跑
result = {"预注册": "预注册_α模型_药物卡1_阻断动力学与状态偏好_2026-09-14.md",
          "判线": {"L1比": RATIO_LINE, "L1b_r": R1_LINE if False else R_LINE,
                   "L2_R2": R2_LINE, "L3": [TRAP_LINE, RELIEVE_LINE]},
          "pools": {}}
fig_data = {}
for name, cfg in POOLS.items():
    recs, base, cd = analyze_pool(name, cfg)
    ok = [r for r in recs if r["判定"] == "ok"]
    # L1
    C = [r["conc_nM"] for r in ok]
    b = [r["b_ss"] for r in ok]
    ic50, nh = hill_fit(C, b)
    L1_ic = ic50 / cfg["lab_ic50_nM"]
    L1_nh = nh / cfg["lab_nh"]
    L1_pass = (RATIO_LINE[0] <= L1_ic <= RATIO_LINE[1]) and (RATIO_LINE[0] <= L1_nh <= RATIO_LINE[1])
    # L2
    kk = [(r["conc_nM"], r["k_obs"]) for r in ok if r["k_obs"]]
    kon = koff = r2 = np.nan
    if len(kk) >= 6:
        kon, koff, r2 = lin_fit([x[0] for x in kk], [x[1] for x in kk])
    L2_pass = bool(np.isfinite(r2) and r2 >= R2_LINE and kon > 0)
    # L1b/L3 子样本
    picks = subset_cells(recs, 1 if SMOKE else 2)
    for r in recs:  # 挂回 _ent 供 csv 用
        pass
    # 重新挂 ent
    cells_ent = cell_records(*load_ted(os.path.join(base, "ted.xlsx"))[:2],
                             cfg["drug_liquid"], cfg["conc_to_nM"], cfg["exclude_prefix"])
    for p in picks:
        p["_ent"] = cells_ent.get(p["cell"], {})
    csvs = csv_audit(name, cfg, base, cd, picks)
    L1b_rs = [c["L1b_r"] for c in csvs if c.get("L1b_r") is not None and np.isfinite(c["L1b_r"])]
    L1b_med = float(np.median(L1b_rs)) if L1b_rs else np.nan
    L1b_pass = bool(np.isfinite(L1b_med) and L1b_med >= R_LINE)
    # L3 每浓度档中位（比值不可判的浓度档照实登记，不参与判型）
    l3 = {}
    l3_nan = {}
    for c in csvs:
        if c.get("判定") == "ok":
            k = str(c["conc_nM"])
            if np.isfinite(c.get("L3_ratio") or np.nan):
                l3.setdefault(k, []).append(c["L3_ratio"])
            else:
                l3_nan[k] = l3_nan.get(k, 0) + 1
    l3_med = {k: float(np.median(v)) for k, v in l3.items()}
    l3_verdict = {}
    for k, v in l3_med.items():
        l3_verdict[k] = "捕获型" if v >= TRAP_LINE else ("缓解型" if v <= RELIEVE_LINE else "中间登记")
    for k in l3_nan:
        l3_verdict.setdefault(k, "低阻断不可判")
    judged = [v for v in l3_verdict.values() if v != "低阻断不可判"]
    L3_pass = bool(judged) and all(v == "捕获型" for v in judged)
    result["pools"][name] = {
        "n_ok": len(ok), "excluded": [r for r in recs if r["判定"] != "ok"],
        "recs": [{k: v for k, v in r.items() if k != "_ent"} for r in recs],
        "L1": {"ic50_self": ic50, "nh_self": nh, "ic50_lab": cfg["lab_ic50_nM"],
               "nh_lab": cfg["lab_nh"], "ratio_ic50": L1_ic, "ratio_nh": L1_nh,
               "过": L1_pass},
        "L2": {"kon_per_nM_s": kon, "koff_per_s": koff, "R2": r2, "n_cells": len(kk), "过": L2_pass},
        "L1b": {"r_med": L1b_med, "n": len(L1b_rs), "过": L1b_pass},
        "L3": {"per_conc_median_ratio": l3_med, "判": l3_verdict, "过": L3_pass},
        "csv": [{k: v for k, v in c.items() if k != "bV"} for c in csvs],
    }
    fig_data[name] = dict(C=C, b=b, ic50=ic50, nh=nh, kk=kk, kon=kon, koff=koff,
                          bV={c["cell"]: c["bV"] for c in csvs if c.get("bV")})
    print(f"\n[{name}] L1: IC50 {ic50:.2f}nM (比 {L1_ic:.2f}) nH {nh:.2f} (比 {L1_nh:.2f}) -> {'过' if L1_pass else '不过'}"
          f"\n      L2: kon={kon:.2e}/nM/s koff={koff:.4f}/s R2={r2:.3f} -> {'过' if L2_pass else '不过'}"
          f"\n      L1b: r_med={L1b_med:.4f} -> {'过' if L1b_pass else '不过'}"
          f"\n      L3: {l3_verdict}", flush=True)

# ------------------------------------------------------------ 总判词
P = result["pools"]
lines = []
for name in POOLS:
    p = P[name]
    lines.append(f"{name}: L1 {'过' if p['L1']['过'] else '不过'} | L1b {'过' if p['L1b']['过'] else '不过'} | "
                 f"L2 {'过' if p['L2']['过'] else '不过'} | L3 {p['L3']['判']}")
result["判词"] = lines
print("\n" + "=" * 76, flush=True)
for s in lines:
    print(" ", s, flush=True)

sfx = "_冒烟" if SMOKE else ""
fjson = os.path.join(HERE, f"2026-09-14_α模型_药物卡1_阻断动力学与状态偏好{sfx}_结果.json")
json.dump(result, open(fjson, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("  结果落盘:", fjson, flush=True)

# ------------------------------------------------------------ 图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
cols = {"dofetilide": "#c0392b", "pimozide": "#2980b9"}

ax = axes[0, 0]
for name, fd in fig_data.items():
    C, b = np.array(fd["C"]), np.array(fd["b"])
    ax.scatter(C, b, c=cols[name], label=name, s=30, alpha=0.7)
    if C.size:
        xs = np.logspace(np.log10(C.min() / 3), np.log10(C.max() * 3), 100)
        ys = xs ** fd["nh"] / (fd["ic50"] ** fd["nh"] + xs ** fd["nh"])
        ax.plot(xs, ys, c=cols[name], lw=1.5)
        ax.axvline(fd["ic50"], c=cols[name], ls=":", lw=1)
ax.set_xscale("log")
ax.set_xlabel("浓度 (nM)")
ax.set_ylabel("稳态阻断分数 b")
ax.set_title("L1 浓度-抑制（点=逐细胞，线=自提 Hill）")
ax.legend()

ax = axes[0, 1]
for name, fd in fig_data.items():
    kk = fd["kk"]
    if not kk:
        continue
    C = np.array([x[0] for x in kk])
    k = np.array([x[1] for x in kk])
    ax.scatter(C, k, c=cols[name], label=name, s=30, alpha=0.7)
    if np.isfinite(fd["kon"]):
        xs = np.linspace(0, C.max() * 1.1, 20)
        ax.plot(xs, fd["kon"] * xs + fd["koff"], c=cols[name], lw=1.5)
ax.set_xscale("log")
ax.set_xlabel("浓度 (nM)")
ax.set_ylabel("k_obs (/s)")
ax.set_title("L2 wash-in 速率 vs 浓度")
ax.legend()

ax = axes[1, 0]
for name, fd in fig_data.items():
    for cell, bV in fd["bV"].items():
        if bV:
            ax.plot(bV["V"], bV["b"], lw=1.2, alpha=0.7, c=cols[name])
ax.axhline(TRAP_LINE, ls="--", c="gray", lw=0.8)
ax.set_xlabel("V (mV)")
ax.set_ylabel("b(V)")
ax.set_title("L3 ramp 阻断电压剖面（子样本逐细胞）")
ax.axvspan(-60, -40, color="orange", alpha=0.15)
ax.axvspan(20, 40, color="green", alpha=0.15)

ax = axes[1, 1]
txt = []
for name in POOLS:
    p = P[name]
    txt.append(f"{name}:  L1 {'✓' if p['L1']['过'] else '✗'}  L1b {'✓' if p['L1b']['过'] else '✗'}  "
               f"L2 {'✓' if p['L2']['过'] else '✗'}  L3 {list(p['L3']['判'].values())}")
ax.text(0.05, 0.7, "\n\n".join(txt), fontsize=12, family="Microsoft YaHei", va="top")
ax.axis("off")
ax.set_title("判词速览")

fig.tight_layout()
fpng = os.path.join(HERE, f"2026-09-14_α模型_药物卡1_阻断动力学与状态偏好{sfx}.png")
fig.savefig(fpng, dpi=130, bbox_inches="tight")
print("  图落盘:", fpng, flush=True)
