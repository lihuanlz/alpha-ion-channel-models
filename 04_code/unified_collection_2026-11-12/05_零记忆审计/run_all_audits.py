# -*- coding: utf-8 -*-
"""
run_all_audits.py — 一键全验（第二遍零记忆审计的可复跑合集）
覆盖: A 清单 / B Kv主判决 / C HCN协议结构 / D HCN复算 / E ICG判决链
      F DANDI Nav / G K2P / H 跨物种·Q10·指纹 / I 去重对拍
用法: python run_all_audits.py     （在 law_channelpedia 目录下运行）
每项输出 PASS/FAIL + 关键数字; 末尾汇总。任何 FAIL 都会给出偏差细节。
"""
import json, re, sys, time, importlib.util
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[2]          # 工作区根
F = ROOT / "features"
RESULTS = []

def report(tag, ok, detail):
    RESULTS.append((tag, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {detail}", flush=True)

# ============ A. 清单核实 ============
def audit_A():
    n_nwb = len(list((ROOT / "nwb_full").glob("c*.nwb")))
    n_json = len(list((ROOT / "acell_full").glob("c*.json")))
    fc = pd.read_csv(F / "features_corpus_v5.csv", low_memory=False)
    ok = (n_nwb == 19344 and n_json == 19282 and len(fc) == 270827 and fc.cell_id.nunique() == 17965)
    report("A 清单", ok, f"NWB {n_nwb}/19344, acell {n_json}/19282, corpus {len(fc)}/270827行 {fc.cell_id.nunique()}/17965细胞")

# ============ B. Kv 主判决 ============
MIN_CELLS = 15
PROC_WIN = {"tau_act": (0, 40), "tau_deact": (-60, -20), "tau_inact": (-20, 40)}
def logsd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v) & (v > 0)]
    if len(v) < MIN_CELLS: return np.nan
    return float(np.std(np.log10(v), ddof=1))

def audit_B():
    df = pd.read_csv(F / "features_corpus_v5.csv", low_memory=False)
    kva = df[~df.channel.str.startswith("HCN")]
    kva = kva[kva.flag == "active"]
    rows = []
    for (ch, T), g in kva.groupby(["channel", "temp_nominal"]):
        if g.cell_id.nunique() < MIN_CELLS: continue
        vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)].value
        kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)].value
        if len(vh) < MIN_CELLS or len(kk) < MIN_CELLS: continue
        row = dict(channel=ch, temp=T, mid_obs=float(vh.std() / kk.median()))
        for proc, (lo, hi) in PROC_WIN.items():
            ta = g[(g.parameter == proc) & (g.r2 >= 0.7) & (g.value > 0) & (g.voltage_mV.between(lo, hi))]
            sds = [logsd(gv.value) for _, gv in ta.groupby("voltage_mV")]
            sds = [s for s in sds if np.isfinite(s)]
            row[proc] = float(np.median(sds)) if sds else np.nan
        rows.append(row)
    P = pd.DataFrame(rows)
    pp = pd.read_csv(F / "per_process_v5.csv")
    m = P.merge(pp, on=["channel", "temp"], suffixes=("_n", "_o"))
    worst = 0.0
    for col in ["mid_obs", "tau_act", "tau_deact", "tau_inact"]:
        d = (m[f"{col}_n"] - m[f"{col}_o"]).abs()
        rel = (d / m[f"{col}_o"].abs()).replace([np.inf], np.nan)
        if len(rel.dropna()): worst = max(worst, float(rel.max()))
    sub = P.dropna(subset=["mid_obs", "tau_act"])
    r = sub.mid_obs / sub.tau_act
    ok = (len(P) == 76 and len(m) == 76 and worst < 1e-12 and len(sub) == 66
          and abs((r > 1).mean() - 0.9848) < 1e-3 and abs(sub.tau_act.median() - 0.2039) < 1e-3)
    report("B Kv主判决", ok, f"76组复现 worst_rel={worst:.1e}; 66组 通过{(r>1).mean()*100:.2f}% τ中位{sub.tau_act.median():.4f}")

# ============ C. HCN 协议结构 ============
def audit_C():
    import h5py
    h = pd.read_csv(F / "features_hcn_v5.csv")
    cids = h.cell_id.drop_duplicates().sample(6, random_state=7).tolist()
    good = 0
    for cid in cids:
        with h5py.File(ROOT / "nwb_full" / f"c{cid}.nwb", "r") as f:
            ts = f["acquisition/timeseries"]
            ra = list(ts["Activation/repetitions"].keys())[0]
            rd = list(ts["Deactivation/repetitions"].keys())[0]
            sa = ts[f"Activation/repetitions/{ra}/data"].shape
            sd = ts[f"Deactivation/repetitions/{rd}/data"].shape
            if sa[1] == 16 and sd[1] == 14 and sa[0] >= 30000: good += 1
    report("C HCN协议", good == 6, f"{good}/6 细胞 Activation 16 sweeps / Deactivation 14 sweeps")

# ============ D. HCN 抽查复算 ============
def audit_D():
    sys.path.insert(0, str(ROOT))
    from extract_features_v5_hcn import parse_cell, read_meta
    cells = pd.read_csv(F / "channelpedia_all_cells.csv", low_memory=False)
    flag_map = cells.drop_duplicates("cell_id").set_index("cell_id").flag.to_dict()
    old_all = pd.read_csv(F / "features_hcn_v5.csv")
    worst = 0.0
    for cid in [27604, 23163, 27899]:
        meta = read_meta(cid)
        new = pd.DataFrame(parse_cell(cid, meta["channel"], meta["temp_nominal"], flag_map.get(cid, "?"), meta["repid"]))
        old = old_all[old_all.cell_id == cid].copy()
        new["voltage_mV"] = new["voltage_mV"].astype(float); old["voltage_mV"] = old["voltage_mV"].astype(float)
        m = new.merge(old, on=["parameter", "voltage_mV"], suffixes=("_n", "_o"))
        if len(m) != len(old): report("D HCN复算", False, f"c{cid} 行数不齐 {len(m)}/{len(old)}"); return
        d = (m.value_n.astype(float) - m.value_o.astype(float)).abs()
        rel = (d / m.value_o.astype(float).abs()).replace([np.inf], np.nan)
        worst = max(worst, float(np.nanmax(rel)))
    report("D HCN复算", worst < 1e-12, f"3细胞 value 最大相对偏差 {worst:.1e}")

# ============ E. ICG 判决链 ============
GRID = np.round(np.arange(-100, 81, 2), 1)
def icg_verdict(sub, gate, cls):
    if sub.shape[0] < 8: return None
    M = {}
    for _, rr in sub.iterrows():
        M[rr.model + "|" + rr.gate] = np.interp(GRID, rr.V, rr.tau, left=np.nan, right=np.nan)
    tm = pd.DataFrame(M, index=GRID)
    vh_med = sub.Vhalf.median()
    win = (GRID >= vh_med - 20) & (GRID <= vh_med + 20)
    logsd_ = tm[win].apply(lambda r: np.nanstd(np.log10(r.values.astype(float)))
                           if np.isfinite(r.values.astype(float)).sum() >= 8 else np.nan, axis=1)
    tau_obs = np.nanmedian(logsd_.values)
    mid_obs = sub.Vhalf.std() / sub.k.median()
    return dict(gate=gate, cls=cls, n_models=sub.shape[0], tau_obs=tau_obs, mid_obs=mid_obs,
                ratio=mid_obs / tau_obs)

def audit_E():
    rows = []
    for FAM in ["K", "Na", "KCa", "IH"]:
        df = pd.read_pickle(ROOT / "icg_out" / f"gates_icg_{FAM}.pkl")
        a = df[df.kind == "act"]; r = icg_verdict(a, "act", FAM)
        if r: rows.append(r)
        i = df[df.kind == "inact"]
        if len(i):
            r = icg_verdict(i, "inact", FAM)
            if r: rows.append(r)
    df = pd.read_pickle(ROOT / "icg_out" / "gates_icg_Ca.pkl")
    act = df[df.kind == "act"]; ina = df[df.kind == "inact"]
    ivh = ina.groupby("model").Vhalf.median(); avh = act.groupby("model").Vhalf.median()
    cls = lambda m: ("T-type(LVA)" if (ivh[m] < -55 if m in ivh.index else avh.get(m, 0) < -45) else "HVA(L/N/P/Q/R)")
    act = act.assign(cls=act.model.map(cls)); ina = ina.assign(cls=ina.model.map(cls))
    for c, s in act.groupby("cls"):
        r = icg_verdict(s, "act", c)
        if r: rows.append(r)
    for c, s in ina.groupby("cls"):
        r = icg_verdict(s, "inact", c)
        if r: rows.append(r)
    new = pd.DataFrame(rows)
    old = pd.read_csv(ROOT / "icg_out" / "verdict_icg_ALL.csv")
    m = new.merge(old, on=["gate", "cls"], suffixes=("_n", "_o"))
    worst = float(((m.ratio_n - m.ratio_o).abs() / m.ratio_o.abs()).max())
    ok = len(m) == 10 and worst < 1e-3 and (m.n_models_n == m.n_models_o).all()
    report("E ICG判决链", ok, f"{len(m)}/10 行, ratio 最大相对偏差 {worst:.1e}")

# ============ F. DANDI Nav 抽查 ============
def audit_F():
    ddir = WS / "09_历史归档" / "2026-10-16_清理" / "data" / "dandi_001750"
    sdir = WS / "09_历史归档" / "_根目录散件_2026-10-15" / "extract_nav_dandi001750.py"
    spec = importlib.util.spec_from_file_location("exnav", sdir)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    tbl = pd.read_csv(ddir / "per_patch_nav_table_2026-10-08.csv", dtype=str)
    ok_ids = set(tbl[tbl.ok == "1"].cell_id.unique())
    worst = 0.0; n = 0
    for fp in sorted((ddir / "full").glob("*.nwb")):
        pid = fp.name.split("obj-")[1].replace("_icephys.nwb", "")
        if pid not in ok_ids: continue
        r = mod.analyze_patch(str(fp))
        if r is None: continue
        sub = tbl[tbl.cell_id == pid]
        vh_old = float(sub[sub.parameter == "V_half_activation"].value.iloc[0])
        worst = max(worst, abs(r["Vh_act"] - vh_old)); n += 1
        if n >= 3: break
    report("F DANDI Nav", n == 3 and worst == 0.0, f"{n} 膜片 Vh_act 最大绝对偏差 {worst:.1e}")

# ============ G. K2P ============
def audit_G():
    df = pd.read_pickle(F / "k2p_recon_cells_2026-11-08.pkl")
    out = []
    for ch, g in df.groupby("channel"):
        S = np.array(g["shape"].tolist())
        ssd = np.nanstd(S, axis=0)
        out.append(dict(channel=ch, n=len(g),
                        amp_CV=round(float(np.abs(g.i80).std() / np.abs(g.i80).mean()), 3),
                        shape_SD_med=round(float(np.median(ssd[3:14])), 3)))
    new = pd.DataFrame(out)
    old = pd.read_csv(F / "k2p_recon_2026-11-08.csv")
    m = new.merge(old, on="channel", suffixes=("_n", "_o"))
    ok = len(m) == 6 and (m.n_n == m.n_o).all() and (m.amp_CV_n == m.amp_CV_o).all() and (m.shape_SD_med_n == m.shape_SD_med_o).all()
    report("G K2P侦察", ok, f"{len(m)}/6 通道 n/amp_CV/shape_SD 逐位一致")

# ============ H1/H2 跨物种 + Q10 ============
def audit_H12():
    df3 = pd.read_csv(F / "features_full_v3.csv", low_memory=False)
    cells = pd.read_csv(F / "channelpedia_all_cells.csv", low_memory=False)
    meta = cells[cells.flag == "active"].drop_duplicates("cell_id")[["cell_id", "combo"]]
    def pc(c):
        m = re.match(r"(\S+)\s+FT\s+([hmr])", str(c))
        return pd.Series({"sp": {"h": "human", "m": "mouse", "r": "rat"}[m.group(2)] if m else None})
    meta = meta.join(meta.combo.apply(pc))
    df3 = df3.merge(meta[["cell_id", "sp"]], on="cell_id", how="left")
    tau = df3[(df3.parameter == "tau_act") & (df3.voltage_mV.between(0, 40)) & (df3.r2 >= 0.7) & (df3.value > 0)]
    sub = tau[(tau.channel == "Kv1.1") & (tau.temp_nominal == 25)]
    med = {s: g.groupby("voltage_mV").value.median() for s, g in sub.groupby("sp")}
    common = med["human"].index.intersection(med["mouse"].index)
    fold = float((med["human"][common] / med["mouse"][common]).median())
    dfa = pd.read_csv(F / "features_all.csv", low_memory=False)
    g = dfa[dfa.channel == "Kv1.1"]
    t15 = g[(g.temp_nominal == 15) & (g.parameter == "tau_act") & (g.voltage_mV == 20) & (g.value > 0)].value.median()
    t35 = g[(g.temp_nominal == 35) & (g.parameter == "tau_act") & (g.voltage_mV == 20) & (g.value > 0)].value.median()
    q10 = float((t15 / t35) ** 0.5)
    ok = abs(fold - 0.867243) < 1e-5 and abs(q10 - 5.0148) < 1e-3
    report("H12 跨物种+Q10", ok, f"tau_fold {fold:.6f} (存档0.867243), Q10@20 {q10:.4f} (存档5.0148)")

# ============ H3 指纹 ============
def audit_H3():
    fc = pd.read_csv(F / "features_corpus_v5.csv", low_memory=False)
    kv = fc[~fc.channel.str.startswith("HCN")]
    VSET = [0, 10, 20, 30, 40]
    ta = kv[(kv.parameter == "tau_act") & (kv.r2 >= 0.7) & (kv.value > 0) & (kv.voltage_mV.isin(VSET))]
    piv = ta.pivot_table(index="cell_id", columns="voltage_mV", values="value", aggfunc="median")
    piv.columns = [f"tau_{int(c)}" for c in piv.columns]
    vh = kv[(kv.parameter == "Vhalf_act") & (kv.r2 >= 0.8)].set_index("cell_id").value.rename("Vhalf")
    kk = kv[(kv.parameter == "k_act") & (kv.r2 >= 0.8)].set_index("cell_id").value.rename("k")
    X = piv.join([vh, kk], how="inner").dropna()
    meta = kv.drop_duplicates("cell_id").set_index("cell_id")
    X["channel"] = meta.loc[X.index, "channel"]; X["flag"] = meta.loc[X.index, "flag"]
    Xa = X[X.flag == "active"]
    keep = Xa.channel.value_counts(); keep = keep[keep >= 150].index
    Xa = Xa[Xa.channel.isin(keep)]
    y = np.asarray(Xa.channel, dtype=str)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    feats = [c for c in Xa.columns if c not in ("channel", "flag")]
    Xt = Xa[[c for c in feats if c.startswith("tau_")]].apply(np.log10)
    Xv = Xa[["Vhalf", "k"]]
    Xall = Xa[feats].copy()
    for c in [c for c in feats if c.startswith("tau_")]: Xall[c] = np.log10(Xall[c])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = {}
    for tag, Xx in [("tau", Xt), ("vhk", Xv), ("all", Xall)]:
        rf = RandomForestClassifier(n_estimators=300, n_jobs=8, random_state=42)
        accs[tag] = float(cross_val_score(rf, Xx.values, y, cv=skf).mean())
    ok = (len(Xa) == 3261 and len(keep) == 9 and abs(accs["tau"] - 0.5032) < 5e-3
          and abs(accs["vhk"] - 0.4149) < 5e-3 and abs(accs["all"] - 0.7120) < 5e-3)
    report("H3 指纹分类器", ok, f"n={len(Xa)} 类={len(keep)} tau {accs['tau']:.4f} / V½k {accs['vhk']:.4f} / all {accs['all']:.4f}")

# ============ I. 去重对拍 ============
def audit_I():
    r4 = pd.read_csv(F / "features_rest_v4.csv", low_memory=False)
    fc = pd.read_csv(F / "features_corpus_v5.csv", low_memory=False)
    key = ["cell_id", "parameter", "voltage_mV", "channel"]
    a = r4[key + ["value"]].copy(); b = fc[key + ["value"]].copy()
    a["value"] = a["value"].astype(float); b["value"] = b["value"].astype(float)
    m = a.merge(b, on=key, suffixes=("_v4", "_v5"))
    rel = (np.abs(m.value_v4 - m.value_v5) / np.abs(m.value_v5)).replace([np.inf], np.nan)
    nonhcn = ~m.channel.str.startswith("HCN")
    same = float((rel == 0).mean())
    bad_non = int((rel[nonhcn] > 0.01).sum())
    ok = len(m) == 95672 and abs(same - 0.9838) < 1e-3 and bad_non == 0
    report("I 去重对拍", ok, f"{len(m)} 行, 逐字节相同 {same*100:.2f}%, 非HCN >1% 差异 {bad_non} 行")

if __name__ == "__main__":
    t0 = time.time()
    for fn in [audit_A, audit_B, audit_C, audit_D, audit_E, audit_F, audit_G, audit_H12, audit_H3, audit_I]:
        try:
            fn()
        except Exception as e:
            report(fn.__name__, False, f"EXCEPTION: {type(e).__name__}: {e}")
    print("=" * 70)
    nfail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"汇总: {len(RESULTS) - nfail}/{len(RESULTS)} PASS, {nfail} FAIL, 耗时 {time.time()-t0:.0f}s")
    sys.exit(1 if nfail else 0)
