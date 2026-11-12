# -*- coding: utf-8 -*-
"""
Channelpedia 全库提取管线 v3
- 数据: nwb_full/c<id>.nwb（文件名无元数据，元数据来自 aCell JSON）
- 范围: active 且有文件的 Kv 细胞（全部物种 × 宿主 × 温度）
- 拟合逻辑完全复用 extract_features_v2（m^p 激活、SE、三参数 Boltzmann、单/双指数去激活）
- 多进程并行
产出: features/features_full_v3.csv
"""
import re, sys, time
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features_v2 import (fit_boltz3, fit_tau_rise_v2, fit_tau_decay_v2,
                                 ACT_V, DEA_V, INA_V, E_REV)

ROOT = Path(__file__).resolve().parent
NWB = ROOT / "nwb_full"
ACJ = ROOT / "acell_full"
OUT = ROOT / "features"

def read_meta(cid):
    """从 aCell JSON 头部抓元数据（只读前 600 字节）"""
    p = ACJ / f"c{cid}.json"
    if not p.exists():
        return None
    head = p.open("r", encoding="utf-8", errors="ignore").read(600)
    def g(k):
        m = re.search(rf'"{k}":"([^"]*)"', head)
        return m.group(1) if m else None
    ch, sp, host, temp = g("ionChannel"), g("species"), g("hostCell"), g("Temp")
    if not ch:
        return None
    tm = re.match(r"(\d+)c", temp or "")
    return dict(channel=ch, species=sp, host=host,
                temp_nominal=int(tm.group(1)) if tm else None)

def parse_cell_v3(cid, channel, temp_nominal):
    fp = NWB / f"c{cid}.nwb"
    rows = []
    def add(par, V, val, unit, r2, se=np.nan, extra=None):
        r = dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp_nominal,
                 parameter=par, voltage_mV=V, value=val, unit=unit, r2=r2, se=se)
        if extra: r.update(extra)
        rows.append(r)
    try:
        with h5py.File(fp, "r") as f:
            ts = f.get("acquisition/timeseries")
            if ts is None: return []
            try:
                temp_real = float(ts["Activation/repetitions/repetition2/head_temp"][0][0])
            except Exception:
                temp_real = float(temp_nominal) if temp_nominal else np.nan
            if "Activation" in ts:
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
                    ft = fit_tau_rise_v2(t, i0, 100, 600)
                    if ft and V >= -40:
                        add("tau_act", V, ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                            extra=dict(p_model=ft["p"], inact_contam=ft["inact_contam"]))
                fh = fit_boltz3(Gv, G, inc=True)
                if fh:
                    add("Vhalf_act", np.nan, fh["Vh"], "mV", fh["r2"], se=fh["Vh_se"])
                    add("k_act", np.nan, fh["k"], "mV", fh["r2"], se=fh["k_se"])
            if "Deactivation" in ts:
                d = ts["Deactivation/repetitions/repetition1/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T
                n = d.shape[1]; t = np.linspace(0, 700, n)
                for si, V in enumerate(DEA_V):
                    if si >= d.shape[0]: break
                    i0 = d[si].astype(float) - d[si][t < 100].mean()
                    ft = fit_tau_decay_v2(t, i0, 402, 600)
                    if ft:
                        add("tau_deact", V, ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                            extra=dict(model=ft.get("model")))
                        if ft.get("model") == "bi":
                            add("tau_deact_slow2", V, ft["tau2"], "ms", ft["r2"], se=ft["tau2_se"])
            if "Inactivation" in ts:
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
                    cond_t = (t >= 200) & (t <= 1500)
                    cond = i0[cond_t]
                    if len(cond) > 50 and cond.max() > 0 and (cond.max()-cond[-1])/(cond.max()+1e-12) > 0.3:
                        ft = fit_tau_decay_v2(t[cond_t], cond, 200, 1500)
                        if ft:
                            add("tau_inact", V, ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                                extra=dict(model=ft.get("model")))
                fh = fit_boltz3(INA_V, peaks, inc=False)
                if fh:
                    add("Vhalf_inact", np.nan, fh["Vh"], "mV", fh["r2"], se=fh["Vh_se"])
    except Exception as e:
        print(f"FAIL c{cid}: {e}", file=sys.stderr, flush=True)
    for r in rows: r["channel"] = channel
    return rows

def job(args):
    cid, ch, tn = args
    return parse_cell_v3(cid, ch, tn)

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    cells = pd.read_csv(OUT / "channelpedia_all_cells.csv")
    act = cells[cells.flag == "active"]
    have = set(int(re.match(r"c(\d+)\.nwb", f.name).group(1))
               for f in NWB.glob("c*.nwb"))
    todo = []
    for cid in sorted(set(act.cell_id) & have):
        meta = read_meta(cid)
        if meta and meta["channel"] and meta["channel"].startswith("Kv") and meta["temp_nominal"]:
            todo.append((cid, meta["channel"], meta["temp_nominal"]))
    if limit: todo = todo[:limit]
    print(f"目标 Kv active 细胞: {len(todo)}", flush=True)
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, a) for a in todo]
        for i, fut in enumerate(as_completed(futs)):
            all_rows.extend(fut.result())
            if (i + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)} cells, {len(all_rows)} rows, {el:.0f}s", flush=True)
    df = pd.DataFrame(all_rows)
    out = OUT / ("features_full_v3.csv" if not limit else "features_v3_smoke.csv")
    df.to_csv(out, index=False)
    print("saved", out, len(df), "rows", flush=True)

if __name__ == "__main__":
    main()
