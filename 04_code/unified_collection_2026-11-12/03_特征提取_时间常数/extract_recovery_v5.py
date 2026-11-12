# -*- coding: utf-8 -*-
"""
Recovery 全库提取 v5 —— 2026-11-08 冻结口径
协议(实测确定, 跨Kv4.2/Kv11.1/Kv3.1/Kv1.1验证):
  16 sweeps, 每sweep: hold -> 0.1s 预脉冲去极化(~1.4s, 激活+失活)
  -> 恢复 @-80 (变长) -> 测试脉冲(末0.25s) 
  Δt_i = dur_i - 1.75 s (恢复间隔), 实测 data1 Δt≈0.2s, 步进≈0.15s
提取:
  预脉冲峰: 0.105-0.4s 窗最大(基线扣 0.02-0.08s)
  测试峰: 末0.26-0.02s 窗最大, 减该sweep恢复段水平(1.55-1.65s)得净值
  frac = 测试净/预脉冲净; 拟合 frac(Δt) = c0 + A(1-exp(-Δt/τ))
门槛(冻结): 预脉冲净>=0.05 nA; 恢复动态幅度 A>=0.15; r2>=0.7; τ∈[1,5000]ms
hERG型(预脉冲净 < 0.3×测试净)打旗标 rectifier_prepulse
产出: features/features_recovery_v5.csv
"""
import re, sys, time
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
NWB = ROOT / "nwb_full"
OUT = ROOT / "features"
XI = 1e-4

def _se(pcov, i):
    try:
        return float(np.sqrt(pcov[i][i])) if np.isfinite(pcov[i][i]) else np.nan
    except Exception:
        return np.nan

def parse_cell(cid, channel, temp_nominal, flag):
    fp = NWB / f"c{cid}.nwb"
    rows = []
    try:
        with h5py.File(fp, "r") as f:
            ts = f.get("acquisition/timeseries")
            if ts is None or "Recovery" not in ts: return []
            reps = ts["Recovery/repetitions"]
            rep = "repetition1" if "repetition1" in reps else list(reps.keys())[0]
            dg = ts[f"Recovery/repetitions/{rep}/data"]
            keys = [k for k in dg.keys() if re.match(r"data\d+$", k)]
            if len(keys) < 8: return []
            keys = sorted(keys, key=lambda x: int(x[4:]))
            dts, fracs = [], []
            rectifier = False
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
                if pp < 0.3*tp: rectifier = True
                dts.append(dur - 1.75)
                fracs.append(tp/pp)
            if len(dts) < 8: return []
            dts = np.array(dts)*1000.0  # ms
            fracs = np.array(fracs)
            if fracs.max() - fracs.min() < 0.15: return []  # 无失活->无恢复动力学
            def f(t, c0, A, tau): return c0 + A*(1-np.exp(-np.clip(t,0,None)/tau))
            p0 = [fracs.min(), max(fracs.max()-fracs.min(), 0.15), 100.0]
            p, pcov = curve_fit(f, dts, fracs, p0=p0,
                                bounds=([-0.5, 0.1, 1], [1.5, 2.0, 5000]), maxfev=20000)
            pred = f(dts, *p)
            r2 = 1 - np.sum((fracs-pred)**2)/max(np.sum((fracs-fracs.mean())**2), 1e-30)
            if r2 < 0.7 or not (1 <= p[2] <= 5000): return []
            rows.append(dict(cell_id=cid, temp_nominal=temp_nominal, parameter="tau_rec",
                             voltage_mV=np.nan, value=float(p[2]), unit="ms", r2=float(r2),
                             se=_se(pcov, 2), channel=channel, flag=flag,
                             rectifier_prepulse=bool(rectifier), n_sweeps=len(dts)))
    except Exception as e:
        print(f"FAIL c{cid}: {e}", file=sys.stderr, flush=True)
    return rows

def job(a): return parse_cell(*a)

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    cells = pd.read_csv(OUT / "channelpedia_all_cells.csv")
    have = set(int(re.match(r"c(\d+)\.nwb", f.name).group(1)) for f in NWB.glob("c*.nwb"))
    kv = cells[~cells.channel.str.startswith(("HCN", "Kir")) &
               ~cells.channel.str.startswith(("TA", "TR", "TW", "TH")) &
               cells.cell_id.isin(have)].drop_duplicates("cell_id")
    todo = [(int(r.cell_id), r.channel, int(re.match(r"(\d+)", str(r.combo).split("°")[0][-2:]).group(1))
             if re.search(r"(\d+)°", str(r.combo)) else None, r.flag)
            for r in kv.itertuples()]
    # 温度从combo末位解析不稳, 改从acell
    import json as _json
    def temp_of(cid):
        p = ROOT / "acell_full" / f"c{cid}.json"
        if not p.exists(): return None
        m = re.search(r'"Temp":"(\d+)c"', p.open(encoding="utf-8", errors="ignore").read(600))
        return int(m.group(1)) if m else None
    todo = [(cid, ch, temp_of(cid), fl) for cid, ch, _, fl in todo]
    todo = [t for t in todo if t[2]]
    if limit: todo = todo[:limit]
    print(f"Recovery 目标细胞: {len(todo)}", flush=True)
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, a) for a in todo]
        for i, fut in enumerate(as_completed(futs)):
            all_rows.extend(fut.result())
            if (i+1) % 1000 == 0:
                print(f"  {i+1}/{len(todo)} cells, {len(all_rows)} rows, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(all_rows)
    out = OUT / ("features_recovery_v5.csv" if not limit else "features_recovery_v5_smoke.csv")
    df.to_csv(out, index=False)
    print("saved", out, len(df), "rows", flush=True)

if __name__ == "__main__":
    main()
