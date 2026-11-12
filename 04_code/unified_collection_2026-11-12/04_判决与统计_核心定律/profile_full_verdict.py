# -*- coding: utf-8 -*-
"""轮廓似然全库重验：对判决实际使用的每一行（64 纯 Kv 组, active,
tau_act V∈[0,40] r2>=0.7, Vhalf_act r2>=0.8）逐行重拟合并做剖面似然 CI,
输出等价 SE，供判决表用轮廓 SE 全权重算。
波形读取与拟合严格复用冻结提取器 extract_features_v2 的逻辑。
用法: python profile_full_verdict.py [预算秒数, 默认250]
断点续跑: 已完成行写入 features/profile_verdict_rows.csv, 重跑自动跳过。
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd, h5py
from scipy.optimize import curve_fit
from multiprocessing import Pool

R = Path(__file__).parent
sys.path.insert(0, str(R))
from extract_features_v2 import fit_tau_rise_v2, fit_boltz3, _rise, boltz3_inc, ACT_V, E_REV

CKPT = R / "features" / "profile_verdict_rows.csv"
LN10 = np.log(10)

# ---------------- 剖面似然 ----------------
def prof_tau(tt, yy, p_fix, tau_hat, t0, n_grid=60):
    n = len(yy)
    def rss_of(tau):
        f = lambda t_, Iss, c: _rise(t_, Iss, tau, p_fix, c, t0)
        try:
            pb, _ = curve_fit(f, tt, yy, p0=[yy.max(), yy[0]], maxfev=8000)
            return float(np.sum((yy - f(tt, *pb)) ** 2))
        except Exception:
            return np.nan
    rss0 = rss_of(tau_hat)
    if not np.isfinite(rss0): return np.nan, np.nan
    thr = rss0 * np.exp(2 * 1.92 / n)
    lo = hi = np.nan
    lo_g = np.linspace(max(tau_hat * 0.4, 0.05), tau_hat, n_grid // 2)
    for tv in lo_g[::-1]:
        r = rss_of(tv)
        if np.isfinite(r) and r > thr: lo = tv; break
    hi_g = np.linspace(tau_hat, tau_hat * 2.5, n_grid // 2)
    for tv in hi_g:
        r = rss_of(tv)
        if np.isfinite(r) and r > thr: hi = tv; break
    return lo, hi

def prof_vhalf(V, y, vh_hat, n_grid=60):
    n = len(y)
    def rss_of(vh):
        f = lambda V_, A, k, c: boltz3_inc(V_, A, vh, k, c)
        try:
            pb, _ = curve_fit(f, V, y, p0=[max(y.max()-y.min(), 1e-12), 8.0, y.min()],
                              bounds=([0, 0.5, -np.inf], [np.inf, 100, np.inf]), maxfev=12000)
            return float(np.sum((y - f(V, *pb)) ** 2))
        except Exception:
            return np.nan
    rss0 = rss_of(vh_hat)
    if not np.isfinite(rss0): return np.nan, np.nan
    thr = rss0 * np.exp(2 * 1.92 / n)
    span = max(8.0, 3.0)
    lo = hi = np.nan
    lo_g = np.linspace(vh_hat - span, vh_hat, n_grid // 2)
    for tv in lo_g[::-1]:
        r = rss_of(tv)
        if np.isfinite(r) and r > thr: lo = tv; break
    hi_g = np.linspace(vh_hat, vh_hat + span, n_grid // 2)
    for tv in hi_g:
        r = rss_of(tv)
        if np.isfinite(r) and r > thr: hi = tv; break
    return lo, hi

# ---------------- 单细胞 worker ----------------
def do_cell(task):
    cid, chan, temp, voltages, want_vhalf = task
    out = []
    fp = R / "nwb_full" / f"c{int(cid)}.nwb"
    try:
        with h5py.File(fp, "r") as f:
            d = f["acquisition/timeseries/Activation/repetitions/repetition2/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T
            n = d.shape[1]; t = np.linspace(0, 700, n)
            # tau_act 逐电压
            for V in voltages:
                si = ACT_V.index(V)
                if si >= d.shape[0]: continue
                i0 = d[si].astype(float) - d[si][t < 100].mean()
                ft = fit_tau_rise_v2(t, i0, 100, 600)
                if ft is None: continue
                m = (t >= 100) & (t <= 600); tt = t[m]; ii = i0[m]
                pk = int(np.argmax(ii)); tt = tt[:pk + 1]; yy = ii[:pk + 1]
                lo, hi = prof_tau(tt, yy, ft["p"], ft["tau"], 100)
                out.append(dict(cell_id=int(cid), channel=chan, temp=temp,
                                parameter="tau_act", voltage_mV=V,
                                value=ft["tau"], se_pcov=ft["tau_se"], p=ft["p"], r2=ft["r2"],
                                prof_lo=lo, prof_hi=hi))
            # Vhalf_act: 重建 G-V
            if want_vhalf:
                G, Gv = [], []
                for si, V in enumerate(ACT_V):
                    if si >= d.shape[0]: break
                    i0 = d[si].astype(float) - d[si][t < 100].mean()
                    seg_end = i0[(t > 550) & (t <= 600)].mean()
                    seg_pk = i0[(t >= 100) & (t <= 600)].max()
                    amp = seg_pk if abs(seg_pk) > abs(seg_end) * 1.2 else seg_end
                    if abs(V - E_REV) >= 1:
                        G.append(amp / (V - E_REV)); Gv.append(V)
                fh = fit_boltz3(Gv, G, inc=True)
                if fh:
                    lo, hi = prof_vhalf(np.asarray(Gv, float), np.asarray(G, float), fh["Vh"])
                    out.append(dict(cell_id=int(cid), channel=chan, temp=temp,
                                    parameter="Vhalf_act", voltage_mV=np.nan,
                                    value=fh["Vh"], se_pcov=fh["Vh_se"], p=np.nan, r2=fh["r2"],
                                    prof_lo=lo, prof_hi=hi))
    except Exception as e:
        out.append(dict(cell_id=int(cid), channel=chan, temp=temp,
                        parameter="ERROR", voltage_mV=np.nan, value=np.nan,
                        se_pcov=np.nan, p=np.nan, r2=np.nan,
                        prof_lo=np.nan, prof_hi=np.nan, err=str(e)[:120]))
    return out

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 250.0
    t_start = time.time()
    c = pd.read_csv(R / "features" / "features_corpus_v5r.csv", low_memory=False)
    pp = pd.read_csv(R / "features" / "per_process_v5r.csv")
    g = pp.dropna(subset=["mid_obs", "tau_obs"])[["channel", "temp"]]
    key = set(zip(g.channel, g.temp))
    m = pd.Series([(ch, tp) in key for ch, tp in zip(c.channel, c.temp_nominal)], index=c.index)
    sub = c[m & (c.flag == "active")]
    t = sub[(sub.parameter == "tau_act") & (sub.r2 >= 0.7) & (sub.value > 0)
            & (sub.voltage_mV >= 0) & (sub.voltage_mV <= 40)]
    h = sub[(sub.parameter == "Vhalf_act") & (sub.r2 >= 0.8)]

    done = set()
    if CKPT.exists():
        dk = pd.read_csv(CKPT, usecols=["cell_id", "parameter", "voltage_mV"])
        for _, r in dk.iterrows():
            v = r.voltage_mV
            done.add(f"{r.cell_id}|{r.parameter}|{'nan' if pd.isna(v) else v}")

    # 组装每细胞任务
    tasks = {}
    for (cid, chan, temp), gv in t.groupby(["cell_id", "channel", "temp_nominal"]):
        vs = [V for V in sorted(gv.voltage_mV.unique())
              if f"{cid}|tau_act|{V}" not in done]
        tasks[(cid, chan, temp)] = [vs, False]
    for (cid, chan, temp), gv in h.groupby(["cell_id", "channel", "temp_nominal"]):
        if f"{cid}|Vhalf_act|nan" in done: continue
        ent = tasks.setdefault((cid, chan, temp), [[], False])
        ent[1] = True
    todo = [(cid, chan, temp, vs, wh) for (cid, chan, temp), (vs, wh) in tasks.items()
            if vs or wh]
    print(f"待处理细胞 {len(todo)} (已完成行 {len(done)})", flush=True)
    if not todo:
        print("ALL DONE"); return

    n_rows = 0
    with Pool(processes=8) as pool:
        for res in pool.imap_unordered(do_cell, todo, chunksize=4):
            if res:
                pd.DataFrame([r for r in res if r.get("parameter") != "ERROR"]).to_csv(
                    CKPT, mode="a", header=not CKPT.exists(), index=False)
                n_rows += sum(1 for r in res if r.get("parameter") != "ERROR")
            if time.time() - t_start > budget:
                pool.terminate(); break
        else:
            pool.close()
    el = time.time() - t_start
    print(f"本轮 {el:.0f}s, 新增 {n_rows} 行, 累计 checkpoint 存在={CKPT.exists()}", flush=True)

if __name__ == "__main__":
    main()
