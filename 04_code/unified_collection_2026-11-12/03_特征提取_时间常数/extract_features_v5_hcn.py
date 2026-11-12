# -*- coding: utf-8 -*-
"""
HCN 专用提取器 v5 —— 2026-11-08 冻结口径
协议锚定(全部实测确定, 见 判词卡):
  Activation: 16 sweeps, 阶梯 0..-150 mV step -10
    锚点: 瞬时电流过零 index=4.1 ≈ v_hold=-40 mV (两细胞两窗口复测一致)
    时序: hold 0-0.1s @-40, step 0.1-3.1s, 回-40 3.1-3.6s; xi=0.1ms
  Deactivation: 14 sweeps, 阶梯 +10..-120 mV step -10
    锚点: 稳态转变 index 与同细胞 V½_act 一致 (4细胞隐含V0=+3..+12)
    时序: 预脉冲 ~0.1-1.597s (全激活), 测试步 1.597-2.699s
  E_REV = -45 mV (Channelpedia HCN model 页官方值)
  HCN 无电压依赖失活 -> Inactivation 协议跳过(记录在日志)
拟合:
  激活 τ: m^p 上升(p∈{1,2,3} AIC), 符号翻转(内向电流翻正), 窗 0.12-3.05s,
          tau∈[0.05, 20000]ms; tau>900ms 打 tau_sat 旗标(窗长3s, 未达稳态)
  G(V): I_ss(2.9-3.05s) 减漏(0..-30mV线性), G=(I-漏)/(V-E_REV), 仅V<=-55,
        Boltzmann 递减三参数, r2>=0.8 -> Vhalf_act, k_act
  去激活 τ: 符号自适应单/双指数 AIC, 窗 1.615-2.65s
产出: features/features_hcn_v5.csv
"""
import re, sys, time, json
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
NWB = ROOT / "nwb_full"
ACJ = ROOT / "acell_full"
OUT = ROOT / "features"

ACT_V = list(range(0, -160, -10))     # 16
DEA_V = list(range(10, -130, -10))    # 14
E_REV = -45.0
LN10 = np.log(10)

def _se(pcov, i):
    try:
        return float(np.sqrt(pcov[i][i])) if np.isfinite(pcov[i][i]) else np.nan
    except Exception:
        return np.nan

def boltz3_dec(V, A, Vh, k, c): return A/(1.0+np.exp((V-Vh)/k)) + c

def fit_boltz_dec(V, y, min_pts=6, r2_min=0.8):
    try:
        V = np.asarray(V, float); y = np.asarray(y, float)
        m = np.isfinite(V) & np.isfinite(y)
        if m.sum() < min_pts: return None
        V, y = V[m], y[m]
        A0 = max(y.max()-y.min(), 1e-12)
        p0 = [A0, np.median(V), 8.0, float(y.min())]
        bounds = ([0, -250, 0.5, -np.inf], [np.inf, 250, 100, np.inf])
        p, pcov = curve_fit(boltz3_dec, V, y, p0=p0, bounds=bounds, maxfev=20000)
        pred = boltz3_dec(V, *p)
        ss = np.sum((y-pred)**2); st = np.sum((y-y.mean())**2)
        r2 = 1-ss/st if st > 0 else 0.0
        if r2 < r2_min: return None
        return dict(Vh=float(p[1]), Vh_se=_se(pcov,1), k=float(p[2]), k_se=_se(pcov,2), r2=float(r2))
    except Exception:
        return None

def fit_iv_boltz(V, Iss, noise, min_pts=6, r2_min=0.8):
    """归一化 G(V) 两参数 Boltzmann:
    漏: -10..-40 mV 线性(避开 sweep0 漂移); G=(Iss-漏)/(V-E_REV), V<=-60;
    归一化到 G(-150) (所有HCN亚型在-150激活>=97%);
    拟合 Gn = 1/(1+exp((V-Vh)/k)), 返回 Vh, k, r2"""
    try:
        V = np.asarray(V, float); Iss = np.asarray(Iss, float)
        lm = (V >= -40) & (V <= -10)
        if lm.sum() < 3: return None
        lk = np.polyfit(V[lm], Iss[lm], 1)
        Ic = Iss - np.polyval(lk, V)
        # 振幅门: -150 处漏扣幅度
        if abs(Ic[-1]) < max(0.02, 5*float(noise[-1])): return None
        gm = V <= -60
        if gm.sum() < min_pts: return None
        Vg = V[gm]; G = Ic[gm]/(Vg-E_REV)
        if G[-1] <= 0: return None
        Gn = G/G[-1]
        p, pcov = curve_fit(lambda V_, Vh, k: 1/(1+np.exp((V_-Vh)/k)), Vg, Gn,
                            p0=[-90, 8], bounds=([-150, 2], [-50, 25]), maxfev=20000)
        pred = 1/(1+np.exp((Vg-p[0])/p[1]))
        ss = np.sum((Gn-pred)**2); st = np.sum((Gn-Gn.mean())**2)
        r2 = 1-ss/st if st > 0 else 0.0
        if r2 < r2_min: return None
        return dict(Vh=float(p[0]), Vh_se=_se(pcov,0), k=float(p[1]), k_se=_se(pcov,1), r2=float(r2))
    except Exception:
        return None

def _rise(t, Iss, tau, p, c, t0):
    return Iss*np.clip(1-np.exp(-(t-t0)/tau), 0, None)**p + c

def fit_tau_rise_signed(t, i, t0, t1):
    """符号自适应 m^p 上升拟合 (内向电流自动翻正)"""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 50: return None
    tt = t[m]; ii = i[m].astype(float)
    # 符号: 用末端-首端判断方向
    if (ii[-int(len(ii)*0.1):].mean() - ii[:int(len(ii)*0.05)].mean()) < 0:
        ii = -ii
    # HCN 无失活: 不用峰值截断, 全程拟合
    npts = len(ii)
    best = None
    for p in (1, 2, 3):
        try:
            f = lambda t_, Iss, tau, c: _rise(t_, Iss, tau, p, c, t0)
            p0 = [max(ii.max()-ii.min(), 1e-9), 300.0, ii[0]]
            pb, pcov = curve_fit(f, tt, ii, p0=p0,
                                 bounds=([0, 0.05, -np.inf], [np.inf, 20000, np.inf]),
                                 maxfev=20000)
            pred = f(tt, *pb)
            rss = float(np.sum((ii-pred)**2))
            aic = npts*np.log(rss/npts + 1e-30) + 2*3
            st = float(np.sum((ii-ii.mean())**2))
            r2 = 1-rss/st if st > 0 else 0.0
            if best is None or aic < best["aic"]:
                best = dict(tau=float(pb[1]), tau_se=_se(pcov,1), p=p, r2=float(r2), aic=aic)
        except Exception:
            continue
    if best is None or best["r2"] < 0.7 or best["tau"] <= 0: return None
    best["tau_sat"] = bool(best["tau"] > 900)
    best["inact_contam"] = False
    return best

def fit_tau_decay_signed(t, i, t0, t1):
    """符号自适应单/双指数衰减 (AIC)"""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 30: return None
    tt = t[m]; ii = i[m].astype(float)
    if ii[:5].mean() < ii[-5:].mean():  # 从负值衰减到0 -> 翻正
        ii = -ii
    npts = len(tt)
    def mono(t_, A, tau, c): return A*np.exp(-(t_-t0)/tau) + c
    def bi(t_, A1, tau1, A2, tau2, c): return A1*np.exp(-(t_-t0)/tau1)+A2*np.exp(-(t_-t0)/tau2)+c
    out = None
    try:
        p1, pc1 = curve_fit(mono, tt, ii, p0=[max(ii.max()-ii.min(),1e-9), 100, ii[-1]],
                            bounds=([0,0.05,-np.inf],[np.inf,20000,np.inf]), maxfev=15000)
        rss1 = float(np.sum((ii-mono(tt,*p1))**2))
        aic1 = npts*np.log(rss1/npts+1e-30) + 2*3
        out = dict(tau=float(p1[1]), tau_se=_se(pc1,1), model="mono", aic=aic1,
                   r2=1-rss1/max(np.sum((ii-ii.mean())**2),1e-30))
        try:
            p0b = [p1[0]*0.6, max(p1[1]/5,1), p1[0]*0.4, p1[1]*5, p1[2]]
            p2, pc2 = curve_fit(bi, tt, ii, p0=p0b,
                                bounds=([0,0.05,0,0.05,-np.inf],[np.inf,20000,np.inf,20000,np.inf]),
                                maxfev=20000)
            rss2 = float(np.sum((ii-bi(tt,*p2))**2))
            aic2 = npts*np.log(rss2/npts+1e-30) + 2*5
            if aic2 < aic1 - 6:
                taus = sorted([(float(p2[1]), _se(pc2,1)), (float(p2[3]), _se(pc2,3))])
                out = dict(tau=taus[0][0], tau_se=taus[0][1], tau2=taus[1][0],
                           tau2_se=taus[1][1], model="bi", aic=aic2,
                           r2=1-rss2/max(np.sum((ii-ii.mean())**2),1e-30))
        except Exception:
            pass
    except Exception:
        return None
    if out is None or out["r2"] < 0.7 or out["tau"] <= 0: return None
    return out

def read_meta(cid):
    p = ACJ / f"c{cid}.json"
    if not p.exists(): return None
    head = p.open("r", encoding="utf-8", errors="ignore").read(600)
    def g(k):
        m = re.search(rf'"{k}":"([^"]*)"', head)
        return m.group(1) if m else None
    ch, temp = g("ionChannel"), g("Temp")
    tm = re.match(r"(\d+)c", temp or "")
    # isGood repID
    repid = None
    m = re.search(r'"isGood_param":\{"repID":(\d+)', p.open("r", encoding="utf-8", errors="ignore").read(200000))
    if m: repid = int(m.group(1))
    return dict(channel=ch, temp_nominal=int(tm.group(1)) if tm else None, repid=repid)

def parse_cell(cid, channel, temp_nominal, flag="?", repid=None):
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
            if ts is None or "Activation" not in ts: return []
            reps = ts["Activation/repetitions"]
            rep = f"repetition{repid}" if repid and f"repetition{repid}" in reps else \
                  ("repetition2" if "repetition2" in reps else list(reps.keys())[0])
            try:
                temp_real = float(ts[f"Activation/repetitions/{rep}/head_temp"][0][0])
            except Exception:
                temp_real = float(temp_nominal) if temp_nominal else np.nan
            # ---- Activation ----
            d = ts[f"Activation/repetitions/{rep}/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T  # (sweeps, samples)
            xi = 1e-4
            npts = d.shape[1]
            t = np.arange(npts)*xi*1000.0   # ms, 与 v2 口径一致
            if npts < 30000: return []  # 非HCN协议
            base = d[:, int(0.05/xi):int(0.09/xi)].mean(axis=1)
            noise = d[:, int(0.05/xi):int(0.09/xi)].std(axis=1)
            d0 = d - base[:, None]
            Iss = d0[:, int(2.9/xi):int(3.05/xi)].mean(axis=1)
            V = np.array(ACT_V[:d.shape[0]], float)
            fh = fit_iv_boltz(V, Iss, noise)
            if fh:
                add("Vhalf_act", np.nan, fh["Vh"], "mV", fh["r2"], se=fh["Vh_se"])
                add("k_act", np.nan, fh["k"], "mV", fh["r2"], se=fh["k_se"])
            # 振幅门参考: 相对最正三扫的中位(通道关闭区)
            Iref = np.median(Iss[:3])
            for si in range(d.shape[0]):
                # 振幅门: 激活幅度 >= max(0.02 nA, 5×噪声)
                if abs(Iss[si]-Iref) < max(0.02, 5*noise[si]): continue
                ft = fit_tau_rise_signed(t, d0[si], 120.0, 3050.0)
                if ft:
                    add("tau_act", float(V[si]), ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                        extra=dict(p_model=ft["p"], inact_contam=False, tau_sat=ft["tau_sat"]))
            # ---- Deactivation ----
            if "Deactivation" in ts:
                dd = ts["Deactivation/repetitions/repetition1/data"][:]
                if dd.ndim == 1: dd = dd[:, None]
                dd = dd.T
                if dd.shape[1] >= 26000:
                    td = np.arange(dd.shape[1])*xi*1000.0   # ms
                    based = dd[:, int(0.05/xi):int(0.09/xi)].mean(axis=1)
                    dd0 = dd - based[:, None]
                    for si in range(min(dd.shape[0], len(DEA_V))):
                        ft = fit_tau_decay_signed(td, dd0[si], 1615.0, 2650.0)
                        if ft:
                            add("tau_deact", float(DEA_V[si]), ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                                extra=dict(model=ft.get("model")))
                            if ft.get("model") == "bi":
                                add("tau_deact_slow2", float(DEA_V[si]), ft["tau2"], "ms", ft["r2"], se=ft["tau2_se"])
    except Exception as e:
        print(f"FAIL c{cid}: {e}", file=sys.stderr, flush=True)
    for r in rows:
        r["channel"] = channel; r["flag"] = flag
    return rows

def job(args):
    return parse_cell(*args)

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    cells = pd.read_csv(OUT / "channelpedia_all_cells.csv")
    have = set(int(re.match(r"c(\d+)\.nwb", f.name).group(1)) for f in NWB.glob("c*.nwb"))
    hcn = cells[cells.channel.str.startswith("HCN") & cells.cell_id.isin(have)].drop_duplicates("cell_id")
    flag_map = cells.drop_duplicates("cell_id").set_index("cell_id").flag.to_dict()
    todo = []
    for cid in sorted(hcn.cell_id):
        meta = read_meta(cid)
        if meta and meta["channel"] and meta["temp_nominal"]:
            todo.append((cid, meta["channel"], meta["temp_nominal"], flag_map.get(cid, "?"), meta["repid"]))
    if limit: todo = todo[:limit]
    print(f"HCN 目标细胞: {len(todo)}", flush=True)
    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, a) for a in todo]
        for i, fut in enumerate(as_completed(futs)):
            all_rows.extend(fut.result())
            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(todo)} cells, {len(all_rows)} rows, {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(all_rows)
    out = OUT / ("features_hcn_v5.csv" if not limit else "features_hcn_v5_smoke.csv")
    df.to_csv(out, index=False)
    print("saved", out, len(df), "rows", flush=True)

if __name__ == "__main__":
    main()
