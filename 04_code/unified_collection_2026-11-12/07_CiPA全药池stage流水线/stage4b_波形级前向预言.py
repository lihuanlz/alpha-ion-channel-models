# -*- coding: utf-8 -*-
# stage4b_波形级前向预言.py
# ============================================================================
# Tier-1 第2条 · Stage4b：波形级前向预言（lab_3/4/5，实测 V(t)，逐细胞）
#
# 与 Stage4a 的差别：4a 只验占据幅度；4b 验动力学形状——冻结的 Channelpedia
# 群体 τ_m 表（35°C，Q10 桥到各文件实测温度）+ 冻结 m_ss/h_ss/τ_h 锚点，
# 每个细胞只用一个自由电导 G（对照段最小二乘），对实测电压轨迹做前向仿真。
#
# 非平凡预言（幅度因 G 线性缩放而平凡，形状才是判决）：
#   T1  对照段去激活 τ：仿真迹线 vs 实测迹线（同一冻结窗程序拟合）——逐细胞 τ 比
#   T2  药物段 τ：模型设定 τ 不变，实测应复现（Stage2 τ守恒 的绝对尺度版）
#   T3  药物段幅度：sim/meas 比值应与 Stage3 tail_slope 相关（模型预言真平台）
# 判线（冻结，同 68 细胞老卡）：比值 ∈ [0.5, 2.0]
#
# 用法：python stage4b_波形级前向预言.py --budget 250 --workers 16
# ============================================================================
import os, json, glob, time, argparse, zipfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
CKPT = os.path.join(HERE, "stage4b_ckpt.jsonl")
OUTCSV = os.path.join(HERE, "stage4b_per_cell_waveform.csv")

# ---------------------------------------------------------- α 模型（冻结，逐字同 2026-11-09 换表卡）
E_REV = -92.1
T_MAIN = 21.5
Q10 = 2.5
YMED = {-70: 0.029477868155508298, -60: 0.03945937095037588, -50: 0.08393733831866093}
H40_POP = 0.014753124449143482
H50_POP = 0.011385195620532567
hB1 = {-80: 0.7980112179405232, -70: 0.42458878766834396, -60: 0.3098949595189373,
       -50: 0.2548959330145549, -40: 0.20781431816537388, -30: 0.16969269623291475,
       -20: 0.13749178922473596, -10: 0.11350819372481134, 0: 0.0904564201253725,
       10: 0.07195623486448417, 20: 0.053275325401300214, 30: 0.039430262401624006}
TH_ORIG = {-130: 0.0030376559246143267, -120: 0.0030376559246143267,
           -110: 0.004254082928519305, -100: 0.005045399817563913,
           -90: 0.0065, -70: 0.0070, -60: 0.0070, -50: 0.0089,
           -40: 0.0144, -30: 0.0144, -20: 0.05, 0: 0.0015, 60: 0.0015}
TM_ORIG = {-130: 0.03033214284254048, -120: 0.03033214284254048,
           -110: 0.049199997974127406, -100: 0.08577126043784405,
           -90: 0.072, -80: 0.24, -70: 0.8127, -60: 1.20549, -50: 10.14073,
           -40: 24.0539, -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
           20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
_FAC = 2.5 ** 1.35
CP_DEACT = {-70: 37.7, -60: 43.4, -50: 56.7, -40: 80.0, -30: 85.1}
CP_ACT = {-20: 280.7, -10: 323.0, 0: 268.0, 10: 150.6, 20: 90.2,
          30: 64.1, 40: 57.5, 50: 71.5, 60: 105.6}
TM_CP = dict(TM_ORIG)
TM_CP.update({v: ms / 1000 * _FAC for v, ms in CP_DEACT.items()})
TM_CP.update({v: ms / 1000 * _FAC for v, ms in CP_ACT.items()})

class Tab:
    def __init__(self, anchors):
        self.xs = np.array(sorted(anchors), float)
        self.ly = np.log(np.array([anchors[int(x)] for x in self.xs], float))

def build_tabs(T_lab):
    m_anchors = {-130: 1e-4, -120: 1e-4, -110: 1e-4, -100: 1e-4, -90: 1e-4,
                 -80: min(1.0, YMED[-70] * 0.5 / hB1[-80]),
                 -70: min(1.0, YMED[-70] / hB1[-70]),
                 -60: min(1.0, YMED[-60] / hB1[-60]),
                 -50: min(1.0, YMED[-50] / hB1[-50]),
                 -40: min(1.0, YMED[-50] * 3.0 / hB1[-40]),
                 -30: min(1.0, YMED[-50] * 8.0 / hB1[-30]),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    h_anchors = {-130: 1.0, -100: 1.0}
    h_anchors.update(hB1)
    h_anchors[40] = H40_POP
    h_anchors[50] = H50_POP
    h_anchors[60] = H50_POP
    fac = Q10 ** ((T_lab - T_MAIN) / 10.0)
    tm = {k: v / fac for k, v in TM_CP.items()}
    th = {k: v / fac for k, v in TH_ORIG.items()}
    return dict(m_ss=Tab(m_anchors), tau_m=Tab(tm), h_ss=Tab(h_anchors), tau_h=Tab(th))

def forward_dt(V, tabs, dt):
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hs = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-dt / tm); eh = np.exp(-dt / th)
    n = len(V); m = np.empty(n); h = np.empty(n)
    m0 = ms[0]; h0 = hs[0]
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hs[i] + (h0 - hs[i]) * eh[i]
        m[i] = m0; h[i] = h0
    return m, h

# ---------------------------------------------------------- 冻结的 τ 窗程序（逐字同 stage2d）
from scipy.optimize import curve_fit
def exp_off(t, a, tau, c):
    return a * np.exp(-(t - t[0]) / tau) + c

def tau_tail_of(t, y, sweep_ms):
    i0 = int(0.05 * len(t))
    ipk = i0 + int(np.argmax(y[i0:]))
    A_peak = float(y[ipk]); t_peak = float(t[ipk])
    t_end = float(t[-1])
    t1 = t_peak + 0.03 * sweep_ms
    t2 = min(t_peak + 0.50 * sweep_ms, t_end - 0.01 * sweep_ms)
    m_tail = (t >= t1) & (t <= t2)
    tau = np.nan
    if m_tail.sum() >= 25:
        tt = t[m_tail]; yy = y[m_tail]
        decay_amp = yy[0] - yy[-1]
        if decay_amp > max(3.0, 0.05 * abs(A_peak)):
            try:
                p0 = [decay_amp, 0.1 * sweep_ms, yy[-1]]
                hi_b = 2.0 * sweep_ms
                popt, _ = curve_fit(exp_off, tt, yy, p0=p0, maxfev=3000,
                                    bounds=([-1e7, 0.001 * sweep_ms, -1e7], [1e7, hi_b, 1e7]))
                pred = exp_off(tt, *popt)
                ss_res = float(np.sum((yy - pred) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                lo_b = 0.001 * sweep_ms
                hit = abs(popt[1] - hi_b) < 0.02 * hi_b or abs(popt[1] - lo_b) < 0.05 * lo_b
                if popt[1] > 0 and r2 >= 0.5 and not hit:
                    tau = float(popt[1])
            except Exception:
                pass
    return A_peak, tau

# ---------------------------------------------------------- LOO 占据参数（与 stage4a 同口径重算）
def build_loo():
    s4 = pd.read_csv(os.path.join(HERE, "stage4a_per_cell_predictions.csv"))
    ok = s4[(s4["join"] == "ok") & np.isfinite(s4.block_meas) & np.isfinite(s4.conc_uM)]
    from scipy.optimize import least_squares
    def hill(C, ic50, h):
        C = np.asarray(C, float); return C ** h / (C ** h + ic50 ** h)
    loo = {}
    for (d, lab), g in ok.groupby(["drug_real", "lab"]):
        tr = ok[(ok.drug_real == d) & (ok.lab != lab)]
        C, y = tr.conc_uM.values, tr.block_meas.values
        m = np.isfinite(C) & np.isfinite(y) & (C > 0)
        if m.sum() < 6 or len(np.unique(C[m])) < 2:
            continue
        best = None
        for a in (-3, -1, 1):
            for b in (0.7, 1.0, 1.5):
                try:
                    r = least_squares(lambda p: hill(C[m], 10 ** p[0], p[1]) - y[m],
                                      [a, b], bounds=([-6, 0.2], [3, 3.5]))
                    if best is None or r.cost < best.cost:
                        best = r
                except Exception:
                    pass
        if best is not None:
            loo[(d, lab)] = (10 ** best.x[0], float(best.x[1]))
    return loo

# ---------------------------------------------------------- 逐细胞处理
def proc_cell(rec):
    zpath, cell, lab, drug_real, conc_uM, occ = rec
    try:
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        t = df["t_ms"].to_numpy(float)
        sweep_ms = float(t.max() - t.min())
        dt_s = float(np.median(np.diff(t))) / 1000.0
        cur = {int(c.split("#")[1].split("_")[0]): c for c in df.columns
               if c.startswith("trace_#") and "_current_" in c}
        # 电压列必须精确后缀 _voltage_mV（排除 _voltage_V 伪装温度列与 .1 重复列）
        vol = {int(c.split("#")[1].split("_")[0]): c for c in df.columns
               if c.startswith("trace_#") and c.endswith("_voltage_mV")}
        tem = [c for c in df.columns if c.startswith("trace_#") and c.endswith("_temperature_C")]
        if len(cur) < 8:
            return dict(ok=False, cell=cell, err="traces<8")
        if not vol:
            return dict(ok=False, cell=cell, err="无电压列")
        T_file = float(np.nanmedian(df[tem].to_numpy(float))) if tem else 36.5
        if not (15.0 < T_file < 45.0):
            T_file = 36.5
        uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(
            list(cur.values())[0].rsplit("_current_", 1)[-1].lower())
        if uf is None:
            return dict(ok=False, cell=cell, err="单位未知")
        tabs = build_tabs(T_file)

        trs = sorted(set(cur) & set(vol))
        ys, vs = {}, {}
        stds = {}
        for tr in trs:
            y = df[cur[tr]].to_numpy(float) * uf
            if np.ptp(y) > 4000 or abs(np.max(y)) > 3000:
                continue
            ys[tr] = y; vs[tr] = df[vol[tr]].to_numpy(float)
            stds[tr] = float(np.std(y))
        if len(ys) < 8:
            return dict(ok=False, cell=cell, err="post-artifact<8")
        sarr = np.array(list(stds.values()))
        thr = max(5.0, 0.05 * np.percentile(sarr, 95))
        test_trs = sorted([k for k in trs if k in ys and stds[k] > thr])
        if len(test_trs) < 12:
            return dict(ok=False, cell=cell, err="test<12")
        # 分段（同 stage2d 冻结逻辑）
        Aps = {tr: tau_tail_of(t, ys[tr], sweep_ms)[0] for tr in test_trs}
        Amax = float(np.percentile(list(Aps.values()), 90))
        if Amax <= 20:
            return dict(ok=False, cell=cell, err="Amax低")
        n = len(test_trs)
        head = float(np.median([Aps[tr] for tr in test_trs[:max(3, n // 3)]]))
        if head <= 0:
            head = Amax
        i_e4 = n
        for i in range(n - 1, -1, -1):
            if Aps[test_trs[i]] < 0.2 * head:
                i_e4 = i
            else:
                break
        e4 = (n - i_e4) >= 3
        ss_end = i_e4 if e4 else n
        ss_trs = test_trs[max(0, ss_end - 5):ss_end]
        i_ve = -1
        for i in range(min(n // 3 + 1, n)):
            if Aps[test_trs[i]] / Amax > 0.55:
                i_ve = i
            else:
                break
        veh_trs = test_trs[:i_ve + 1] if i_ve >= 2 else test_trs[:3]
        veh_use = veh_trs[-3:]
        # G：对照段最小二乘
        num = den = 0.0
        shapes = {}
        for tr in veh_use:
            m, h = forward_dt(vs[tr], tabs, dt_s)
            gsh = m * h * (vs[tr] - E_REV)
            shapes[tr] = gsh
            okk = np.isfinite(ys[tr])
            num += float(np.sum(gsh[okk] * ys[tr][okk]))
            den += float(np.sum(gsh[okk] ** 2))
        if den <= 0:
            return dict(ok=False, cell=cell, err="G退化")
        G = num / den
        # T1：对照 τ 预言
        tau_mea_c, tau_sim_c = [], []
        rms_c = []
        for tr in veh_use:
            i_sim = G * shapes[tr]
            _, taum = tau_tail_of(t, ys[tr], sweep_ms)
            _, taus = tau_tail_of(t, i_sim, sweep_ms)
            if np.isfinite(taum):
                tau_mea_c.append(taum)
            if np.isfinite(taus):
                tau_sim_c.append(taus)
            okk = np.isfinite(ys[tr])
            rms_c.append(float(np.sqrt(np.mean((i_sim[okk] - ys[tr][okk]) ** 2)) /
                               (np.sqrt(np.mean(ys[tr][okk] ** 2)) + 1e-12)))
        rms_ctrl = float(np.median(rms_c)) if rms_c else np.nan
        # T2/T3：药物稳态段
        tau_mea_d, tau_sim_d, a_rat = [], [], []
        for tr in ss_trs:
            m, h = forward_dt(vs[tr], tabs, dt_s)
            i_sim = (1.0 - occ) * G * (m * h * (vs[tr] - E_REV))
            Am, taum = tau_tail_of(t, ys[tr], sweep_ms)
            As, taus = tau_tail_of(t, i_sim, sweep_ms)
            if np.isfinite(taum):
                tau_mea_d.append(taum)
            if np.isfinite(taus):
                tau_sim_d.append(taus)
            if abs(Am) > 1.0 and np.isfinite(As):
                a_rat.append(float(As / Am))
        return dict(ok=True, cell=cell, lab=lab, drug=drug_real, conc_uM=conc_uM, occ=occ,
                    T_file=T_file, G=G, rms_ctrl=rms_ctrl,
                    tau_meas_ctrl=float(np.median(tau_mea_c)) if tau_mea_c else np.nan,
                    tau_sim_ctrl=float(np.median(tau_sim_c)) if tau_sim_c else np.nan,
                    tau_meas_drug=float(np.median(tau_mea_d)) if tau_mea_d else np.nan,
                    tau_sim_drug=float(np.median(tau_sim_d)) if tau_sim_d else np.nan,
                    A_ratio_drug=float(np.median(a_rat)) if a_rat else np.nan,
                    n_veh=len(veh_use), n_ss=len(ss_trs), e4=e4)
    except Exception as e:
        return dict(ok=False, cell=cell, err=f"{type(e).__name__}: {e}")

def find_zip(cell):
    lab, drug, base = cell.split("/", 2)
    hits = glob.glob(os.path.join(DB, lab, "*", drug, "**", base + ".csv.zip"), recursive=True)
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    s2 = pd.read_csv(os.path.join(HERE, "stage2_final_cell_metrics.csv"))
    qc = s2[(s2.qc_pass == True) & s2.lab.isin(["lab_3", "lab_4", "lab_5"])].copy()
    s4 = pd.read_csv(os.path.join(HERE, "stage4a_per_cell_predictions.csv"))
    qc = qc.merge(s4[["cell", "conc_uM", "drug_real"]], on="cell", how="left")
    loo = build_loo()
    todo, skip = [], {"no_zip": 0, "no_loo": 0, "no_conc": 0}
    for r in qc.itertuples():
        zp = find_zip(r.cell)
        if zp is None:
            skip["no_zip"] += 1; continue
        if not np.isfinite(r.conc_uM):
            skip["no_conc"] += 1; continue
        key = (r.drug_real, r.lab)
        if key not in loo:
            skip["no_loo"] = skip.get("no_loo", 0) + 1; continue
        ic50, hh = loo[key]
        occ = float(r.conc_uM ** hh / (r.conc_uM ** hh + ic50 ** hh))
        todo.append((zp, r.cell, r.lab, r.drug_real, float(r.conc_uM), occ))
    print(f"目标 {len(qc)} 细胞 → 可跑 {len(todo)}（跳过 {skip}）", flush=True)

    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cell"])
                except Exception:
                    pass
    todo = [r for r in todo if r[1] not in done]
    print(f"断点：已完成 {len(done)}，本轮 {len(todo)}", flush=True)

    t0 = time.time()
    with open(CKPT, "a", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, res in enumerate(ex.map(proc_cell, todo, chunksize=4)):
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
                if time.time() - t0 > a.budget:
                    print(f"  [budget] {a.budget}s 到点自停", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    print(f"本轮耗时 {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
