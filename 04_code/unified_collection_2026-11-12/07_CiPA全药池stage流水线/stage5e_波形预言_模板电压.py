# -*- coding: utf-8 -*-
# stage5e_波形预言_模板电压.py
# ============================================================================
# Stage5e：把 Stage4b 的波形级前向预言扩到 lab_1/lab_2。
# 这两室电流 zip 无电压列（Stage4b 的 176 个"无电压列"失败即源于此），
# 但官方 ted.xlsx 的 IntendedVoltageProtocol 表给了协议模板 V(t)：
#   lab_1 全 30 目录同一模板(5000点)；lab_2 两种(19968/19840点)。
# 做法：模板按 (lab,drug) 读一次，np.interp 到电流时间轴，其余逐字复用
# stage4b 的冻结管线（α模型/τ窗程序/LOO占据/G最小二乘）。
# 温度：lab_1/2 无波形温度列，用 stage5b_cell_properties_v2.csv 的 Temp。
# 用法：python stage5e_波形预言_模板电压.py --budget 250 --workers 16
# ============================================================================
import os, json, glob, time, argparse, zipfile, importlib.util
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
XLSXROOT = os.path.join(ROOT, "CiPA全药库_全量_2026-11-09")
CKPT = os.path.join(HERE, "stage5e_ckpt.jsonl")
OUTCSV = os.path.join(HERE, "stage5e_per_cell_waveform.csv")

# importlib 加载 stage4b（文件名含中文）
spec = importlib.util.spec_from_file_location("stage4b", os.path.join(HERE, "stage4b_波形级前向预言.py"))
s4b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4b)
E_REV = s4b.E_REV

# ---------------------------------------------------------- 协议模板
def load_proto(lab, drug):
    """从 ted.xlsx 的 IntendedVoltageProtocol 读模板电压。返回 (v_proto, dt_ms) 或 None。"""
    hits = glob.glob(os.path.join(XLSXROOT, lab, "*", drug, "subtracted", "ted", "ted.xlsx"))
    if not hits:
        return None
    try:
        vp = pd.read_excel(hits[0], sheet_name="IntendedVoltageProtocol")
        t = vp.iloc[:, 0].to_numpy(float)
        v = vp.iloc[:, 1].to_numpy(float)
        if len(t) < 100:
            return None
        dt = float(np.median(np.diff(t)))
        return v.astype(np.float64), dt
    except Exception:
        return None

# ---------------------------------------------------------- 逐细胞处理
def proc_cell(rec):
    zpath, cell, lab, drug_real, conc_uM, occ, T_file, v_proto, dt_proto = rec
    try:
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        t = df["t_ms"].to_numpy(float)
        sweep_ms = float(t.max() - t.min())
        dt_s = float(np.median(np.diff(t))) / 1000.0
        cur = {int(c.split("#")[1].split("_")[0]): c for c in df.columns
               if c.startswith("trace_#") and "_current_" in c}
        if len(cur) < 8:
            return dict(ok=False, cell=cell, err="traces<8")
        # 模板电压插值到电流时间轴（所有 sweep 共用同一协议模板）
        t_proto = np.arange(len(v_proto)) * dt_proto
        v_t = np.interp(t, t_proto, v_proto)
        uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(
            list(cur.values())[0].rsplit("_current_", 1)[-1].lower())
        if uf is None:
            return dict(ok=False, cell=cell, err="单位未知")
        tabs = s4b.build_tabs(T_file)

        trs = sorted(cur)
        ys, vs, stds = {}, {}, {}
        for tr in trs:
            y = df[cur[tr]].to_numpy(float) * uf
            if np.ptp(y) > 4000 or abs(np.max(y)) > 3000:
                continue
            ys[tr] = y; vs[tr] = v_t
            stds[tr] = float(np.std(y))
        if len(ys) < 8:
            return dict(ok=False, cell=cell, err="post-artifact<8")
        sarr = np.array(list(stds.values()))
        thr = max(5.0, 0.05 * np.percentile(sarr, 95))
        test_trs = sorted([k for k in trs if k in ys and stds[k] > thr])
        if len(test_trs) < 12:
            return dict(ok=False, cell=cell, err="test<12")
        # 分段（逐字同 stage4b/stage2d 冻结逻辑）
        Aps = {tr: s4b.tau_tail_of(t, ys[tr], sweep_ms)[0] for tr in test_trs}
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
            m, h = s4b.forward_dt(vs[tr], tabs, dt_s)
            gsh = m * h * (vs[tr] - E_REV)
            shapes[tr] = gsh
            okk = np.isfinite(ys[tr])
            num += float(np.sum(gsh[okk] * ys[tr][okk]))
            den += float(np.sum(gsh[okk] ** 2))
        if den <= 0:
            return dict(ok=False, cell=cell, err="G退化")
        G = num / den
        # T1：对照 τ 预言
        tau_mea_c, tau_sim_c, rms_c = [], [], []
        for tr in veh_use:
            i_sim = G * shapes[tr]
            _, taum = s4b.tau_tail_of(t, ys[tr], sweep_ms)
            _, taus = s4b.tau_tail_of(t, i_sim, sweep_ms)
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
            m, h = s4b.forward_dt(vs[tr], tabs, dt_s)
            i_sim = (1.0 - occ) * G * (m * h * (vs[tr] - E_REV))
            Am, taum = s4b.tau_tail_of(t, ys[tr], sweep_ms)
            As, taus = s4b.tau_tail_of(t, i_sim, sweep_ms)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    s2 = pd.read_csv(os.path.join(HERE, "stage2_final_cell_metrics.csv"))
    qc = s2[(s2.qc_pass == True) & s2.lab.isin(["lab_1", "lab_2"])].copy()
    s4 = pd.read_csv(os.path.join(HERE, "stage4a_per_cell_predictions.csv"))
    qc = qc.merge(s4[["cell", "conc_uM", "drug_real"]], on="cell", how="left")
    # 温度：stage5b v2
    s5b = pd.read_csv(os.path.join(HERE, "stage5b_cell_properties_v2.csv"))
    s5b["cell"] = s5b["lab"] + "/" + s5b["drug"] + "/" + s5b["CELLID"]
    qc = qc.merge(s5b[["cell", "Temp"]], on="cell", how="left")
    loo = s4b.build_loo()

    proto_cache = {}
    todo, skip = [], {"no_zip": 0, "no_loo": 0, "no_conc": 0, "no_proto": 0}
    for r in qc.itertuples():
        zp = s4b.find_zip(r.cell)
        if zp is None:
            skip["no_zip"] += 1; continue
        if not np.isfinite(r.conc_uM):
            skip["no_conc"] += 1; continue
        key = (r.drug_real, r.lab)
        if key not in loo:
            skip["no_loo"] += 1; continue
        lab, drug = r.lab, r.cell.split("/")[1]
        if (lab, drug) not in proto_cache:
            proto_cache[(lab, drug)] = load_proto(lab, drug)
        proto = proto_cache[(lab, drug)]
        if proto is None:
            skip["no_proto"] += 1; continue
        v_proto, dt_proto = proto
        ic50, hh = loo[key]
        occ = float(r.conc_uM ** hh / (r.conc_uM ** hh + ic50 ** hh))
        T = float(r.Temp) if np.isfinite(getattr(r, "Temp", np.nan)) else 36.5
        if not (15.0 < T < 45.0):
            T = 36.5
        todo.append((zp, r.cell, lab, r.drug_real, float(r.conc_uM), occ, T, v_proto, dt_proto))
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
