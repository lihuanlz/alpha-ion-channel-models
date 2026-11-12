# -*- coding: utf-8 -*-
"""
统一重算 Batch 3：Lei 全量五温度 hERG 群体（670 孔）剖面似然重提
方法基座 = lei211_J3_提取.py（staircase 下跳 τ_deact / 上跳 τ_act / sinactiv τ_rec DoE）
升级点：
  1) 双指数网格拟合保留，但对 τ1/τ2 加剖面似然 95% CI
     （固定一个 τ，另一个网格扫描 + 线性幅度最小二乘 → SSE 剖面 → F 判据找交点）
  2) DoE（τ_rise/τ_decay）同样剖面似然
  3) 输出逐孔逐电压行级表 → 统一总表格式
数据：09_历史归档/.../数据/Lei全量（protocol/*.csv + qc/selected-*.txt）
温度映射：herg25oc1=25, herg27oc1=27, herg30oc1=30, herg33oc1=33, herg37oc3/4=37
跑法：python batch3_lei_剖面升级.py [--batch herg37oc3] [--budget 250 --workers 20]
"""
import os, json, glob, time, argparse, warnings
import numpy as np
import pandas as pd
from scipy import stats as sst
from multiprocessing import Pool
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档\2026-10-16_清理\04_细胞线4\_归档\数据\Lei全量"
OUT = HERE
CKPT = os.path.join(OUT, "lei3_ckpt.jsonl")

DT_I = 2e-4
DT_V = 1e-4
BATCH_TEMP = {"herg25oc1": 25, "herg27oc1": 27, "herg30oc1": 30,
              "herg33oc1": 33, "herg37oc3": 37, "herg37oc4": 37}
STAIR_SEQ = [-40, -60, -20, -40, 0, -20, 20, 0, 40, 20, 40, 0, 20, -20, 0, -40, -20, -60, -40]
SIN_V = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]

# ---------- 剖面似然：双指数（τ1<τ2） ----------
def fit_biexp_prof(tt, y, sig, alpha=0.05):
    """带锚双指数网格 + τ1/τ2 剖面似然 CI。
    SSE(τ1,τ2)：两 τ 固定时 (y0,a1,a2) 线性。剖面：固定 τ1 扫 τ2 取 min，反之亦然。"""
    n = len(tt); p = 5
    if n < p + 5:
        return None
    E = lambda tau: np.exp(-tt / tau)
    def solve(t1, t2):
        X = np.column_stack([np.ones(n), E(t1), E(t2)])
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ sol
        return float(r @ r), sol
    best = None
    g1 = np.exp(np.linspace(np.log(0.005), np.log(0.3), 18))
    for t1 in g1:
        for t2 in np.exp(np.linspace(np.log(max(0.02, t1 * 1.8)), np.log(6.0), 22)):
            sse, sol = solve(t1, t2)
            if best is None or sse < best[0]:
                best = (sse, t1, t2, sol)
    if best is None:
        return None
    sse0, t1h, t2h, sol = best
    a0, a1, a2 = sol
    if abs(a1) + abs(a2) < 4 * sig:
        return None
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    def prof_t1():
        """固定 τ1 扫 τ2"""
        lo = g1[0]; hi = g1[-1]
        for cand in np.exp(np.linspace(np.log(0.005), np.log(t1h), 24))[::-1]:
            sse_c = min(solve(cand, t2)[0] for t2 in np.exp(np.linspace(np.log(max(0.02, cand * 1.8)), np.log(6.0), 16)))
            if sse_c > thresh:
                lo = cand; break
        for cand in np.exp(np.linspace(np.log(t1h), np.log(0.3), 16)):
            sse_c = min(solve(cand, t2)[0] for t2 in np.exp(np.linspace(np.log(max(0.02, cand * 1.8)), np.log(6.0), 16)))
            if sse_c > thresh:
                hi = cand; break
        return lo, hi
    def prof_t2():
        lo = 0.02; hi = 6.0
        for cand in np.exp(np.linspace(np.log(max(0.02, t1h * 1.8)), np.log(t2h), 24))[::-1]:
            sse_c = min(solve(t1, cand)[0] for t1 in np.exp(np.linspace(np.log(0.005), np.log(min(0.3, cand / 1.8)), 16)))
            if sse_c > thresh:
                lo = cand; break
        for cand in np.exp(np.linspace(np.log(t2h), np.log(6.0), 24)):
            sse_c = min(solve(t1, cand)[0] for t1 in np.exp(np.linspace(np.log(0.005), np.log(min(0.3, cand / 1.8)), 16)))
            if sse_c > thresh:
                hi = cand; break
        return lo, hi
    t1_lo, t1_hi = prof_t1()
    t2_lo, t2_hi = prof_t2()
    return {"tau1": float(t1h), "tau1_lo": float(t1_lo), "tau1_hi": float(t1_hi),
            "tau2": float(t2h), "tau2_lo": float(t2_lo), "tau2_hi": float(t2_hi),
            "a1": float(a1), "a2": float(a2), "y0": float(a0), "sse": sse0, "n": n}

def fit_doe_prof(tt, y, sig, alpha=0.05):
    """DoE y=y0+A*(e(-t/td)-e(-t/tr)) + tr/td 剖面似然。"""
    n = len(tt); p = 3
    if n < p + 5:
        return None
    def solve(tr, td):
        x = np.exp(-tt / td) - np.exp(-tt / tr)
        X = np.column_stack([np.ones(n), x])
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ sol
        return float(r @ r), sol
    best = None
    for tr in np.exp(np.linspace(np.log(0.001), np.log(0.08), 22)):
        for td in np.exp(np.linspace(np.log(max(0.004, tr * 1.5)), np.log(2.0), 26)):
            sse, sol = solve(tr, td)
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    if best is None:
        return None
    sse0, trh, tdh, sol = best
    if abs(sol[1]) < 4 * sig:
        return None
    thresh = sse0 * (1 + sst.f.ppf(1 - alpha, 1, n - p) / (n - p))
    def prof_tr():
        lo = 0.001; hi = 0.08
        for cand in np.exp(np.linspace(np.log(0.001), np.log(trh), 20))[::-1]:
            if min(solve(cand, td)[0] for td in np.exp(np.linspace(np.log(max(0.004, cand * 1.5)), np.log(2.0), 14))) > thresh:
                lo = cand; break
        for cand in np.exp(np.linspace(np.log(trh), np.log(0.08), 14)):
            if min(solve(cand, td)[0] for td in np.exp(np.linspace(np.log(max(0.004, cand * 1.5)), np.log(2.0), 14))) > thresh:
                hi = cand; break
        return lo, hi
    def prof_td():
        lo = 0.004; hi = 2.0
        for cand in np.exp(np.linspace(np.log(max(0.004, trh * 1.5)), np.log(tdh), 20))[::-1]:
            if min(solve(tr, cand)[0] for tr in np.exp(np.linspace(np.log(0.001), np.log(min(0.08, cand / 1.5)), 14))) > thresh:
                lo = cand; break
        for cand in np.exp(np.linspace(np.log(tdh), np.log(2.0), 20)):
            if min(solve(tr, cand)[0] for tr in np.exp(np.linspace(np.log(0.001), np.log(min(0.08, cand / 1.5)), 14))) > thresh:
                hi = cand; break
        return lo, hi
    tr_lo, tr_hi = prof_tr()
    td_lo, td_hi = prof_td()
    return {"tau_r": float(trh), "tau_r_lo": float(tr_lo), "tau_r_hi": float(tr_hi),
            "tau_d": float(tdh), "tau_d_lo": float(td_lo), "tau_d_hi": float(td_hi),
            "A": float(sol[1]), "y0": float(sol[0]), "sse": sse0, "n": n}

# ---------- 数据装载 ----------
def load_protocol(name):
    d = np.genfromtxt(os.path.join(DATA, "protocol", f"protocol-{name}.csv"),
                      delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1:]

def load_current(batch, proto, well):
    p = os.path.join(DATA, batch, proto, f"{batch}-{proto}-{well}.csv")
    if not os.path.exists(p):
        p = os.path.join(DATA, proto, f"{batch}-{proto}-{well}.csv")  # 25oc1 旧布局
    d = np.genfromtxt(p, delimiter=",", skip_header=1)
    if d.ndim == 1:
        d = d[:, None]
    return d

def releak_sweep(i5, v10):
    v = v10[::2][:len(i5)]
    v0 = v[0]
    wa, wb = int(0.175 / DT_I), int(0.2 / DT_I)
    mv = np.mean(v[wa:wb] - v0)
    g = 0.0 if abs(mv) < 1e-9 else -np.mean(i5[wa:wb]) / mv
    return i5 + g * (v - v0), g

def segments(v10):
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]

def extract_well(args):
    batch, well = args
    try:
        rows = []
        _, Vs = load_protocol("staircaseramp")
        v10 = Vs[:, 0]
        Is = load_current(batch, "staircaseramp", well)[:, 0]
        segs = segments(v10)
        sig_s = float(np.std(Is[: int(0.2 / DT_I)]))
        step_segs = [(vv, a, b) for vv, a, b in segs if 0.45 <= (b - a) * DT_V <= 0.55]
        # 下跳 → τ_deact
        for k in range(1, len(step_segs)):
            vv, a, b = step_segs[k]
            vp = step_segs[k - 1][0]
            if vv in (-40, -60) and vp > vv:
                seg = Is[a // 2: b // 2]
                tt = np.arange(len(seg)) * DT_I
                r = fit_biexp_prof(tt, seg, sig_s)
                if r:
                    for comp, pnm in [("tau1", "tau_deact_fast"), ("tau2", "tau_deact_slow")]:
                        rows.append({"parameter": pnm, "process": "deactivation",
                                     "voltage_mV": vv, "value_s": r[comp],
                                     "prof_lo_s": r[comp + "_lo"], "prof_hi_s": r[comp + "_hi"],
                                     "r2": np.nan, "model": "biexp_anchored"})
        # 大补跳 -80→+40 → τ_act/τ_inact
        for vv, a, b in segs:
            if vv == 40 and 0.9 <= (b - a) * DT_V <= 1.1:
                seg = Is[a // 2: b // 2]
                tt = np.arange(len(seg)) * DT_I
                r = fit_biexp_prof(tt, seg, sig_s)
                if r:
                    rows.append({"parameter": "tau_act_fast", "process": "activation",
                                 "voltage_mV": 40, "value_s": r["tau1"],
                                 "prof_lo_s": r["tau1_lo"], "prof_hi_s": r["tau1_hi"],
                                 "r2": np.nan, "model": "biexp_anchored|bigstep"})
                    rows.append({"parameter": "tau_inact", "process": "inactivation",
                                 "voltage_mV": 40, "value_s": r["tau2"],
                                 "prof_lo_s": r["tau2_lo"], "prof_hi_s": r["tau2_hi"],
                                 "r2": np.nan, "model": "biexp_anchored|bigstep"})
                break
        # 阶梯上跳 +20 → τ_act
        for k in range(1, len(step_segs)):
            vv, a, b = step_segs[k]
            vp = step_segs[k - 1][0]
            if vv > vp and (vv - vp) == 20:
                seg = Is[a // 2: b // 2]
                tt = np.arange(len(seg)) * DT_I
                r = fit_biexp_prof(tt, seg, sig_s)
                if r:
                    rows.append({"parameter": "tau_act_fast", "process": "activation",
                                 "voltage_mV": vv, "value_s": r["tau1"],
                                 "prof_lo_s": r["tau1_lo"], "prof_hi_s": r["tau1_hi"],
                                 "r2": np.nan, "model": "biexp_anchored|stepup"})
        # sinactiv τ_rec DoE
        _, Vn = load_protocol("sinactiv")
        In = load_current(batch, "sinactiv", well)
        for sw in [0, 1, 2]:
            if sw >= In.shape[1]:
                break
            ic, _ = releak_sweep(In[:, sw], Vn[:, sw])
            segs_n = segments(Vn[:, sw])
            test = None
            for vv, a, b in segs_n:
                if vv == SIN_V[sw] and (b - a) * DT_V >= 0.4:
                    test = (a, b)
            if test is None:
                continue
            a, b = test[0] // 2, test[1] // 2
            seg = ic[a: a + int(0.15 / DT_I)]
            tt = np.arange(len(seg)) * DT_I
            r = fit_doe_prof(tt, seg, sig_s)
            if r:
                rows.append({"parameter": "tau_rec", "process": "recovery",
                             "voltage_mV": SIN_V[sw], "value_s": r["tau_d"],
                             "prof_lo_s": r["tau_d_lo"], "prof_hi_s": r["tau_d_hi"],
                             "r2": np.nan, "model": "DoE_anchored"})
        return {"ok": True, "batch": batch, "well": well, "sigma": sig_s, "rows": rows}
    except Exception as e:
        return {"ok": False, "batch": batch, "well": well, "err": f"{type(e).__name__}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--batch", type=str, default="")
    ap.add_argument("--max-tasks", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    batches = [a.batch] if a.batch else list(BATCH_TEMP.keys())
    tasks = []
    for b in batches:
        sel = os.path.join(DATA, "qc", f"selected-{b}.txt")
        if not os.path.exists(sel):
            print(f"{b}: 无 selected 名单，跳过"); continue
        wells = [l.strip() for l in open(sel, encoding="utf-8") if l.strip() and not l.startswith("#")]
        for w in wells:
            tasks.append((b, w))
    print(f"总孔任务 {len(tasks)}", flush=True)
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        done.add((r["batch"], r["well"]))
                except Exception:
                    pass
    todo = [t for t in tasks if t not in done]
    if a.max_tasks:
        todo = todo[:a.max_tasks]
    print(f"已完成 {len(done)}，待处理 {len(todo)}", flush=True)
    if not todo:
        print("ALL DONE"); finalize(); return
    n0 = 0
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(proc_one_star, todo, chunksize=4):
            with open(CKPT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n0 += 1
            if n0 % 50 == 0:
                print(f"  {n0}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("预算到点自停（可续跑）", flush=True)
                pool.terminate(); pool.join(); return
    print("ALL DONE", flush=True)
    finalize()

def proc_one_star(args):
    return extract_well(args)

def finalize():
    seen = {}
    with open(CKPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
                if r.get("ok"):
                    seen[(r["batch"], r["well"])] = r
            except Exception:
                pass
    rows = []
    for (batch, well), r in seen.items():
        for row in r["rows"]:
            rows.append({"source_big": "lei2019",
                         "dataset": batch,
                         "cell_id": f"{batch}/{well}",
                         "channel": "hERG", "family": "Kv", "species": "human|CHO",
                         "temp_C": BATCH_TEMP[batch],
                         "parameter": row["parameter"], "process": row["process"],
                         "voltage_mV": row["voltage_mV"],
                         "value": row["value_s"] * 1000.0,  # s → ms
                         "se_pcov": np.nan,
                         "prof_lo": row["prof_lo_s"] * 1000.0,
                         "prof_hi": row["prof_hi_s"] * 1000.0,
                         "model": row["model"], "p_model": np.nan, "r2": row["r2"],
                         "qc_flag": "ok", "method": "profile_v1",
                         "drug": "", "blind": False, "A_peak_pA": np.nan})
    df = pd.DataFrame(rows)
    fp = os.path.join(OUT, "lei3_unified.csv")
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"落盘 {fp}：{len(df)} 行，孔数 {len(seen)}")
    print(df.groupby(["parameter"])["value"].agg(["count", "median"]).to_string())

if __name__ == "__main__":
    main()
