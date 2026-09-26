# -*- coding: utf-8 -*-
"""
Nα药 · Nav1.5 药物形状调制判决（预注册 v1.0 · 2026-09-15）
数据源：Tran et al. 2020 PLOS ONE，OSF tjuev，本机 04_细胞线4\\数据\\nav15_drug_tjuev\\
用法（Spyder）：
  冒烟：%runfile '.../2026-09-15_Nα药_Nav15药物形状调制判决.py' --wdir   （首行 SMOKE=True）
  正式：把 SMOKE 改 False 后同一命令
"""
import json, os, re, sys
import numpy as np

SMOKE = False  # 正式跑时改 False（冒烟已绿：C1 过、TTX 残余 0.13–0.17、epoch 核实、浓度-阻断与论文三点互洽）

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "数据", "nav15_drug_tjuev")
CELLS_JSON = os.path.join(ROOT, "α模型", "nav_drug_cells_parsed.json")
TAG = "冒烟" if SMOKE else "结果"
OUT_JSON = os.path.join(ROOT, "α模型", f"2026-09-15_Nα药_Nav15药物形状调制判决_{TAG}.json")
OUT_CSV = os.path.join(ROOT, "α模型", f"2026-09-15_Nα药_Nav15药物形状调制判决_逐条件表_{TAG}.csv")
OUT_PNG = os.path.join(ROOT, "α模型", f"2026-09-15_Nα药_Nav15药物形状调制判决_{TAG}.png")

import pyabf
from scipy.optimize import curve_fit
from scipy.stats import wilcoxon
from scipy.signal import medfilt

# ---------------- epoch（预注册 §二 + v1.1 修订钉死，ms） ----------------
BASE = (10.0, 230.0)     # −95 mV 基线窗
V120 = (300.0, 440.0)    # −120 mV 平台（R_input / Ipassive 锚）
VPLAT = (460.0, 480.0)   # −15 mV 平台（v_check 显示用，v1.1）
STEP_PK = (445.5, 450.0) # INaP 峰窗（v1.1：排除 445.2–445.5 电容伪影带）
WF = (445.5, 465.5)      # 形状比较域 20 ms（v1.1 同因），400 点 @20 kHz
RAMP = (692.0, 783.0)    # INaL 斜坡窗
LASTN = 5                # 每条件末 5 迹（§三）

PAPER = {  # 判4 锚：论文 Table 1（预注册 §五抄录）
    "Buprenorphine":    {"P": (31.4, 1.1), "L": (19.4, 1.0)},
    "Norbuprenorphine": {"P": (8.0, 1.0),  "L": (4.0, 1.1)},
    "Methadone":        {"P": (40.1, 1.3), "L": (8.5, 0.9)},
    "Naltrexone":       {"P": (139.5, 0.8),"L": (148.5, 0.8)},
    "Naloxone":         {"P": (451.8, 0.9),"L": (284.7, 1.0)},
}

_rng = np.random.default_rng(20260915)

# ---------------- 波形级 ----------------
def _win(t, w):
    return (t >= w[0] / 1e3) & (t < w[1] / 1e3)

def parse_sweep(a, s):
    """返回 dict(t,i,v,T)；自动识别 pA/mV 通道。"""
    chI = chV = None
    for ch in range(a.channelCount):
        a.setSweep(s, channel=ch)
        u = str(getattr(a, "sweepUnitsY", ""))
        if "pA" in u:
            chI = ch
        elif "mV" in u:
            chV = ch
    if chI is None:
        chI = 0
    if chV is None:
        chV = 1 if a.channelCount > 1 else 0
    a.setSweep(s, channel=chI)
    i = a.sweepY.astype(float)
    t = a.sweepX.astype(float)
    a.setSweep(s, channel=chV)
    v = a.sweepY.astype(float)
    return t, i, v

def sweep_metrics(t, i, v):
    """单 sweep 的全部口径量（§三 + v1.1：峰/波形先 3 点中值滤波）。"""
    mb = _win(t, BASE); m2 = _win(t, V120); mp = _win(t, STEP_PK); mw = _win(t, WF); mr = _win(t, RAMP)
    mpl = _win(t, VPLAT)
    if mb.sum() < 50 or m2.sum() < 50 or mp.sum() < 10 or mw.sum() < 300 or mr.sum() < 500:
        return None
    i95 = float(i[mb].mean())
    i120 = float(i[m2].mean())
    iseg = medfilt(i - i95, 3)  # v1.1：灭单点电容伪迹
    step_pk = float(iseg[mp].min())
    wf = iseg[mw]
    # Ipassive：两点 (−95,i95)/(−120,i120) 直线，逐点外推斜坡电压
    slope = (i120 - i95) / (-120.0 + 95.0)
    ipass = i95 + (v[mr] + 95.0) * slope
    ramp_pk = float((i[mr] - ipass).min())
    r_in = 25000.0 / abs(i120 - i95) if abs(i120 - i95) > 1e-6 else np.nan  # MΩ
    return {"step_pk": step_pk, "wf": wf, "ramp_pk": ramp_pk, "r_in": r_in,
            "v_step": float(v[mpl].mean()), "v_120": float(v[m2].mean()), "v_base": float(v[mb].mean())}

def mean_wf(traces):
    """末 N 迹平均波形，按 |峰| 归一。"""
    w = np.mean([x["wf"] for x in traces], axis=0)
    pk = float(np.min(w))
    if pk > -1e-9:
        return None, pk
    return w / abs(pk), pk

def eps_kappa(wc, wd):
    """ε_x + κ*（单参数时间伸缩）+ ε_x1（warp 后残差）。与 Cα-5 同码。"""
    ex = float(np.sqrt(np.mean((wc - wd) ** 2)))
    best = (ex, 0.0)
    t = np.arange(len(wc))
    for kap in np.arange(-0.5, 0.5001, 0.005):
        bw = np.interp(t, t * (1.0 + kap), wd)
        r = float(np.sqrt(np.mean((wc - bw) ** 2)))
        if r < best[0]:
            best = (r, float(kap))
    return ex, best[1], best[0]

# ---------------- 元数据 ----------------
def parse_range(s):
    m = re.match(r"\s*(\d+)\s*[-–]\s*(\d+)", str(s))
    return (int(m.group(1)), int(m.group(2))) if m else None

def parse_conc(s):
    try:
        return float(re.match(r"\s*([\d.]+)", str(s)).group(1))
    except Exception:
        return None

def cell_files(grp, drug, file_id):
    names = [x.strip() for x in re.split(r"[&,]", str(file_id)) if x.strip().lower().endswith(".abf")]
    out = []
    for n in names:
        p = os.path.join(DATA, grp, drug, n)
        if not os.path.exists(p):
            return None, n
        out.append(p)
    return (out or None), None

# ---------------- 细胞级 ----------------
def note_bad_sweep(row):
    """精细闸：从备注解析坏事件起始 sweep（1-based）。
    返回 None=无事件；'EXCLUDE'=无法定位（保守剔除）；int=坏事件起点。"""
    note = str(row.get("note", ""))
    if not re.search(r"died|lost clamp|unstable", note, re.I):
        return None
    if re.search(r"ttx", note, re.I):
        rt = parse_range(row.get("sw_ttx", ""))
        if rt:
            return rt[0]
    if re.search(r"2nd concentration", note, re.I):
        r2 = parse_range(row.get("sw2", ""))
        if r2:
            return r2[0]
    for pat in [r"sweep\s*#?\s*(\d+)", r"at\s+(?:sweep\s*#?\s*)?(\d+)", r"during\s+(\d+)"]:
        m = re.search(pat, note, re.I)
        if m:
            return int(m.group(1))
    return "EXCLUDE"

def process_cell(grp, row, boot=200, refined=False):
    """返回细胞记录（含质量门判决与全部条件量）。refined=True 走 v1.2 精细备注闸。"""
    drug = row["drug"].strip()
    rec = {"group": grp, "drug": drug, "file_id": row["file_id"], "note": row.get("note", "")}
    # 质量门：备注（v1.0 粗闸 / v1.2 精细闸）
    bad_sw = None
    if re.search(r"died|lost clamp|unstable", str(row.get("note", "")), re.I):
        if not refined:
            rec["exclude"] = "note:" + row["note"][:60]
            return rec
        bad_sw = note_bad_sweep(row)
        if bad_sw == "EXCLUDE":
            rec["exclude"] = "note不可定位:" + row["note"][:50]
            return rec
    files, missing = cell_files(grp, drug, row["file_id"])
    if files is None:
        rec["exclude"] = f"缺文件 {missing}"
        return rec
    # 拼接 sweeps
    sweeps = []
    for fp in files:
        a = pyabf.ABF(fp)
        for s in range(a.sweepCount):
            sweeps.append(parse_sweep(a, s))
    rec["n_sweeps"] = len(sweeps)
    # 条件区间
    r1 = parse_range(row.get("sw1", "")); r2 = parse_range(row.get("sw2", "")); rt = parse_range(row.get("sw_ttx", ""))
    c1 = parse_conc(row.get("c1", "")); c2 = parse_conc(row.get("c2", ""))
    if r1 is None or c1 is None:
        rec["exclude"] = "浓度/区间解析失败"
        return rec
    ctrl_end = r1[0] - 1
    if ctrl_end < 8:
        rec["exclude"] = f"对照不足({ctrl_end})"
        return rec
    if r1[1] > len(sweeps) or (r2 and r2[1] > len(sweeps)) or (rt and rt[1] > len(sweeps)):
        rec["exclude"] = "区间超出总sweeps"
        return rec
    # v1.2 精细闸：坏事件起点落在哪段，哪段及之后作废
    if bad_sw is not None and ctrl_end >= bad_sw:
        rec["exclude"] = f"对照段受损(坏事件@{bad_sw})"
        return rec

    def cond_traces(rng2):
        idx = list(range(rng2[0], rng2[1] + 1))[-LASTN:]
        if len(idx) < 3:
            return None
        out = []
        for g in idx:
            m = sweep_metrics(*sweeps[g - 1])
            if m is not None:
                out.append(m)
        return out if len(out) >= 3 else None

    ctrl_idx = list(range(1, ctrl_end + 1))
    ctrl_all = [sweep_metrics(*sweeps[g - 1]) for g in ctrl_idx]
    ctrl_all = [m for m in ctrl_all if m is not None]
    ctrl_tr = cond_traces((1, ctrl_end))
    if ctrl_tr is None or len(ctrl_all) < 6:
        rec["exclude"] = "对照迹无效"
        return rec
    ctrl_pk = np.array([m["step_pk"] for m in ctrl_tr])
    rec["ctrl_pk_pA"] = float(np.mean(ctrl_pk))
    rec["ctrl_pk_cv"] = float(np.std(ctrl_pk) / abs(np.mean(ctrl_pk)))
    rec["r_in_MΩ"] = float(np.nanmedian([m["r_in"] for m in ctrl_tr]))
    rec["v_check"] = [round(float(np.mean([m["v_base"] for m in ctrl_tr])), 1),
                      round(float(np.mean([m["v_step"] for m in ctrl_tr])), 1),
                      round(float(np.mean([m["v_120"] for m in ctrl_tr])), 1)]
    # 质量门
    if rec["ctrl_pk_cv"] > 0.15:
        rec["exclude"] = f"对照峰CV={rec['ctrl_pk_cv']:.3f}"
        return rec
    if abs(rec["ctrl_pk_pA"]) < 500:
        rec["exclude"] = f"对照峰过小({rec['ctrl_pk_pA']:.0f}pA)"
        return rec
    if not (50 <= rec["r_in_MΩ"] <= 500):
        rec["exclude"] = f"R_in={rec['r_in_MΩ']:.0f}MΩ"
        return rec

    wc, ctrl_pk_wf = mean_wf(ctrl_tr)
    if wc is None:
        rec["exclude"] = "对照波形外向"
        return rec
    rec["wc"] = wc
    ctrl_ramp = float(np.mean([m["ramp_pk"] for m in ctrl_tr]))

    # TTX 锚（v1.2：坏事件在 TTX 段内/前则整锚作废）
    if rt and (bad_sw is None or rt[1] < bad_sw):
        tt = cond_traces(rt)
        if tt:
            rec["ttx_resid_step"] = abs(float(np.mean([m["step_pk"] for m in tt])) / ctrl_pk_wf)
            rec["ttx_resid_ramp"] = abs(float(np.mean([m["ramp_pk"] for m in tt])) / ctrl_ramp) if ctrl_ramp < 0 else None

    # Null-B 自助：对照段随机二分 → ε_x / Δf 零分布
    null_ex, null_df = [], []
    n_sw = len(ctrl_all)
    for _ in range(boot):
        perm = _rng.permutation(n_sw)
        ha = [ctrl_all[j] for j in perm[: n_sw // 2]]
        hb = [ctrl_all[j] for j in perm[n_sw // 2:]]
        wa, pka = mean_wf(ha); wb, pkb = mean_wf(hb)
        if wa is None or wb is None:
            continue
        ex, _, _ = eps_kappa(wa, wb)
        null_ex.append(ex)
        ra = float(np.mean([m["ramp_pk"] for m in ha])); rb = float(np.mean([m["ramp_pk"] for m in hb]))
        if ra < 0 and pka != 0:
            null_df.append((1 - abs(rb) / abs(ra)) - (1 - abs(pkb) / abs(pka)))
    rec["null_ex"] = null_ex
    rec["null_df"] = null_df
    rec["null_ex_q95"] = float(np.percentile(null_ex, 95)) if null_ex else None
    rec["null_df_med"] = float(np.median(null_df)) if null_df else None

    # 药物条件（v1.2：坏事件起点及之后所在条件作废）
    rec["conds"] = []
    for conc, rr in [(c1, r1), (c2, r2)]:
        if conc is None or rr is None:
            continue
        if bad_sw is not None and rr[1] >= bad_sw:
            rec["conds"].append({"conc_uM": conc, "range": rr, "valid": False, "why": f"坏事件@{bad_sw}"})
            continue
        tr = cond_traces(rr)
        if tr is None:
            rec["conds"].append({"conc_uM": conc, "range": rr, "valid": False})
            continue
        wd, pk = mean_wf(tr)
        f_step = 1.0 - abs(pk) / abs(ctrl_pk_wf)
        ramp_d = float(np.mean([m["ramp_pk"] for m in tr]))
        f_ramp = (1.0 - abs(ramp_d) / abs(ctrl_ramp)) if ctrl_ramp < 0 and ramp_d < 0 else None
        ex = kap = ex1 = None
        if wd is not None:
            ex, kap, ex1 = eps_kappa(wc, wd)
        rec["conds"].append({"conc_uM": conc, "range": rr, "valid": True,
                             "f_step": float(f_step), "f_ramp": None if f_ramp is None else float(f_ramp),
                             "eps_x": ex, "kappa": kap, "eps_x1": ex1,
                             "df": None if f_ramp is None else float(f_ramp - f_step)})
    return rec

# ---------------- 冒烟 C1 / Null-A ----------------
def synth_wf(n=400, ts=1.2, tf=6.0, C=0.02, warp=0.0, morph=0.0, noise=0.0):
    t = np.arange(n) / 20.0  # ms
    tw = t * (1.0 + warp)
    rise = 1.0 - np.exp(-tw / 0.25)
    decay = (1 - C) * np.exp(-tw / (ts if morph < 0.5 else tf)) + C
    y = -rise * decay
    y = y / abs(y.min())
    if noise > 0:
        y = y + _rng.normal(0, noise, n)
    return y

def c1_check():
    w0 = synth_wf()
    w_amp = synth_wf() * 1.0  # 纯幅度缩放在归一化后应为 0 形状差
    ex0, _, _ = eps_kappa(w0, w_amp)
    w_w = synth_wf(warp=0.20)
    exw, kap, _ = eps_kappa(w0, w_w)
    ok1 = ex0 < 0.01
    ok2 = abs(kap - 0.20) / 0.20 < 0.20
    print(f"  C1 纯幅度缩放 ε_x={ex0:.5f}（判线<0.01）{'过' if ok1 else '挂'}", flush=True)
    print(f"  C1 τ变形 κ*=+0.20 → 回收 {kap:+.3f}（误差<20%）{'过' if ok2 else '挂'}", flush=True)
    return ok1 and ok2

# ---------------- 判词 ----------------
def verdicts(recs):
    out = {}
    valid = [r for r in recs if "exclude" not in r and r.get("conds")]
    # 判1 池（INa_P，f_step≥0.15）
    poolP = []
    for r in valid:
        if r["group"] != "INa_P":
            continue
        for cd in r["conds"]:
            if cd.get("valid") and cd["f_step"] is not None and cd["f_step"] >= 0.15 and cd["eps_x"] is not None:
                poolP.append((r, cd))
    exs = np.array([cd["eps_x"] for _, cd in poolP])
    nulls = np.array([r["null_ex_q95"] for r, _ in poolP if r["null_ex_q95"] is not None])
    allnull = np.concatenate([np.array(r["null_ex"]) for r, _ in poolP if r["null_ex"]])
    q95 = float(np.percentile(allnull, 95)) if len(allnull) else None
    frac_over = float(np.mean([cd["eps_x"] > r["null_ex_q95"] for r, cd in poolP if r["null_ex_q95"] is not None])) if len(poolP) else None
    wp = None
    if len(exs) >= 8 and len(exs) == len(nulls):
        try:
            wp = float(wilcoxon(exs - nulls, alternative="greater").pvalue)
        except Exception:
            wp = None
    out["J1"] = {"n_pool": len(poolP), "eps_x_med": float(np.median(exs)) if len(exs) else None,
                 "nullB_Q95": q95, "frac_over_own": frac_over, "wilcoxon_p": wp,
                 "PASS": bool(q95 is not None and len(exs) and np.median(exs) > q95 and frac_over is not None and frac_over >= 0.70 and wp is not None and wp < 0.01)}
    # 判2（同池）
    kaps = np.array([abs(cd["kappa"]) for _, cd in poolP if cd["kappa"] is not None])
    out["J2"] = {"n": len(kaps), "abs_kappa_med": float(np.median(kaps)) if len(kaps) else None,
                 "PASS": bool(len(kaps) and np.median(kaps) < 0.10)}
    # 判3（INa_L 配对，f_step≥0.15）
    poolL = []
    for r in valid:
        if r["group"] != "INa_L":
            continue
        for cd in r["conds"]:
            if cd.get("valid") and cd["f_step"] is not None and cd["f_step"] >= 0.15 and cd["df"] is not None:
                poolL.append((r, cd))
    dfs = np.array([cd["df"] for _, cd in poolL])
    alldf = np.concatenate([np.array(r["null_df"]) for r, _ in poolL if r["null_df"]]) if poolL else np.array([])
    q95df = float(np.percentile(np.abs(alldf), 95)) if len(alldf) else None
    wp3 = None
    if len(dfs) >= 8:
        try:
            wp3 = float(wilcoxon(np.abs(dfs), alternative="greater").pvalue) if q95df is not None and np.median(np.abs(dfs)) > 0 else None
        except Exception:
            wp3 = None
    out["J3"] = {"n_pool": len(poolL), "abs_df_med": float(np.median(np.abs(dfs))) if len(dfs) else None,
                 "df_med": float(np.median(dfs)) if len(dfs) else None,
                 "null_Q95": q95df, "wilcoxon_p": wp3,
                 "PASS": bool(q95df is not None and len(dfs) and np.median(np.abs(dfs)) > q95df and wp3 is not None and wp3 < 0.01)}
    # 判4（Hill 对拍）
    def hill(C, ic50, n):
        return np.power(C, n) / (np.power(ic50, n) + np.power(C, n))
    j4 = {"detail": {}, "in_band": 0, "total": 0}
    for drug, dd in PAPER.items():
        for cur, gname in [("P", "INa_P"), ("L", "INa_L")]:
            pts = []
            for r in valid:
                if r["drug"] != drug or r["group"] != gname:
                    continue
                for cd in r["conds"]:
                    if not cd.get("valid"):
                        continue
                    f = cd["f_step"] if cur == "P" else cd["f_ramp"]
                    if f is not None:
                        pts.append((cd["conc_uM"], f))
            j4["total"] += 1
            key = f"{drug}_{cur}"
            if len(pts) < 3 or len(set(p[0] for p in pts)) < 2:
                j4["detail"][key] = {"note": f"点不足({len(pts)})", "paper": dd[cur][0]}
                continue
            Cs = np.array([p[0] for p in pts]); fs = np.array([p[1] for p in pts])
            try:
                p_, _ = curve_fit(hill, Cs, fs, p0=[dd[cur][0], 1.0],
                                  bounds=([1e-3, 0.3], [1e6, 3.0]), maxfev=20000)
                ic = float(p_[0])
                ok = 0.5 * dd[cur][0] <= ic <= 1.5 * dd[cur][0]
                j4["detail"][key] = {"ours_IC50": ic, "nH": float(p_[1]), "paper": dd[cur][0],
                                     "n_pts": len(pts), "in_band": bool(ok)}
                j4["in_band"] += int(ok)
            except Exception as e:
                j4["detail"][key] = {"note": "拟合失败", "paper": dd[cur][0]}
    j4["PASS"] = bool(j4["in_band"] >= 8)
    out["J4"] = j4
    return out

# ---------------- 主流程 ----------------
def load_rows():
    d = json.load(open(CELLS_JSON, encoding="utf-8"))
    rows = []
    for grp in ["INa_P", "INa_L"]:
        for row in d[grp]:
            if not str(row.get("drug", "")).strip():
                continue
            rows.append((grp, row))
    return rows

def main():
    print("=" * 72, flush=True)
    print(f" Nα药 · Nav1.5 药物形状调制判决（预注册 v1.0）  模式: {'冒烟' if SMOKE else '正式'}", flush=True)
    print("=" * 72, flush=True)
    print("[自检] C1 合成波形", flush=True)
    c1_ok = c1_check()

    rows = load_rows()
    if SMOKE:
        seen, picked = set(), []
        for grp, row in rows:
            key = (grp, row["drug"])
            if key not in seen and str(row.get("sw_ttx", "")).strip():
                seen.add(key)
                picked.append((grp, row))
        rows = picked[:6]
        print(f"[冒烟] 每组首个带TTX细胞，共 {len(rows)} 个", flush=True)
    else:
        print(f"[正式] 细胞行 {len(rows)}", flush=True)

    recs = []
    for grp, row in rows:
        r = process_cell(grp, row, boot=100 if SMOKE else 200)
        recs.append(r)
        tag = r.get("exclude")
        if tag:
            print(f"  [{grp}/{row['drug']}] 剔除: {tag}", flush=True)
        else:
            conds = " | ".join(
                f"{cd['conc_uM']}µM f_step={cd.get('f_step'):.2f}" + (f" f_ramp={cd['f_ramp']:.2f}" if cd.get("f_ramp") is not None else "")
                for cd in r["conds"] if cd.get("valid"))
            ttx = f" TTX残余step={r.get('ttx_resid_step'):.3f}" if r.get("ttx_resid_step") is not None else ""
            print(f"  [{grp}/{row['drug']}] {r['file_id'][:40]} n={r['n_sweeps']} 峰={r['ctrl_pk_pA']:.0f}pA CV={r['ctrl_pk_cv']:.3f} R={r['r_in_MΩ']:.0f}MΩ V={r['v_check']}{ttx} | {conds}", flush=True)

    res = {"smoke": SMOKE, "c1_ok": c1_ok, "n_rows": len(rows),
           "n_valid": sum(1 for r in recs if "exclude" not in r),
           "excluded": [{"group": r["group"], "drug": r["drug"], "file_id": r["file_id"], "why": r["exclude"]} for r in recs if "exclude" in r]}
    if not SMOKE:
        # ---- v1.0 主判臂（粗备注闸） ----
        res["verdicts"] = verdicts(recs)
        print("\n[判词组件 · v1.0 主判臂]", flush=True)
        print(json.dumps(res["verdicts"], ensure_ascii=False, indent=1), flush=True)
        # ---- v1.2 敏感性臂（精细备注闸） ----
        print("\n[敏感性臂 v1.2] 精细备注闸重处理……", flush=True)
        recs2 = [process_cell(grp, row, boot=200, refined=True) for grp, row in load_rows()]
        res["verdicts_v12"] = verdicts(recs2)
        res["n_valid_v12"] = sum(1 for r in recs2 if "exclude" not in r)
        print(f"[敏感性臂 v1.2] 有效 {res['n_valid_v12']}（主判臂 {res['n_valid']}）", flush=True)
        print(json.dumps(res["verdicts_v12"], ensure_ascii=False, indent=1), flush=True)
        # 逐条件 CSV（v1.0 臂）
        with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
            f.write("group,drug,file_id,conc_uM,f_step,f_ramp,eps_x,kappa,eps_x1,df,ttx_resid_step\n")
            for r in recs:
                if "exclude" in r:
                    continue
                for cd in r["conds"]:
                    if not cd.get("valid"):
                        continue
                    f.write(f"{r['group']},{r['drug']},{r['file_id'][:36]},{cd['conc_uM']},{cd.get('f_step')},{cd.get('f_ramp')},{cd.get('eps_x')},{cd.get('kappa')},{cd.get('eps_x1')},{cd.get('df')},{r.get('ttx_resid_step')}\n")
        print(f"[落盘] {OUT_CSV}", flush=True)
        # ---- 图 ----
        try:
            import matplotlib
            matplotlib.use("Agg")
            import sys as _sys
            from pathlib import Path as _P
            _sys.path.insert(0, str(_P(_sys.executable).parent.parent.parent))
            from daimon_runtime import setup_plot
            setup_plot()
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            v = res["verdicts"]
            # (a) ε_x vs Null-B
            a = ax[0, 0]
            poolP = [(r, cd) for r in recs if "exclude" not in r for cd in r.get("conds", [])
                     if r["group"] == "INa_P" and cd.get("valid") and cd.get("f_step") is not None and cd["f_step"] >= 0.15 and cd.get("eps_x") is not None]
            drugs = sorted(set(r["drug"] for r, _ in poolP))
            for j, dg in enumerate(drugs):
                ys = [cd["eps_x"] for r, cd in poolP if r["drug"] == dg]
                a.scatter([j] * len(ys), ys, s=40, alpha=0.75, label=dg)
            if v["J1"]["nullB_Q95"]:
                a.axhline(v["J1"]["nullB_Q95"], color="r", ls="--", lw=1.5, label=f"Null-B Q95={v['J1']['nullB_Q95']:.3f}")
            a.set_xticks(range(len(drugs))); a.set_xticklabels([d[:7] for d in drugs])
            a.set_ylabel("ε_x"); a.set_title(f"(a) 判1：药物 ε_x vs Null-B（池 n={v['J1']['n_pool']}）"); a.legend(fontsize=7)
            # (b) κ*
            b = ax[0, 1]
            for j, dg in enumerate(drugs):
                ys = [cd["kappa"] for r, cd in poolP if r["drug"] == dg and cd.get("kappa") is not None]
                b.scatter([j] * len(ys), ys, s=40, alpha=0.75)
            b.axhline(0.10, color="r", ls="--", lw=1.5); b.axhline(-0.10, color="r", ls="--", lw=1.5)
            b.axhline(0, color="k", lw=0.5)
            b.set_xticks(range(len(drugs))); b.set_xticklabels([d[:7] for d in drugs])
            b.set_ylabel("κ*"); b.set_title(f"(b) 判2：κ* 分布（中位 {v['J2']['abs_kappa_med']:.3f}，判线 |κ*|<0.10）")
            # (c) Δf（INa_L）
            c = ax[1, 0]
            poolL = [(r, cd) for r in recs if "exclude" not in r for cd in r.get("conds", [])
                     if r["group"] == "INa_L" and cd.get("valid") and cd.get("f_step") is not None and cd["f_step"] >= 0.15 and cd.get("df") is not None]
            drugsL = sorted(set(r["drug"] for r, _ in poolL))
            for j, dg in enumerate(drugsL):
                ys = [cd["df"] for r, cd in poolL if r["drug"] == dg]
                c.scatter([j] * len(ys), ys, s=40, alpha=0.75)
            if v["J3"]["null_Q95"]:
                c.axhline(v["J3"]["null_Q95"], color="r", ls="--", lw=1.5); c.axhline(-v["J3"]["null_Q95"], color="r", ls="--", lw=1.5)
            c.axhline(0, color="k", lw=0.5)
            c.set_xticks(range(len(drugsL))); c.set_xticklabels([d[:7] for d in drugsL])
            c.set_ylabel("Δf = f_ramp − f_step"); c.set_title(f"(c) 判3：INa_L 组分差异阻断（池 n={v['J3']['n_pool']}）")
            # (d) 判4 Hill
            d = ax[1, 1]
            xs, ys, cs = [], [], []
            for key, det in v["J4"]["detail"].items():
                if "ours_IC50" in det:
                    xs.append(det["paper"]); ys.append(det["ours_IC50"]); cs.append("g" if det["in_band"] else "r")
            if xs:
                d.scatter(xs, ys, c=cs, s=60)
                lim = [min(xs + ys) / 3, max(xs + ys) * 3]
                d.plot(lim, lim, "k--", lw=1); d.plot(lim, [x * 1.5 for x in lim], "r:", lw=1); d.plot(lim, [x / 1.5 for x in lim], "r:", lw=1)
                d.set_xscale("log"); d.set_yscale("log")
            d.set_xlabel("论文 IC50 µM"); d.set_ylabel("我方 IC50 µM")
            d.set_title(f"(d) 判4：外部对拍 {v['J4']['in_band']}/{v['J4']['total']} 在 ±50% 带")
            fig.tight_layout()
            fig.savefig(OUT_PNG, bbox_inches="tight", dpi=130)
            print(f"[落盘] {OUT_PNG}", flush=True)
        except Exception as e:
            print(f"[图] 失败（不影响判词）: {e}", flush=True)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[落盘] {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
