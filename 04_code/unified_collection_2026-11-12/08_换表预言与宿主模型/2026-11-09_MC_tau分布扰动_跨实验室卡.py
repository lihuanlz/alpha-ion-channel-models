# -*- coding: utf-8 -*-
# 2026-11-09_MC_tau分布扰动_跨实验室卡.py
# ============================================================================
# 蒙特卡洛：Kv11.1 τ 分布扰动下的 CiPA 跨实验室判决存活率
#
# 问题：换表测试封卷判决（CP_full 67/68 过线）用的是 QC 中位数表。
#       中位数是从 ~100 个 QC 细胞的分布里估出来的——如果把"抽样噪声"考虑进去，
#       判决还稳吗？本卡按细胞整体 bootstrap 重抽（保留细胞内相关），
#       每次重抽重建 14 个锚点中位 → 换表 → 全 68 细胞判决，统计过线率分布。
#
# 设计（冻结）：
#   - QC 集：corpus(flag=active) ⋈ profile_all_rows(prof_lo/prof_hi 有限) ∧ r2>=0.9 ∧ 35°C
#     —— 已验证逐点复现 CP_DEACT/CP_ACT 全部 14 个锚点（2026-11-09 复核）
#   - bootstrap 单元 = 细胞（有放回，抽 M 次，M = QC 池细胞数）
#   - 每个 (参数, 电压) 锚点取重抽细胞的中位数；零覆盖锚点回退 CP 中位并登记
#   - 锚点→换表：ms@35°C /1000 ×3.445 → s@21.5°C，TM_ORIG 其余锚点不动（同封卷）
#   - 判线/特征窗/E_rev/Q10/温度桥 全部逐字同封卷卡
#
# 内控（硬断言）：iteration 0 = 恒等表（CP 常数原样），必须逐细胞复现封卷
#   CP_full 判决 67/68（lab3 20/20、lab4 16/16、lab5 31/32、manual 3/3）。
#
# 加速：fast_forward（恒定电压段解析跳步 + 斜坡段短循环，数值差 ~1e-12，
#   由恒等内控验证判决逐细胞一致）；实测侧特征 F*_mea 预计算缓存（与 τ 无关）。
#
# 用法：
#   python 2026-11-09_MC_tau分布扰动_跨实验室卡.py            # 正式跑（断点续跑）
#   SMOKE=1 python ...                                         # 冒烟：5 次串行
#   MC_N=200 NWORKERS=24 BUDGET_S=270                          # 可调
# ============================================================================
import os, sys, json, time, pickle, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIPA_DIR = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/"
            "04_细胞线4/_归档/结果/hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包/data/cipa")
SEALED_JSON = os.path.join(HERE, "2026-11-09_CiPA跨实验室_换表测试_结果.json")
PACK_PKL = os.path.join(HERE, "_mc_pack.pkl")
CKPT = os.path.join(HERE, "2026-11-09_MC_tau扰动_ckpt.jsonl")
OUT_JSON = os.path.join(HERE, "2026-11-09_MC_tau扰动_结果.json")

MC_N = int(os.environ.get("MC_N", "200"))
SMOKE = os.environ.get("SMOKE", "0") == "1"
NWORKERS = int(os.environ.get("NWORKERS", "24"))
BUDGET_S = float(os.environ.get("BUDGET_S", "270"))
if SMOKE:
    MC_N, NWORKERS = 5, 1

DT_NPZ = 2e-4
E_REV_CIPA = -92.1
T_MAIN = 21.5
T_LAB = {"lab3": 35.5, "lab4": 36.1, "lab5": 37.4}
Q10_MAIN = 2.5
RATIO_LO, RATIO_HI = 0.5, 2.0
MIN_MEAS = 0.02
MIN_PLATEAU = 0.05
MIN_SWEEPS = 8
MS = 5

# ------------------------------------------------ 冻结成分（逐字同封卷卡）
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
           -90: 0.072, -80: 0.24,
           -70: 0.8127, -60: 1.20549, -50: 10.14073, -40: 24.0539,
           -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
           20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
_FAC = 2.5 ** 1.35
CP_DEACT = {-70: 37.7, -60: 43.4, -50: 56.7, -40: 80.0, -30: 85.1}          # ms @35°C
CP_ACT = {-20: 280.7, -10: 323.0, 0: 268.0, 10: 150.6, 20: 90.2,
          30: 64.1, 40: 57.5, 50: 71.5, 60: 105.6}                            # ms @35°C
# bootstrap 锚点清单：(参数, 电压) —— 顺序固定，扰动表按此重建
ANCHORS = ([("tau_deact", v) for v in (-70, -60, -50, -40, -30)] +
           [("tau_act", v) for v in (-20, -10, 0, 10, 20, 30, 40, 50, 60)])
CP_MED = {("tau_deact", v): m for v, m in CP_DEACT.items()}
CP_MED.update({("tau_act", v): m for v, m in CP_ACT.items()})


class Tab:
    def __init__(self, anchors):
        self.xs = np.array(sorted(anchors), float)
        self.ly = np.log(np.array([anchors[int(x)] for x in self.xs], float))


def build_tabs(T_lab, tm_anchors, q10=Q10_MAIN, E_rev=E_REV_CIPA):
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
    fac = q10 ** ((T_lab - T_MAIN) / 10.0)
    tm = {k: v / fac for k, v in tm_anchors.items()}
    th = {k: v / fac for k, v in TH_ORIG.items()}
    return dict(m_ss=Tab(m_anchors), tau_m=Tab(tm), h_ss=Tab(h_anchors),
                tau_h=Tab(th), E_rev=E_rev)


def forward_fast(v, tabs, dt):
    """恒定电压段解析跳步；变化段（斜坡/噪声段）逐样本短循环。
    与封卷 forward_dt 数值差 ~1e-12（恒等内控已验证判决逐细胞一致）。"""
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hs = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em_all = np.exp(-dt / tm); eh_all = np.exp(-dt / th)
    n = len(v)
    m = np.empty(n); h = np.empty(n)
    chg = np.flatnonzero(v[1:] != v[:-1]) + 1
    starts = np.concatenate(([0], chg)); ends = np.concatenate((chg, [n]))
    m0 = ms[0]; h0 = hs[0]
    for a, b in zip(starts, ends):
        L = b - a
        if L >= 16:
            kk = np.arange(1, L + 1)
            m[a:b] = ms[a] + (m0 - ms[a]) * em_all[a] ** kk
            h[a:b] = hs[a] + (h0 - hs[a]) * eh_all[a] ** kk
            m0 = m[b - 1]; h0 = h[b - 1]
        else:
            for i in range(a, b):
                m0 = ms[i] + (m0 - ms[i]) * em_all[i]
                h0 = hs[i] + (h0 - hs[i]) * eh_all[i]
                m[i] = m0; h[i] = h0
    return m, h


def sweep_onsets(v):
    return np.where((v[:-1] < 30.0) & (v[1:] >= 30.0))[0] + 1


def extract(v, i_sig, t0):
    def seg(ms0, ms1):
        a, b = t0 + int(ms0 * MS), min(t0 + int(ms1 * MS), len(i_sig))
        return a, b
    a1, b1 = seg(400, 495)
    if b1 <= a1 or np.isnan(i_sig[a1:b1]).any():
        return None
    F1 = float(np.mean(i_sig[a1:b1]))
    ra, rb = seg(500, 640)
    if rb <= ra:
        return None
    vv = v[ra:rb]; ii = i_sig[ra:rb]
    m2 = (vv >= -21.0) & (vv <= -19.0) & np.isfinite(ii)
    m3 = (vv >= -71.0) & (vv <= -69.0) & np.isfinite(ii)
    if m2.sum() < 2 or m3.sum() < 2:
        return None
    F2 = float(np.mean(ii[m2])); F3 = float(np.mean(ii[m3]))
    a4, b4 = seg(800, 850)
    F4 = float(np.mean(i_sig[a4:b4])) if b4 <= len(i_sig) and b4 > a4 and not np.isnan(i_sig[a4:b4]).any() else np.nan
    half = 0.5 * F1
    b5 = min(t0 + int(400 * MS), len(i_sig))
    seg_i = i_sig[t0:b5]
    above = np.where(np.isfinite(seg_i) & (seg_i >= half))[0]
    F5 = float(above[0] / MS) if len(above) else np.nan
    return dict(F1=F1, F2=F2, F3=F3, F4=F4, F5=F5)


def med_ratio(rows, key):
    rs = [r[key + "_r"] for r in rows if r[key + "_r"] is not None and np.isfinite(r[key + "_r"])]
    return float(np.median(rs)) if rs else None


def cell_verdict(rows, n_sweeps):
    if n_sweeps < MIN_SWEEPS:
        return "excluded_few_sweeps", {}
    if len(rows) < MIN_SWEEPS:
        return "excluded_few_valid", {}
    f1m = [r["F1_mea"] for r in rows if np.isfinite(r["F1_mea"])]
    if not f1m or abs(float(np.median(f1m))) < MIN_PLATEAU:
        return "excluded_low_signal", {}
    passes = []
    for key in ("F1", "F2", "F3"):
        r = med_ratio(rows, key)
        if r is None:
            passes.append(False)
        else:
            passes.append(RATIO_LO <= r <= RATIO_HI)
    return ("pass" if all(passes) else "fail"), {}


def lab_of(name):
    if name.startswith("Hh11"):
        return "lab4"
    if name.startswith(("2020_", "207", "208")):
        return "lab5"
    if name[:3].isdigit() and "_" in name:
        return "lab3"
    return "manual"


# ============================================================ Phase 0: 打包
def build_pack():
    import pandas as pd
    print("[pack] 构建细胞包 ...", flush=True)
    t0 = time.time()
    # ---- QC 池（corpus flag=active ⋈ profile CI 有限, r2>=0.9, 35°C）
    prof = pd.read_csv(os.path.join(HERE, "features", "profile_all_rows.csv"), low_memory=False)
    corp = pd.read_csv(os.path.join(HERE, "features", "features_corpus_v5r.csv"), low_memory=False)
    k = corp[(corp.channel == "Kv11.1") & (corp.flag == "active")]
    mg = prof.merge(k[["cell_id", "channel", "temp_nominal", "parameter", "voltage_mV", "flag"]],
                    left_on=["cell_id", "channel", "temp", "parameter", "voltage_mV"],
                    right_on=["cell_id", "channel", "temp_nominal", "parameter", "voltage_mV"],
                    how="inner")
    qc = mg[(mg.temp == 35) & np.isfinite(mg.prof_lo) & np.isfinite(mg.prof_hi) & (mg.r2 >= 0.9)
            & (mg.parameter.isin(("tau_act", "tau_deact")))]
    anchor_map = {}   # (param, v) -> {cell_id: value}
    for (par, vv), grp in qc.groupby(["parameter", "voltage_mV"]):
        key = (par, int(vv))
        if key in CP_MED:
            anchor_map[key] = {int(c): float(x) for c, x in zip(grp.cell_id, grp.value)}
    pool = sorted({c for am in anchor_map.values() for c in am})
    # ---- 内控：重建中位数必须逐点等于 CP 常数
    for key, cmed in CP_MED.items():
        vals = list(anchor_map[key].values())
        med = float(np.median(vals))
        assert abs(med - cmed) < 0.051, f"QC 中位不复现 {key}: {med} vs {cmed}"
    print(f"[pack] QC 池细胞数 M={len(pool)}；14 锚点中位全部复现 CP 常数 ✓", flush=True)

    # ---- CiPA 细胞包（v,  sweeps,  G 窗, 实测特征）
    files = sorted(f for f in os.listdir(CIPA_DIR) if f.endswith(".npz"))
    cells, seen = [], {}
    for f in files:
        d = np.load(os.path.join(CIPA_DIR, f))
        i_arr = d["i"]; iok = i_arr[np.isfinite(i_arr)]
        hh = hashlib.md5(iok.tobytes()).hexdigest() if len(iok) else "empty"
        if hh in seen:
            continue
        seen[hh] = f
        v = d["v"].astype(float); i_meas = d["i"].astype(float)
        t0s = sweep_onsets(v)
        sw = []
        for t0i in t0s:
            fm = extract(v, i_meas, t0i)   # None 的 sweep 仍须进 G 校准（封卷如此）
            a, b = t0i, min(t0i + int(600 * MS), len(v))
            sw.append(dict(t0=int(t0i), a=int(a), b=int(b), ii=i_meas[a:b].copy(), fm=fm))
        cells.append(dict(fn=f, lab=lab_of(f),
                          pool=("manual" if lab_of(f) == "manual" else "main"),
                          T=T_LAB.get(lab_of(f), 35.5), n_sweeps=len(t0s), v=v, sw=sw))
    pack = dict(anchor_map=anchor_map, pool=np.array(pool), cells=cells)
    pickle.dump(pack, open(PACK_PKL, "wb"), protocol=4)
    print(f"[pack] {len(cells)} 细胞入库，{time.time()-t0:.0f}s，"
          f"{os.path.getsize(PACK_PKL)/2**20:.0f} MB", flush=True)
    return pack


# ============================================================ 单次迭代（worker）
G_PACK = None

def _init_worker(pack_path):
    global G_PACK
    with open(pack_path, "rb") as fh:
        G_PACK = pickle.load(fh)


def perturbed_table(rng):
    """bootstrap 重抽 → 14 锚点中位（ms@35）→ TM_CP_full 样式表（s@21.5）。"""
    am = G_PACK["anchor_map"]; pool = G_PACK["pool"]
    tm = dict(TM_ORIG); fallbacks = []
    if rng is None:                      # 恒等：CP 常数原样
        meds = dict(CP_MED)
    else:
        drawn = pool[rng.integers(0, len(pool), len(pool))]
        meds = {}
        for key in ANCHORS:
            amap = am[key]
            vals = [amap[c] for c in drawn if c in amap]
            if vals:
                meds[key] = float(np.median(vals))
            else:
                meds[key] = CP_MED[key]; fallbacks.append(f"{key[0]}@{key[1]}")
    for (par, vv), msd in meds.items():
        tm[vv] = msd / 1000.0 * _FAC
    return tm, meds, fallbacks


def run_iteration(iter_id, seed):
    rng = None if seed is None else np.random.default_rng(seed)
    tm, meds, fallbacks = perturbed_table(rng)
    out = dict(iter=iter_id, seed=(seed if seed is not None else -1),
               meds={f"{p}@{v}": round(m, 3) for (p, v), m in meds.items()},
               fallbacks=fallbacks, cells={})
    for c in G_PACK["cells"]:
        tabs = build_tabs(c["T"], tm)
        m, h = forward_fast(c["v"], tabs, DT_NPZ)
        gden = m * h * (c["v"] - tabs["E_rev"])
        num = den = 0.0
        for s in c["sw"]:
            gg = gden[s["a"]:s["b"]]; ii = s["ii"]
            ok = np.isfinite(ii)
            num += float(np.sum(gg[ok] * ii[ok])); den += float(np.sum(gg[ok] ** 2))
        G = num / den if den > 0 else np.nan
        i_sim = G * gden
        rows = []
        for s in c["sw"]:
            if s["fm"] is None:
                continue
            fs = extract(c["v"], i_sim, s["t0"])
            if fs is None:
                continue
            row = {"F1_mea": s["fm"]["F1"]}
            for key in ("F1", "F2", "F3"):
                mv, sv = s["fm"][key], fs[key]
                row[key + "_r"] = (sv / mv) if (np.isfinite(mv) and abs(mv) >= MIN_MEAS) else None
            rows.append(row)
        verdict, _ = cell_verdict(rows, c["n_sweeps"])
        out["cells"][c["fn"]] = dict(lab=c["lab"], pool=c["pool"], v=verdict)
    return out


def summarize_cells(cells_dict):
    main = {k: v for k, v in cells_dict.items() if v["pool"] == "main"}
    judged = [v["v"] for v in main.values() if v["v"] in ("pass", "fail")]
    npass = sum(1 for x in judged if x == "pass")
    per_lab = {}
    for lab in ("lab3", "lab4", "lab5"):
        lj = [v["v"] for v in main.values() if v["lab"] == lab and v["v"] in ("pass", "fail")]
        per_lab[lab] = (sum(1 for x in lj if x == "pass"), len(lj))
    man = [v["v"] for v in cells_dict.values() if v["pool"] == "manual" and v["v"] in ("pass", "fail")]
    rate = npass / len(judged) if judged else None
    line1 = rate is not None and rate >= 2.0 / 3.0
    line2 = all((p / n >= 0.5) for p, n in per_lab.values() if n > 0)
    verdict = "过线" if (line1 and line2) else ("部分过" if (rate is not None and rate >= 0.5) else "不过")
    return dict(n_pass=npass, n_judged=len(judged), rate=rate, per_lab=per_lab,
                manual=(sum(1 for x in man if x == "pass"), len(man)), 总判=verdict)


# ============================================================ 主流程
def main():
    t_start = time.time()
    pack = pickle.load(open(PACK_PKL, "rb")) if os.path.exists(PACK_PKL) else build_pack()
    global G_PACK
    G_PACK = pack

    # ---- 已完成迭代（断点续跑）
    done = {}
    if os.path.exists(CKPT):
        for line in open(CKPT, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["iter"]] = r
        print(f"[resume] 已完成 {len(done)} 次", flush=True)

    # ---- 恒等内控（iter 0）
    if 0 not in done:
        r0 = run_iteration(0, None)
        s0 = summarize_cells(r0["cells"])
        r0["summary"] = s0
        sealed = json.load(open(SEALED_JSON, encoding="utf-8"))
        sc = sealed["cells"]
        mism = []
        for fn, cv in r0["cells"].items():
            sv = sc[fn]["cfgs"]["CP_full"]["判定"]
            if sv != cv["v"]:
                mism.append((fn, sv, cv["v"]))
        r0["identity_check"] = dict(
            封卷=summarize_cells({fn: dict(lab=sc[fn]["lab"], pool=sc[fn]["pool"],
                                            v=sc[fn]["cfgs"]["CP_full"]["判定"])
                                   for fn in sc}),
            复现=s0, 逐细胞不一致=mism)
        ok = (s0["n_pass"] == 67 and s0["n_judged"] == 68 and not mism)
        print(f"[identity] CP_full 复现 {s0['n_pass']}/{s0['n_judged']}，"
              f"逐细胞不一致 {len(mism)} 个 → {'✓ 内控通过' if ok else '✗ 内控失败'}", flush=True)
        if not ok:
            print("  不一致明细:", mism, flush=True)
            json.dump(r0, open(OUT_JSON + ".identity_fail.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            sys.exit(2)
        with open(CKPT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r0, ensure_ascii=False) + "\n")
        done[0] = r0

    # ---- MC 迭代
    todo = [k for k in range(1, MC_N + 1) if k not in done]
    print(f"[MC] 待跑 {len(todo)} / {MC_N} 次，workers={NWORKERS}", flush=True)
    if todo:
        if NWORKERS <= 1:
            for k in todo:
                r = run_iteration(k, 20261109 * 1000 + k)
                r["summary"] = summarize_cells(r["cells"])
                with open(CKPT, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                done[k] = r
                print(f"  iter {k}: {r['summary']['n_pass']}/{r['summary']['n_judged']} "
                      f"{r['summary']['总判']}", flush=True)
                if time.time() - t_start > BUDGET_S:
                    print("[budget] 时间到，落盘退出", flush=True)
                    return
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=NWORKERS, initializer=_init_worker,
                                     initargs=(PACK_PKL,)) as ex:
                futs = {}
                for k in todo:
                    if time.time() - t_start > BUDGET_S:
                        break
                    futs[ex.submit(run_iteration, k, 20261109 * 1000 + k)] = k
                nfin = 0
                for fut in as_completed(futs):
                    r = fut.result()
                    r["summary"] = summarize_cells(r["cells"])
                    with open(CKPT, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done[r["iter"]] = r
                    nfin += 1
                    if nfin % 20 == 0 or nfin == len(futs):
                        print(f"  ... {nfin}/{len(futs)} 完成 "
                              f"({time.time()-t_start:.0f}s)", flush=True)
                    if time.time() - t_start > BUDGET_S:
                        print("[budget] 时间到，取消未完成任务并落盘", flush=True)
                        ex.shutdown(wait=False, cancel_futures=True)
                        return
    print(f"[done] 共 {len(done)-1} 次 MC + 恒等内控", flush=True)


if __name__ == "__main__":
    main()
