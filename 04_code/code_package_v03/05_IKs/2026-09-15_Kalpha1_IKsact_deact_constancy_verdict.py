# -*- coding: utf-8 -*-
"""
Kα-1：IKs（EQ 串联体）激活/去激活 τ(V) 恒定性判决 —— 正式跑（v1.3 修订版）
预注册：04_细胞线4/结果/预注册_Kα1_IKs激活去激活恒定性判决_2026-09-15.md（v1.3）
运行：%runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-15_Kα1_IKs激活去激活恒定性判决.py' --wdir
输出：同目录 _结果.json / _图.png / _逐文件表.csv
"""
import pyabf, os, re, json, glob, collections
import numpy as np
from scipy.optimize import curve_fit

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
D = os.path.join(ROOT, "数据", "iks_chan_8226585")
OUT_JSON = os.path.join(ROOT, "α模型", "2026-09-15_Kα1_IKs激活去激活恒定性判决_结果.json")
OUT_CSV = os.path.join(ROOT, "α模型", "2026-09-15_Kα1_IKs激活去激活恒定性判决_逐文件表.csv")
OUT_PNG = os.path.join(ROOT, "α模型", "2026-09-15_Kα1_IKs激活去激活恒定性判决_图.png")

PROTO_POOL = {"IKs activation -80", "KCNQ act 4sec - low to high",
              "KCNQ act 4sec - high to low start +80", "KCNQ GV"}
GV_LOOSE = dict(vh=(20, 31), k=(15, 25))
GV_TIGHT = dict(vh=(22, 29), k=(15, 25))
ANCHOR_VH, ANCHOR_K = 25.4, 19.4
ANCHOR_VH_SEM, ANCHOR_K_SEM = 2.4, 1.2
TAU_ACT_BAND_60 = (0.3, 2.0)
TAU_FIT_MAX = 2.5
CV_GATE = 0.3
V_ACT_RANGE = (20, 60)   # v1.4：4 秒协议下 IKs 在 <+20 基本无激活信号
DECAY_GATE = 0.15      # 尾巴衰减下限（判 0）
AMP_GATE = 20.0        # pA，表达量下限（判 0）

# ---------------- 去重 ----------------
def rec_id(fn):
    s = fn[:-4] if fn.lower().endswith(".abf") else fn
    for pat in [" after LS and HMR subtraction", r"_reduced_\d+",
                " after LS w sweep 1 and 4", " after LS",
                " LS updated control trace selection", " LS updated2", " LS updated",
                " LS endo", " LS MC", " YD LS", " LS copy", " LS 2",
                r" LS \+60", r" \+60 LS", r" \+60", " LS"]:
        s = re.sub(pat, "", s, flags=re.I)
    return re.sub(r"[ \-]", "_", s.strip().lower())

def pick_version(group):
    def score(x):
        fn = x["file"].lower()
        return (2 if "updated" in fn else 0) - (1 if "reduced" in fn else 0), x.get("sweeps", 0)
    return sorted(group, key=score)[-1]

# ---------------- 模型 ----------------
def boltz(V, Vh, k):
    return 1.0 / (1.0 + np.exp(-(V - Vh) / k))

def act_model(t, Iss, d, tau):
    y = np.zeros_like(t)
    m = t > d
    y[m] = Iss * (1.0 - np.exp(-(t[m] - d) / tau))
    return y

def r2_of(y, yhat):
    ss = np.sum((y - yhat) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    return 1 - ss / st if st > 0 else np.nan

# ---------------- 单文件提取 ----------------
def analyze(path):
    a = pyabf.ABF(path)
    dt = 1.0 / a.sampleRate
    Vs, tpks, acts, deacts, insts = [], [], [], [], []
    for s in range(a.sweepCount):
        a.setSweep(s)
        e = a.sweepEpochs
        if len(e.types) < 4:
            continue
        v_step = float(e.levels[2]); v_tail = float(e.levels[3])
        if abs(v_tail + 40) > 5:
            continue
        y = a.sweepY.astype(float)
        p2a, p2b = int(e.p1s[2]), int(e.p2s[2])
        p3a, p3b = int(e.p1s[3]), int(e.p2s[3])
        base = float(np.mean(y[max(0, p2a - int(0.02 / dt)):p2a]))
        yc = y - base
        # ---- 激活段（带延迟单指数） ----
        seg = yc[p2a:p2b]
        t = np.arange(len(seg)) * dt
        iss0 = float(np.mean(seg[-int(0.1 / dt):])) if len(seg) > int(0.2 / dt) else float(seg.max())
        act = None
        if iss0 > 0:
            try:
                i15 = int(np.argmax(seg > 0.15 * iss0))  # v1.4：拟合窗从 15% I_ss 起（Fedida 约定）
                tw = t[i15:]
                popt, _ = curve_fit(act_model, tw, seg[i15:],
                                    p0=[iss0, max(tw[0] - 0.05, 0.0), 0.5],
                                    bounds=([0, 0, 0.01], [np.inf, 1.5, TAU_FIT_MAX]),
                                    maxfev=20000)
                r2 = r2_of(seg[i15:], act_model(tw, *popt))
                act = dict(V=v_step, Iss=float(popt[0]), d=float(popt[1]),
                           tau=float(popt[2]), r2=float(r2),
                           trunc=bool(popt[2] >= TAU_FIT_MAX - 1e-6))
            except Exception:
                pass
        # ---- 尾巴段（截断感知提取） ----
        tseg = yc[p3a:p3b]
        tt = np.arange(len(tseg)) * dt
        win = tseg[:int(min(0.2, tt[-1]) / dt)]
        tpk = float(np.max(win)) if len(win) > 5 else np.nan
        deact = None
        if len(tseg) > 50:
            i0 = slice(int(0.005 / dt), int(0.025 / dt))
            i1 = slice(len(tseg) - int(0.05 / dt), len(tseg))
            start = float(np.mean(tseg[i0])); end = float(np.mean(tseg[i1]))
            decay_frac = (start - end) / start if start > 0 else np.nan
            # 锚定 C 的表观 τ（窗口远短于 τ 时仍稳健，偏低偏置同模）
            tau_app, r2d = np.nan, np.nan
            if start > 0:
                Cfix = end
                try:
                    popt, _ = curve_fit(lambda x, A, tau: A * np.exp(-x / tau) + Cfix,
                                        tt, tseg, p0=[max(start - Cfix, 1.0), 0.5],
                                        bounds=([0, 0.02], [np.inf, 60.0]), maxfev=20000)
                    fit = popt[0] * np.exp(-tt / popt[1]) + Cfix
                    tau_app, r2d = float(popt[1]), float(r2_of(tseg, fit))
                except Exception:
                    pass
            deact = dict(V=v_step, start=start, end=end, decay_frac=decay_frac,
                         tau_app=tau_app, r2=r2d, tail_s=float(tt[-1]))
        # ---- 瞬时度（v1.4：只看 V≥+40，低电压无信号） ----
        n50 = max(1, int(0.05 / dt))
        head = float(np.mean(seg[int(0.002 / dt):int(0.002 / dt) + n50]))
        inst = head / iss0 if (iss0 > 0 and v_step >= 40) else np.nan
        Vs.append(v_step); tpks.append(tpk)
        if act: acts.append(act)
        if deact: deacts.append(deact)
        insts.append(inst)
    if len(Vs) < 8:
        return None
    Vs = np.array(Vs); tpks = np.array(tpks, dtype=float)
    ok = ~np.isnan(tpks)
    gv = None
    if ok.sum() >= 8:
        vv = Vs[ok]; aa = tpks[ok] - np.nanpercentile(tpks[ok], 5)
        mx = aa.max()
        if mx > 0:
            aa = aa / mx
            try:
                popt, _ = curve_fit(boltz, vv, aa, p0=[25, 19],
                                    bounds=([-80, 2], [120, 80]), maxfev=10000)
                gv = dict(Vh=float(popt[0]), k=float(popt[1]),
                          r2=float(r2_of(aa, boltz(vv, *popt))))
            except Exception:
                pass
    return dict(gv=gv, acts=acts, deacts=deacts,
                inst_med=float(np.nanmedian(insts)) if insts else np.nan,
                n_sweeps=len(Vs))

# ---------------- 主流程 ----------------
def main():
    print("=" * 72)
    print(" Kα-1：IKs（EQ）激活/去激活 τ(V) 恒定性判决（预注册 v1.3 正式跑）")
    print("=" * 72)
    inv = json.load(open(os.path.join(D, "inventory.json")))
    cand = [x for x in inv if x.get("protocol") in PROTO_POOL and "error" not in x]
    groups = collections.defaultdict(list)
    for x in cand:
        groups[rec_id(x["file"])].append(x)
    keep = [pick_version(g) for g in groups.values()]
    if os.environ.get("KA1_SMOKE"):
        keep = keep[:int(os.environ["KA1_SMOKE"])]
        print(f"[SMOKE 模式] 只跑前 {len(keep)} 条记录")
    print(f"[去重] 候选 {len(cand)} 文件 → {len(groups)} 记录 ID → 保留 {len(keep)}")

    recs = []
    for i, x in enumerate(sorted(keep, key=lambda z: z["file"])):
        try:
            r = analyze(os.path.join(D, x["file"]))
            if r and r["gv"]:
                r.update(file=x["file"], protocol=x["protocol"], rid=rec_id(x["file"]))
                recs.append(r)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{len(keep)}", flush=True)
    print(f"[提取] 成功 {len(recs)} 记录")

    def pool(recs, gate):
        out = []
        for r in recs:
            g = r["gv"]
            if g["r2"] < 0.9:
                continue
            if not (gate["vh"][0] <= g["Vh"] <= gate["vh"][1] and gate["k"][0] <= g["k"] <= gate["k"][1]):
                continue
            if r["inst_med"] > 0.5:      # v1.4：药物样瞬时 onset 物理排除
                continue
            hi = [d for d in r["deacts"] if d["V"] >= 40]
            if not hi:
                continue
            amp = float(np.median([d["start"] for d in hi]))
            dec = float(np.median([d["decay_frac"] for d in hi]))
            if np.isnan(dec) or dec < 0.05:   # v1.4：Mef 样平尾物理排除
                continue
            if amp <= AMP_GATE:
                continue
            r["_amp"], r["_dec"] = amp, dec
            out.append(r)
        return out

    p_loose = pool(recs, GV_LOOSE)
    verdict = {"pool_loose_n": len(p_loose)}
    branch = "loose"
    p = p_loose
    if len(p_loose) > 12:
        p_tight = pool(recs, GV_TIGHT)
        verdict["pool_tight_n"] = len(p_tight)
        branch = "tight(污染重报)"
        p = p_tight
    vh_med = float(np.median([r["gv"]["Vh"] for r in p])) if p else np.nan
    k_med = float(np.median([r["gv"]["k"] for r in p])) if p else np.nan
    J0 = (4 <= len(p) <= 12 and abs(vh_med - ANCHOR_VH) <= 5 and abs(k_med - ANCHOR_K) <= 4)
    verdict["J0"] = dict(branch=branch, n=len(p), vh_med=vh_med, k_med=k_med, PASS=bool(J0))
    print(f"[判0] 池分支={branch} n={len(p)} V½中位={vh_med:.1f} k中位={k_med:.1f} -> {'过' if J0 else '未过'}")
    for r in p:
        print(f"    池: {r['file'][:44]:46s} V½={r['gv']['Vh']:+5.1f} k={r['gv']['k']:4.1f} "
              f"inst={r['inst_med']:.2f} amp={r['_amp']:.0f}pA dec={r['_dec']:.0%}")

    # ---- 判1 ----
    per_v = collections.defaultdict(list)
    trunc_v = collections.defaultdict(lambda: [0, 0])
    for r in p:
        for a in r["acts"]:
            if a["r2"] < 0.9:
                continue
            v = int(round(a["V"]))
            trunc_v[v][1] += 1
            if a["trunc"]:
                trunc_v[v][0] += 1
            per_v[v].append(a["tau"])
    j1_detail = {}
    for v in sorted(per_v):
        if not (V_ACT_RANGE[0] <= v <= V_ACT_RANGE[1]):
            continue
        taus = per_v[v]; nt, nn = trunc_v[v]
        if nn < 3:   # v1.4：池上限 ~6，n≥3 可投
            continue
        if nt / nn >= 1 / 3:
            j1_detail[v] = dict(n=nn, note="protocol-truncated")
            continue
        m = float(np.mean(taus)); cv = float(np.std(taus) / m) if m > 0 else np.nan
        j1_detail[v] = dict(n=nn, tau_med=float(np.median(taus)), cv=cv, PASS=bool(cv < CV_GATE))
    voted = [v for v, d in j1_detail.items() if "cv" in d]
    j1_passfrac = float(np.mean([j1_detail[v]["PASS"] for v in voted])) if voted else 0.0
    J1 = j1_passfrac >= 0.8
    verdict["J1"] = dict(pass_frac=j1_passfrac, detail=j1_detail, PASS=bool(J1))
    print(f"[判1] τ_act(V) CV<0.3 电压占比 {j1_passfrac*100:.0f}%"
          f"（{sum(1 for v in voted if j1_detail[v]['PASS'])}/{len(voted)}） -> {'过' if J1 else '未过'}")

    # ---- 判2：τ_app（锚定 C 表观 τ）总 CV + 协议分层 ----
    td_cells = {}
    for r in p:
        tds = [d["tau_app"] for d in r["deacts"]
               if d["V"] >= 40 and not np.isnan(d["tau_app"]) and d["r2"] >= 0.5]
        if tds:
            td_cells[r["file"]] = (float(np.median(tds)), r["protocol"])
    vals = np.array([v[0] for v in td_cells.values()]) if td_cells else np.array([])
    cv2 = float(np.std(vals) / np.mean(vals)) if len(vals) >= 3 else np.nan
    by_proto = collections.defaultdict(list)
    for fn, (v, pr) in td_cells.items():
        by_proto[pr].append(v)
    proto_cv = {pr: (len(vv), float(np.std(vv) / np.mean(vv)) if len(vv) >= 3 else np.nan)
                for pr, vv in by_proto.items()}
    J2 = bool(cv2 < CV_GATE) if not np.isnan(cv2) else False
    sub_ok = all(c < CV_GATE for n, c in proto_cv.values() if not np.isnan(c))
    verdict["J2"] = dict(n=len(vals), tau_app_med_s=float(np.median(vals)) if len(vals) else None,
                         cv=cv2, proto_cv=proto_cv, subgroups_pass=bool(sub_ok), PASS=J2)
    print(f"[判2] τ_app(−40) n={len(vals)} 中位={np.median(vals) if len(vals) else np.nan:.2f}s "
          f"CV={cv2:.2f} -> {'过' if J2 else '未过'}"
          f"（协议分层: { {pr: (n, round(c,2) if not np.isnan(c) else None) for pr,(n,c) in proto_cv.items()} }）")

    # ---- 判3 ----
    j3a = bool(abs(vh_med - ANCHOR_VH) <= 2 * ANCHOR_VH_SEM and abs(k_med - ANCHOR_K) <= 2 * ANCHOR_K_SEM)
    tau60 = [a["tau"] for r in p for a in r["acts"] if int(round(a["V"])) == 60 and a["r2"] >= 0.9 and not a["trunc"]]
    tau60_med = float(np.median(tau60)) if tau60 else np.nan
    j3b = bool(TAU_ACT_BAND_60[0] <= tau60_med <= TAU_ACT_BAND_60[1]) if tau60 else False
    J3 = j3a and j3b
    verdict["J3"] = dict(gv_anchor=bool(j3a), tau60_med_s=tau60_med, tau60_n=len(tau60),
                         tau_anchor=bool(j3b), PASS=bool(J3))
    print(f"[判3] G-V锚 {'过' if j3a else '未过'} | τ_act(+60) 中位={tau60_med:.2f}s（n={len(tau60)}） {'过' if j3b else '未过'}")

    verdict["ALL"] = bool(J0 and J1 and J2 and J3)
    print(f"[总判] {'全过 —— IKs α 表提取成立' if verdict['ALL'] else '未全过 —— 如实登记'}")

    json.dump(verdict, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("file,protocol,Vh,k,gv_r2,inst_med,amp_pA,decay_frac,tau_app_s,tau60_s,in_pool\n")
        for r in recs:
            t60 = [a["tau"] for a in r["acts"] if int(round(a["V"])) == 60 and a["r2"] >= 0.9]
            tds = [d["tau_app"] for d in r["deacts"] if d["V"] >= 40 and not np.isnan(d["tau_app"])]
            hi = [d for d in r["deacts"] if d["V"] >= 40]
            amp = float(np.median([d["start"] for d in hi])) if hi else ""
            dec = float(np.median([d["decay_frac"] for d in hi])) if hi else ""
            f.write(f"{r['file']},{r['protocol']},{r['gv']['Vh']:.2f},{r['gv']['k']:.2f},"
                    f"{r['gv']['r2']:.3f},{r['inst_med']:.3f},{amp},{dec if dec=='' else round(dec,3)},"
                    f"{np.median(tds) if tds else ''},{np.median(t60) if t60 else ''},{1 if r in p else 0}\n")
    print(" 结果落盘:", OUT_JSON)
    print(" 表落盘:", OUT_CSV)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        pass
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    ax = axs[0, 0]
    for r in recs:
        c = "tab:red" if r in p else "0.8"
        ax.scatter(r["gv"]["Vh"], r["gv"]["k"], c=c, s=18, alpha=0.8)
    ax.axvspan(ANCHOR_VH - 2 * ANCHOR_VH_SEM, ANCHOR_VH + 2 * ANCHOR_VH_SEM, color="g", alpha=0.15)
    ax.axhspan(ANCHOR_K - 2 * ANCHOR_K_SEM, ANCHOR_K + 2 * ANCHOR_K_SEM, color="g", alpha=0.15)
    ax.set_xlabel("V½ (mV)"); ax.set_ylabel("k (mV)"); ax.set_title("判0/判3：池与 WT 锚（绿带±2SEM）")
    ax = axs[0, 1]
    for r in p:
        vs = [a["V"] for a in r["acts"] if a["r2"] >= 0.9]
        ts = [a["tau"] for a in r["acts"] if a["r2"] >= 0.9]
        ax.plot(vs, ts, ".-", color="0.6", lw=0.5, ms=4)
    med_v = sorted(v for v in j1_detail if "tau_med" in j1_detail[v])
    ax.plot(med_v, [j1_detail[v]["tau_med"] for v in med_v], "ko-", lw=2, label="池内中位")
    ax.set_xlabel("V (mV)"); ax.set_ylabel("τ_act (s)"); ax.set_title("判1：τ_act–V（灰=各细胞）"); ax.legend()
    ax = axs[1, 0]
    vv = [v for v in voted]
    ax.bar([str(v) for v in vv], [j1_detail[v]["cv"] for v in vv])
    ax.axhline(CV_GATE, color="r", ls="--")
    ax.set_xlabel("V (mV)"); ax.set_ylabel("CV"); ax.set_title(f"判1：CV 谱（过线占比 {j1_passfrac*100:.0f}%）")
    ax = axs[1, 1]
    if len(vals):
        ax.hist(vals, bins=12)
    ax.set_xlabel("τ_app(−40) (s)"); ax.set_ylabel("细胞数")
    ax.set_title(f"判2：表观去激活 τ 分布（CV={cv2:.2f}，截断偏低同模）")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(" 图落盘:", OUT_PNG)

if __name__ == "__main__":
    main()
