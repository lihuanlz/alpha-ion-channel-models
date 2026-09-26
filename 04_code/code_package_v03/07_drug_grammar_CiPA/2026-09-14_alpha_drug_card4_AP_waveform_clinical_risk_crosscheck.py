# 2026-09-14_α模型_药物卡4_AP波形临床风险对拍.py
# ============================================================================
# 药物卡4 · AP波形下药物阻断的临床风险对拍（预注册判决卡）
# 预注册：结果\预注册_α模型_药物卡4_AP波形临床风险对拍_2026-09-14.md（判线跑前钉死）
#
# 模型：I = G·m·h·(1−D)·(V−E_rev)；引擎四表逐字不动（import 正式引擎）
# 药物模块（捕获型）：dD/dt = k_on·[D]·m·(1−D) − k_off·D
# 参数：k_obs([D]) 汇总线性拟合斜率 = k_on（卡3封卷结果，k_obs≥0.9饱和排除）；
#   k_off := k_on·IC50（一致性锚定：持续去极化稳态回到实测 Hill 平衡 D_ss(IC50)=0.5；
#   拟合截距只作诊断登记，不进模型——Milnes 主要观测结合起始段，截距不可靠）。
#   斜率≤0 或有效浓度档<2 → 静态退化 D=Hill(IC50,nH) 钳位，旗帜登记
# 锚：newCiPA.csv（FDA/CiPA 官方；therapeutic=游离Cmax nM；CiPA 2高/1中/0低）
# 判线：V1 Spearman ρ(S_eq@1×, 风险序)≥0.60 且单侧p<0.01；V2 AUC(高vs低)≥0.80；
#   V3/稳态/敏感性只登记。合成自检三条件（单调/捕获极限/Hill一致）跑前。
# v2（冒烟后正式跑前修订，预注册§八在案）：
#   (1) k_off 改一致性锚定 k_off=k_on·IC50（拟合截距只登记）；
#   (2) 主判指标换 S_eq（占空比调整解析平衡：S_eq=[D]·⟨m⟩/([D]·⟨m⟩+IC50)，
#       ⟨m⟩=对照 AP 末循环 m 时间平均）——有限列车 S@12循环 改为暂态登记，
#       因为慢解离药在 105.9s 内远未达平衡（临床为慢性给药=平衡态）。
# 运行：Spyder %runfile '...py' --wdir 。SMOKE=1 冒烟 3 药（dof/ond/ver），协议逐字相同。
# ============================================================================
import os
import json
import csv
import importlib.util
import numpy as np
import scipy.io as sio
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1e-4
E_REV = -88.33
SMOKE = os.environ.get("SMOKE", "0") == "1"

F_CARD3 = os.path.join(HERE, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")
F_CIPA = os.path.join(HERE, "CiPA官方锚", "newCiPA.csv")
F_PROTO = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/"
           "wA1/代码A1_16704007_流程固定重标卡_2026-09-11/data/protocols/ap_protocol.mat")

N_LOOP = 12                      # 8.8245s × 12 ≈ 105.9s ≈ 84拍
CONC_MULT = [1, 3, 10, 25]       # × 游离Cmax
SS_CHECK_LOOP = 6                # 稳态诊断：第6 vs 第12循环
SAT_KOBS = 0.9                   # k_obs 饱和排除阈

DRUGS_24 = ["azimilide", "bepridil", "disopyramide", "dofetilide", "ibutilide", "sotalol",
            "vandetanib",
            "astemizole", "chlorpromazine", "cisapride", "clarithromycin", "clozapine",
            "domperidone", "droperidol", "ondansetron", "pimozide", "risperidone",
            "terfenadine",
            "diltiazem", "metoprolol", "mexiletine", "ranolazine", "tamoxifen", "verapamil"]
SMOKE_DRUGS = ["dofetilide", "ondansetron", "verapamil"]


# ---------- 引擎导入（文件名以数字开头，用 importlib） ----------
def load_engine():
    fp = os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py")
    spec = importlib.util.spec_from_file_location("alpha_engine", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_anchor_tabs(eng):
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    return eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)


# ---------- 药物模块前向（引擎 forward 的逐字扩展：加 D 态） ----------
def forward_drug(V, tabs, kon_nM, koff, conc_nM, hill_clamp=None):
    """kon_nM: 1/(nM·s); koff: 1/s; hill_clamp: 静态退化时的常数 D。
    返回每循环 Q 列表（I = G·m·h·(1−D)·(V−E_rev)，∫dt 按循环切段）。"""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    G = tabs["G"] if tabs["G"] else 1.0
    n = len(V)
    nlp = n // N_LOOP
    m0, h0, D = ms[0], hss[0], 0.0
    Q = np.zeros(N_LOOP)
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hss[i] + (h0 - hss[i]) * eh[i]
        if hill_clamp is not None:
            D = hill_clamp
        else:
            D += DT * (kon_nM * conc_nM * m0 * (1.0 - D) - koff * D)
            if D < 0.0:
                D = 0.0
        Q[i // nlp] += G * m0 * h0 * (1.0 - D) * (v[i] - E_REV)
    return Q * DT


# ---------- 卡3 → 每药参数 ----------
def drug_params(card3, drug):
    ic50s, nhs, ic50_lab, nh_lab = [], [], [], []
    pts = []  # (conc_nM, k_obs)
    n_sat = 0
    for key, ds in card3["datasets"].items():
        if key.split("|")[1] != drug:
            continue
        L1 = ds.get("L1") or {}
        if L1.get("ic50_self"):
            ic50s.append(L1["ic50_self"])
            nhs.append(L1["nh_self"])
        if L1.get("ic50_lab"):
            ic50_lab.append(L1["ic50_lab"])
            nh_lab.append(L1["nh_lab"])
        for u in ds.get("units") or []:
            k = u.get("k_obs")
            if k is None:
                continue
            if k >= SAT_KOBS:
                n_sat += 1
                continue
            pts.append((u["conc_nM"], k))
    ic50 = float(np.median(ic50s)) if ic50s else None
    nh = float(np.median(nhs)) if nhs else None
    kon = koff = None
    koff_fit = None
    degen = None
    if pts:
        tiers = sorted({c for c, _ in pts})
        if len(tiers) >= 2:
            x = np.array([c for c, _ in pts])
            y = np.array([k for _, k in pts])
            sl, ic, r, p, se = stats.linregress(x, y)
            koff_fit = float(ic)
            if sl > 0 and ic50:
                kon = float(sl)
                koff = float(sl * ic50)      # 一致性锚定：D_ss(IC50)=0.5
            elif sl > 0:
                degen = "静态退化(无IC50)"
            else:
                degen = "静态退化(斜率≤0)"
        else:
            degen = "静态退化(浓度档<2)"
    else:
        degen = "静态退化(无k_obs)"
    return dict(ic50=ic50, nh=nh, ic50_lab=float(np.median(ic50_lab)) if ic50_lab else None,
                nh_lab=float(np.median(nh_lab)) if nh_lab else None,
                kon=kon, koff=koff, koff_fit=koff_fit, degen=degen, n_sat=n_sat, n_pts=len(pts))


def hill_block(ic50_nM, nh, conc_nM):
    if ic50_nM is None or nh is None or ic50_nM <= 0:
        return None
    return float(conc_nM ** nh / (ic50_nM ** nh + conc_nM ** nh))


# ---------- 合成自检（预注册钉死三条件） ----------
def synth_check(V1, tabs):
    print("\n[合成自检]", flush=True)
    kon, koff, ic50 = 1e-4, 1e-3, 10.0     # 自洽: IC50 = koff/kon
    ok = True
    # (a) AP 波形单调性：S@25× > S@1×（Cmax=10nM 虚拟）
    q1 = forward_drug(V1, tabs, kon, koff, 10.0)[-1]
    q25 = forward_drug(V1, tabs, kon, koff, 250.0)[-1]
    q0 = forward_drug(V1, tabs, 0.0, 0.0, 0.0)[-1]
    s1, s25 = 1 - q1 / q0, 1 - q25 / q0
    ca = s25 > s1 > 0
    print(f"  (a) 单调: S@1×={s1:.4f} S@25×={s25:.4f} -> {'过' if ca else '不过'}", flush=True)
    ok &= ca
    # (b) k_off=0 捕获极限：D 只增不减（直接积分数值检查）
    v = V1.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    em = np.exp(-DT / tm)
    m0, D, dmin = ms[0], 0.0, 0.0
    for i in range(len(v)):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        dD = DT * (kon * 10.0 * m0 * (1.0 - D))
        D += dD
        dmin = min(dmin, dD)
    cb = dmin >= -1e-15
    print(f"  (b) 捕获极限(k_off=0): min(dD)={dmin:.2e} -> {'过' if cb else '不过'}", flush=True)
    ok &= cb
    # (c) 持续 +20mV（m→1）@ [D]=IC50：D_ss→0.5
    mss1 = float(np.exp(np.interp(20.0, tabs["m_ss"].xs, tabs["m_ss"].ly)))
    Dss = kon * ic50 * mss1 / (kon * ic50 * mss1 + koff)
    cc = abs(Dss - 0.5) < 0.02
    print(f"  (c) Hill一致: m_ss(+20)={mss1:.4f} D_ss={Dss:.4f}（应≈0.5）-> {'过' if cc else '不过'}",
          flush=True)
    ok &= cc
    print(f"  合成自检 {'过' if ok else '不过——全卡降级登记'}", flush=True)
    return ok


def auc_high_low(s_by_drug, risk_by_drug):
    hi = [s_by_drug[d] for d in s_by_drug if risk_by_drug[d] == 2]
    lo = [s_by_drug[d] for d in s_by_drug if risk_by_drug[d] == 0]
    if not hi or not lo:
        return None
    u, _ = stats.mannwhitneyu(hi, lo, alternative="greater")
    return float(u / (len(hi) * len(lo)))


def main():
    print("=" * 76, flush=True)
    print(" 药物卡4 · AP波形临床风险对拍（预注册判决卡）", flush=True)
    print("=" * 76, flush=True)

    eng = load_engine()
    tabs = build_anchor_tabs(eng)
    print(f"[锚] 细胞 16713003 G={tabs['G']:.4f} gflag={tabs['gflag']}", flush=True)

    V1 = sio.loadmat(F_PROTO)["T"].flatten().astype(float)
    n1 = len(V1)
    V = np.tile(V1, N_LOOP)
    print(f"[协议] ap_protocol 单循环 {n1 * DT:.4f}s × {N_LOOP} = {n1 * N_LOOP * DT:.1f}s", flush=True)

    synth_ok = synth_check(V, tabs)

    card3 = json.load(open(F_CARD3, encoding="utf-8"))
    cmax, risk = {}, {}
    with open(F_CIPA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["drug"].strip()
            if d in DRUGS_24:
                cmax[d] = float(row["therapeutic"])     # nM
                risk[d] = int(row["CiPA"])
    drugs = SMOKE_DRUGS if SMOKE else DRUGS_24
    print(f"[药物] {len(drugs)} 种（{'冒烟' if SMOKE else '全量'}）", flush=True)

    # 对照（无药）只做一次；⟨m⟩ 取对照末循环时间平均（占空比，药无关）
    Q0 = forward_drug(V, tabs, 0.0, 0.0, 0.0)
    q0_last, q0_mid = Q0[-1], Q0[SS_CHECK_LOOP - 1]
    m_arr, _h_arr = eng.forward(V, tabs)
    m_bar = float(m_arr[-len(m_arr) // N_LOOP:].mean())
    print(f"[占空比] AP 末循环 ⟨m⟩ = {m_bar:.4f}", flush=True)

    res = {}
    for i, d in enumerate(drugs):
        p = drug_params(card3, d)
        entry = dict(risk=risk[d], cmax_nM=cmax[d], **p)
        S, S_mid, S_static, S_eq, S_eq_lab = {}, {}, {}, {}, {}
        for mult in CONC_MULT:
            conc = cmax[d] * mult
            b = hill_block(p["ic50"], p["nh"], conc)
            S_static[str(mult)] = b
            if p["ic50"]:
                S_eq[str(mult)] = float(conc * m_bar / (conc * m_bar + p["ic50"]))
            if p["ic50_lab"]:
                S_eq_lab[str(mult)] = float(conc * m_bar / (conc * m_bar + p["ic50_lab"]))
            if p["degen"]:
                if b is None:
                    S[str(mult)] = None
                    continue
                Qd = forward_drug(V, tabs, 0.0, 0.0, 0.0, hill_clamp=b)
            else:
                Qd = forward_drug(V, tabs, p["kon"], p["koff"], conc)
            S[str(mult)] = float(1.0 - Qd[-1] / q0_last)
            S_mid[str(mult)] = float(1.0 - Qd[SS_CHECK_LOOP - 1] / q0_mid)
        entry["S_transient"] = S
        entry["S_transient_loop%d" % SS_CHECK_LOOP] = S_mid
        entry["S_eq"] = S_eq
        entry["S_eq_锚版"] = S_eq_lab
        entry["S_static_Hill"] = S_static
        s1, s6 = S.get("1"), S_mid.get("1")
        entry["稳态旗"] = (s1 is not None and s6 is not None and abs(s1 - s6) > 0.05)
        res[d] = entry
        tag = p["degen"] or f"kon={p['kon']:.2e} koff={p['koff']:.2e}"
        print(f"  [{i + 1}/{len(drugs)}] {d} (风险{risk[d]} Cmax={cmax[d]}nM): {tag}"
              f"  S_eq@1×={S_eq.get('1')}  暂态S@1×={S.get('1')}", flush=True)

    # ---------- 判线（主判指标 = S_eq@1×，v2 钉死） ----------
    s1map = {d: r["S_eq"]["1"] for d, r in res.items() if r["S_eq"].get("1") is not None}
    rho, pval = (np.nan, np.nan)
    if len(s1map) >= 8:
        rho, pval = stats.spearmanr([s1map[d] for d in s1map],
                                    [risk[d] for d in s1map])
        pval = pval / 2.0 if rho > 0 else 1.0 - pval / 2.0   # 单侧
    auc = auc_high_low(s1map, risk)
    # 敏感性：锚 IC50 版 S_eq
    s1lab = {d: r["S_eq_锚版"]["1"] for d, r in res.items() if r["S_eq_锚版"].get("1") is not None}
    rho_lab, p_lab = (np.nan, np.nan)
    if len(s1lab) >= 8:
        rho_lab, p_lab = stats.spearmanr([s1lab[d] for d in s1lab], [risk[d] for d in s1lab])
        p_lab = p_lab / 2.0 if rho_lab > 0 else 1.0 - p_lab / 2.0
    auc_lab = auc_high_low(s1lab, risk) if s1lab else None
    # 敏感性：静态 Hill 版
    s1st = {d: r["S_static_Hill"]["1"] for d, r in res.items()
            if r["S_static_Hill"].get("1") is not None}
    rho_st, p_st = (np.nan, np.nan)
    if len(s1st) >= 8:
        rho_st, p_st = stats.spearmanr([s1st[d] for d in s1st], [risk[d] for d in s1st])
        p_st = p_st / 2.0 if rho_st > 0 else 1.0 - p_st / 2.0
    auc_st = auc_high_low(s1st, risk) if s1st else None
    # 暂态登记
    s1tr = {d: r["S_transient"]["1"] for d, r in res.items()
            if r["S_transient"].get("1") is not None}
    rho_tr, p_tr = (np.nan, np.nan)
    if len(s1tr) >= 8:
        rho_tr, p_tr = stats.spearmanr([s1tr[d] for d in s1tr], [risk[d] for d in s1tr])
        p_tr = p_tr / 2.0 if rho_tr > 0 else 1.0 - p_tr / 2.0
    v3 = [d for d, r in res.items() if (r["S_eq"].get("25") or 0) > 0.5]

    verdict = {
        "合成自检": bool(synth_ok),
        "主判指标": "S_eq@1×Cmax（占空比调整平衡，v2）",
        "m_bar_AP末循环": m_bar,
        "V1_Spearman": {"rho": float(rho), "p_one_side": float(pval),
                        "过": bool(synth_ok and rho >= 0.60 and pval < 0.01)},
        "V2_AUC高低": {"auc": auc, "过": bool(synth_ok and auc is not None and auc >= 0.80)},
        "V3_S_eq25x过半药数": len(v3),
        "敏感性_锚IC50版": {"rho": float(rho_lab), "p_one_side": float(p_lab), "auc": auc_lab},
        "敏感性_静态Hill版": {"rho": float(rho_st), "p_one_side": float(p_st), "auc": auc_st},
        "登记_暂态84拍版": {"rho": float(rho_tr), "p_one_side": float(p_tr)},
        "稳态旗药数": sum(1 for r in res.values() if r["稳态旗"]),
        "静态退化药": [d for d, r in res.items() if r["degen"]],
    }

    print("\n" + "=" * 76, flush=True)
    print(f" V1(主判 S_eq@1×): ρ={rho:.3f} p={pval:.4f} -> "
          f"{'过' if verdict['V1_Spearman']['过'] else '不过'}", flush=True)
    print(f" V2: AUC(高vs低)={auc} -> {'过' if verdict['V2_AUC高低']['过'] else '不过'}", flush=True)
    print(f" V3登记: S_eq@25×>0.5 的药 {len(v3)}/{len(res)}；稳态旗 {verdict['稳态旗药数']}；"
          f"静态退化 {len(verdict['静态退化药'])}", flush=True)
    print(f" 敏感性(锚IC50): ρ={rho_lab:.3f} p={p_lab:.4f} AUC={auc_lab}", flush=True)
    print(f" 敏感性(静态Hill): ρ={rho_st:.3f} p={p_st:.4f} AUC={auc_st}", flush=True)
    print(f" 登记(暂态84拍): ρ={rho_tr:.3f} p={p_tr:.4f}", flush=True)

    sfx = "_冒烟" if SMOKE else ""
    fj = os.path.join(HERE, f"2026-09-14_α模型_药物卡4_AP风险对拍{sfx}_结果.json")
    json.dump({"预注册": "预注册_α模型_药物卡4_AP波形临床风险对拍_2026-09-14.md",
               "判词": verdict, "逐药": res}, open(fj, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n  结果落盘: {fj}", flush=True)

    # 图
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
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = {2: "#c0392b", 1: "#2980b9", 0: "#27ae60"}
    labels = {2: "高", 1: "中", 0: "低"}
    for rk in (2, 1, 0):
        ds = [d for d in s1map if risk[d] == rk]
        ys = [s1map[d] for d in ds]
        ax.scatter([rk] * len(ds), ys, c=colors[rk], s=60, alpha=0.85, label=f"{labels[rk]}风险")
        for d, y in zip(ds, ys):
            ax.annotate(d, (rk, y), fontsize=6, alpha=0.7,
                        xytext=(3, 2), textcoords="offset points")
    ax.set_xticks([2, 1, 0])
    ax.set_xticklabels(["高风险", "中风险", "低风险"])
    ax.set_ylabel("S_eq@1×Cmax（平衡态 hERG 抑制率）")
    ax.set_title(f"卡4 主判：ρ={rho:.2f} p={pval:.4f}  AUC={auc if auc else float('nan'):.2f}")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for d in drugs:
        r = res[d]
        xs, ys = [], []
        for mult in CONC_MULT:
            v_ = r["S_eq"].get(str(mult))
            if v_ is not None:
                xs.append(mult)
                ys.append(v_)
        if xs:
            ax.plot(xs, ys, "o-", color=colors[risk[d]], alpha=0.6, lw=1, ms=3)
    ax.set_xscale("log")
    ax.set_xlabel("浓度 × 游离Cmax")
    ax.set_ylabel("S_eq")
    ax.set_title("剂量-抑制曲线（红=高 蓝=中 绿=低）")
    ax.grid(alpha=0.3, which="both")
    fpng = os.path.join(HERE, f"2026-09-14_α模型_药物卡4_AP风险对拍{sfx}.png")
    fig.tight_layout()
    fig.savefig(fpng, dpi=140, bbox_inches="tight")
    print(f"  图落盘: {fpng}", flush=True)


if __name__ == "__main__":
    main()
