# 2026-11-09_药物卡4_AP风险对拍_换表.py
# ============================================================================
# 药物卡4 换表版：AP 波形下 24 药临床风险对拍（τ_m 三配置：orig / full / act）
#
# 与原卡逐字同构：I = G·m·h·(1-D)·(V-E_rev)；药物模块 trapping 型
#   dD/dt = k_on·[D]·m·(1-D) - k_off·D；k_off=k_on·IC50 一致性锚定（v2 修订在案）
# 主判指标 S_eq@1×Cmax（占空比调整平衡）对 m_bar 公共因子秩不变 —— 理论预告：
#   V1/V2 对换表不变；可变量是暂态登记臂（84拍）与 m_bar 绝对值。如实记录。
#
# 判线（冻结）：V1 Spearman rho(S_eq@1x, risk)>=0.60 且单侧 p<0.01；V2 AUC(high vs low)>=0.80
# 数据：卡3封卷 JSON + CiPA官方锚 newCiPA.csv + 归档 ap_protocol.mat + 归档引擎（细胞16713003）
# 断点：逐配置落盘 _ckpt_{cfg}.json，重跑自动跳过已完成配置。
# ============================================================================
import os, sys, json, csv
import numpy as np
import scipy.io as sio
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cp_swap_lib_2026_11_09 import load_engine, anchor_tabs, swap_tau_m

PKG07 = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
         "alpha-ion-channel-models/04_code/code_package_v03/07_drug_grammar_CiPA")
F_CARD3 = os.path.join(PKG07, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")
F_CIPA = os.path.join(PKG07, "CiPA官方锚", "newCiPA.csv")
ARCH_PACK = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/"
             "04_细胞线4/_归档/结果/hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包")
F_PROTO = os.path.join(ARCH_PACK, "data", "protocols", "ap_protocol.mat")

DT = 1e-4
E_REV = -88.33
N_LOOP = 12
CONC_MULT = [1, 3, 10, 25]
SS_CHECK_LOOP = 6
SAT_KOBS = 0.9
SMOKE = os.environ.get("SMOKE", "0") == "1"

DRUGS_24 = ["azimilide", "bepridil", "disopyramide", "dofetilide", "ibutilide", "sotalol",
            "vandetanib",
            "astemizole", "chlorpromazine", "cisapride", "clarithromycin", "clozapine",
            "domperidone", "droperidol", "ondansetron", "pimozide", "risperidone",
            "terfenadine",
            "diltiazem", "metoprolol", "mexiletine", "ranolazine", "tamoxifen", "verapamil"]
SMOKE_DRUGS = ["dofetilide", "ondansetron", "verapamil"]

CFGS = [("orig", None), ("full", "full"), ("act", "act")]


def forward_drug(V, tabs, kon_nM, koff, conc_nM, hill_clamp=None):
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-DT / tm); eh = np.exp(-DT / th)
    G = tabs["G"] if tabs["G"] else 1.0
    n = len(V); nlp = n // N_LOOP
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


def drug_params(card3, drug):
    ic50s, nhs, ic50_lab, nh_lab = [], [], [], []
    pts = []; n_sat = 0
    for key, ds in card3["datasets"].items():
        if key.split("|")[1] != drug:
            continue
        L1 = ds.get("L1") or {}
        if L1.get("ic50_self"):
            ic50s.append(L1["ic50_self"]); nhs.append(L1["nh_self"])
        if L1.get("ic50_lab"):
            ic50_lab.append(L1["ic50_lab"]); nh_lab.append(L1["nh_lab"])
        for u in ds.get("units") or []:
            k = u.get("k_obs")
            if k is None:
                continue
            if k >= SAT_KOBS:
                n_sat += 1; continue
            pts.append((u["conc_nM"], k))
    ic50 = float(np.median(ic50s)) if ic50s else None
    nh = float(np.median(nhs)) if nhs else None
    kon = koff = koff_fit = None; degen = None
    if pts:
        tiers = sorted({c for c, _ in pts})
        if len(tiers) >= 2:
            x = np.array([c for c, _ in pts]); y = np.array([k for _, k in pts])
            sl, ic, r, p, se = stats.linregress(x, y)
            koff_fit = float(ic)
            if sl > 0 and ic50:
                kon = float(sl); koff = float(sl * ic50)
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


def auc_high_low(s_by_drug, risk_by_drug):
    hi = [s_by_drug[d] for d in s_by_drug if risk_by_drug[d] == 2]
    lo = [s_by_drug[d] for d in s_by_drug if risk_by_drug[d] == 0]
    if not hi or not lo:
        return None
    u, _ = stats.mannwhitneyu(hi, lo, alternative="greater")
    return float(u / (len(hi) * len(lo)))


def run_config(cfg_name, swap_which, eng, V, card3, cmax, risk, drugs):
    tabs0 = anchor_tabs(eng)
    tabs = swap_tau_m(tabs0, eng, swap_which) if swap_which else tabs0
    print(f"[{cfg_name}] anchor G={tabs['G']:.4f}", flush=True)

    Q0 = forward_drug(V, tabs, 0.0, 0.0, 0.0)
    q0_last, q0_mid = Q0[-1], Q0[SS_CHECK_LOOP - 1]
    m_arr, _h = eng.forward(V, tabs)
    m_bar = float(m_arr[-len(m_arr) // N_LOOP:].mean())
    print(f"[{cfg_name}] m_bar(AP末循环)={m_bar:.6f}  (封卷原值 0.255931)", flush=True)

    res = {}
    for i, d in enumerate(drugs):
        p = drug_params(card3, d)
        entry = dict(risk=risk[d], cmax_nM=cmax[d], **p)
        S, S_mid, S_static, S_eq = {}, {}, {}, {}
        for mult in CONC_MULT:
            conc = cmax[d] * mult
            b = hill_block(p["ic50"], p["nh"], conc)
            S_static[str(mult)] = b
            if p["ic50"]:
                S_eq[str(mult)] = float(conc * m_bar / (conc * m_bar + p["ic50"]))
            if p["degen"]:
                if b is None:
                    S[str(mult)] = None; continue
                Qd = forward_drug(V, tabs, 0.0, 0.0, 0.0, hill_clamp=b)
            else:
                Qd = forward_drug(V, tabs, p["kon"], p["koff"], conc)
            S[str(mult)] = float(1.0 - Qd[-1] / q0_last)
            S_mid[str(mult)] = float(1.0 - Qd[SS_CHECK_LOOP - 1] / q0_mid)
        entry["S_transient"] = S
        entry["S_transient_loop6"] = S_mid
        entry["S_eq"] = S_eq
        entry["S_static_Hill"] = S_static
        res[d] = entry
        print(f"  [{cfg_name}][{i+1}/{len(drugs)}] {d} done", flush=True)

    s1map = {d: r["S_eq"]["1"] for d, r in res.items() if r["S_eq"].get("1") is not None}
    rho, pval = (np.nan, np.nan)
    if len(s1map) >= 8:
        rho, pval = stats.spearmanr([s1map[d] for d in s1map], [risk[d] for d in s1map])
        pval = pval / 2.0 if rho > 0 else 1.0 - pval / 2.0
    auc = auc_high_low(s1map, risk)
    s1tr = {d: r["S_transient"]["1"] for d, r in res.items() if r["S_transient"].get("1") is not None}
    rho_tr = p_tr = np.nan
    if len(s1tr) >= 8:
        rho_tr, p_tr = stats.spearmanr([s1tr[d] for d in s1tr], [risk[d] for d in s1tr])
        p_tr = p_tr / 2.0 if rho_tr > 0 else 1.0 - p_tr / 2.0
    verdict = {"m_bar": m_bar,
               "V1_Spearman": {"rho": float(rho), "p_one_side": float(pval),
                               "过": bool(rho >= 0.60 and pval < 0.01)},
               "V2_AUC高低": {"auc": auc, "过": bool(auc is not None and auc >= 0.80)},
               "登记_暂态84拍版": {"rho": float(rho_tr), "p_one_side": float(p_tr)}}
    print(f"[{cfg_name}] V1 rho={rho:.3f} p={pval:.4f} {'过' if verdict['V1_Spearman']['过'] else '不过'}"
          f"  V2 AUC={auc}  暂态rho={rho_tr:.3f}", flush=True)
    return {"verdict": verdict, "逐药": res}


def main():
    print("=" * 76, flush=True)
    print(" 药物卡4 换表版（orig / full / act 三配置）", flush=True)
    eng = load_engine()
    V1 = sio.loadmat(F_PROTO)["T"].flatten().astype(float)
    V = np.tile(V1, N_LOOP)
    print(f"[protocol] ap loop {len(V1) * DT:.4f}s x {N_LOOP}", flush=True)
    card3 = json.load(open(F_CARD3, encoding="utf-8"))
    cmax, risk = {}, {}
    with open(F_CIPA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = row["drug"].strip()
            if d in DRUGS_24:
                cmax[d] = float(row["therapeutic"]); risk[d] = int(row["CiPA"])
    drugs = SMOKE_DRUGS if SMOKE else DRUGS_24

    out = {}
    for cfg_name, sw in CFGS:
        ck = os.path.join(HERE, f"2026-11-09_药物卡4_换表_ckpt_{cfg_name}{'_冒烟' if SMOKE else ''}.json")
        if os.path.exists(ck):
            print(f"[{cfg_name}] 断点已存在，跳过", flush=True)
            out[cfg_name] = json.load(open(ck, encoding="utf-8"))
            continue
        r = run_config(cfg_name, sw, eng, V, card3, cmax, risk, drugs)
        json.dump(r, open(ck, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        out[cfg_name] = r
        print(f"[{cfg_name}] ckpt saved", flush=True)

    summary = {c: out[c]["verdict"] for c, _ in CFGS if c in out}
    sfx = "_冒烟" if SMOKE else ""
    fj = os.path.join(HERE, f"2026-11-09_药物卡4_换表{sfx}_结果.json")
    json.dump({"说明": "药物卡4 τ_m 换表三配置；判线冻结；S_eq 主判对 m_bar 公共因子秩不变",
               "汇总": summary, "封卷对照": {"V1_rho": 0.47502055022717615, "V2_AUC": 0.8095238095238095,
                                          "暂态rho": 0.33242087717669516, "m_bar": 0.2559312493795162}},
              open(fj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("=" * 76, flush=True)
    for c, v in summary.items():
        print(f"  [{c}] m_bar={v['m_bar']:.4f} V1 rho={v['V1_Spearman']['rho']:.3f} "
              f"V2 AUC={v['V2_AUC高低']['auc']} 暂态rho={v['登记_暂态84拍版']['rho']:.3f}", flush=True)
    print(f"  saved: {fj}", flush=True)


if __name__ == "__main__":
    main()
