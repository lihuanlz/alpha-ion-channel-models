# -*- coding: utf-8 -*-
"""
2026-09-15 · ORd α 化组装 + AP 仿真（v1.1：Kr+Ks 主臂 + INa 登记失败臂）
====================================================================================
预注册：结果\\预注册_ORd三通道α化组装_2026-09-15.md（v1.0 + v1.1 修订，判线正式跑前钉死）

v1.1 要点：
  主臂 = Kr+Ks 双通道 α 化（INa 保留官方）；
  N-arm = INa 换芯登记臂（G∈{1,16}×100拍，文档化"表观 m∞ 细胞级不可携带"失败模式）；
  ICaL 保留官方（CaV12 模型卡 §5 激活表缺口）。
56 态：0–48 ORd、49 qNet、50 m_Kr、51 h_Kr、52 D_Kr、53 m_Na、54 h_Na、55 a_dyn_Ks。

判线（v1.1 §3，正式跑前钉死）：
  J1 稳态：末10拍 APD90 极差 <0.5ms（CL=1000，500拍）
  J2 形态：|APD90−252.118|≤2%；APA∈[124.4,130.0]；RMP∈[−95,−85]；qNet∈[0.05,0.12]
  J3a Kr单换(G=0.2943)：APD90 与 B2-stat 253.965 差≤1%
  J3b Ks单换(G_Ks 标定值)：J1+J2 带内
  J3c N-arm 登记成立：APA<90 且 G×16 下 ΔAPA<5mV（饱和签名；APA≥90 属意外，升级重议）
  J4 上冲：dV/dt_max∈[50,500] V/s（登记比值）
  J5 频率：CL{2000,1000,500,300}×200拍 两臂，APD90 随频率单调缩短 + 1:1 夺获（登记判）
  J6 电流量级：IKr/IKs 峰与参考臂比值∈[0.2,5]（INa/ICaL 仅 ≈1 sanity 登记）
  总判：J1+J2+J3+J4 全过 = 组装封卷。J5/J6 登记判。
敏感性臂（正式）：S1 Q10_Ks=2.0 / S2 Q10_Ks=3.0 / S3 τ_act×0.5（各重标 G_Ks + 300拍）

环境变量：SMOKE=1 冒烟。
纪律：本侧 ast.parse + SMOKE 冒烟；正式跑用户 Spyder：
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-15_ORd三通道α化组装.py' --wdir
输出：本脚本同目录 _结果.json/.png（冒烟带 _冒烟 后缀）。
"""

import os
import sys
import csv
import json
import math
import time
import importlib.util

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
B0_SS = os.path.join(BASE, "StageB_B0稳态_CL1000.json")
FEDIDA_CSV = os.path.join(BASE, "2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv")

SMOKE = os.environ.get("SMOKE", "0") == "1"

# ---- 拍数预算 ----
CAL_BEATS = int(os.environ.get("CAL_BEATS", "12" if SMOKE else "60"))
REF_BEATS = int(os.environ.get("REF_BEATS", "25" if SMOKE else "500"))
MAIN_BEATS = int(os.environ.get("MAIN_BEATS", "25" if SMOKE else "500"))
J3_BEATS = int(os.environ.get("J3_BEATS", "15" if SMOKE else "200"))
J3A_BEATS = int(os.environ.get("J3A_BEATS", "15" if SMOKE else "500"))   # 与 B2 锚同拍数，同基准对拍
NARM_BEATS = int(os.environ.get("NARM_BEATS", "10" if SMOKE else "100"))
J5_BEATS = int(os.environ.get("J5_BEATS", "12" if SMOKE else "200"))
S_BEATS = int(os.environ.get("S_BEATS", "300"))
KR_ITERS = 3 if SMOKE else 8
J5_CLS = [1000.0, 500.0] if SMOKE else [2000.0, 1000.0, 500.0, 300.0]
RUN_SENS = (not SMOKE)

DT_REC = 0.1
B0_APD90 = 252.1181387725961
B2_GKR = 0.29427271762092816
B2_CTRL_APD90 = 253.96494081245848

# ---- 温度桥（预注册 §2.4 钉死）----
Q10_KR = 2.5; QFAC_KR = Q10_KR ** ((37.0 - 21.5) / 10.0)     # ÷4.138
Q10_NA = 1.8; QFAC_NA = Q10_NA ** ((37.0 - 35.0) / 10.0)     # ÷1.125
Q10_KS = 2.5; QFAC_KS = Q10_KS ** 1.5                        # ÷3.952（主臂）

RT_F = 8314.0 * 310.0 / 96485.0
NAO = 140.0
PKNA = 0.01833
BETA = 0.30


def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Lut:
    """线性插值查找表，端点外钳制。"""
    def __init__(self, xs, ys):
        o = np.argsort(xs)
        self.xs = np.asarray(xs, dtype=float)[o]
        self.ys = np.asarray(ys, dtype=float)[o]
    def __call__(self, v):
        return float(np.interp(v, self.xs, self.ys))


# ----------------------------------------------------------------------------
# α 表
# ----------------------------------------------------------------------------
def build_tabs_kr(eng):
    """hERG 四表（B2 build_alpha_tabs 同逻辑，TAU_ARM=1，Q10=2.5）。"""
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)

    def scaled(tab):
        out = {}
        for x, ly in zip(tab.xs, tab.ly):
            out[float(x)] = float(np.exp(ly)) * 1000.0 / QFAC_KR
        return eng.Tab(out, log_y=True)

    return {"m_ss": tabs["m_ss"], "h_ss": tabs["h_ss"],
            "tau_m": scaled(tabs["tau_m"]), "tau_h": scaled(tabs["tau_h"])}


def build_tabs_na():
    """Nav1.5 三表（模型卡 §3.1/3.3/3.4；τ ÷QFAC_NA）——仅供 N-arm 登记臂。"""
    tm = Lut([-20, -10, 0, 10, 20, 30, 40],
             [0.075, 0.0323, 0.0262, 0.0264, 0.0251, 0.0216, 0.0216])
    th = Lut([-30, -20, -10, 0, 10, 20, 30, 40],
             [1.639, 0.494, 0.613, 0.511, 0.414, 0.335, 0.303, 0.270])
    return {"m_ss": lambda v: 1.0 / (1.0 + math.exp(-(v + 35.2) / 3.67)),
            "h_ss": lambda v: 1.0 / (1.0 + math.exp((v + 66.1) / 4.79)),
            "tau_m": lambda v: tm(v) / QFAC_NA,
            "tau_h": lambda v: th(v) / QFAC_NA}


def build_tabs_ks_scaled(q10, tscale):
    """IKs：a_ss Boltzmann + Fedida τ_act 25 点（s→ms，÷q10^1.5，×tscale）。"""
    qfac = q10 ** 1.5
    xs, ys = [], []
    with open(FEDIDA_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["V_mV"]))
            ys.append(float(row["tau_s"]) * 1000.0 / qfac * tscale)
    tact = Lut(xs, ys)
    return {"a_ss": lambda v: 1.0 / (1.0 + math.exp(-(v - 25.05) / 22.6)),
            "tau_act": tact}


def build_tabs_ks():
    return build_tabs_ks_scaled(Q10_KS, 1.0)


# ----------------------------------------------------------------------------
# 组装 rhs（B0 移植件 + 通道置换；冻结官方门；电导置零）
# ----------------------------------------------------------------------------
def build_asm(b0, P_ctrl, tabs, G, swaps):
    P2 = list(P_ctrl)
    if "Kr" in swaps:
        P2[1] = float("inf")
    if "Na" in swaps:
        P2[3] = float("inf")
    if "Ks" in swaps:
        P2[4] = float("inf")
    rhs0 = b0.build_model(P2)
    ko = P_ctrl[b0.PARS_NAMES.index("ko")]
    rad, L = 0.0011, 0.01
    vcell = 1000 * 3.14 * rad * rad * L
    Ageo = 2 * 3.14 * rad * rad + 2 * 3.14 * rad * L
    Acap_Fvmyo = 2 * Ageo / (96485.0 * 0.68 * vcell)

    tkr = tabs.get("Kr")
    tna = tabs.get("Na")
    tks = tabs.get("Ks")

    def rhs(t, y):
        dy, cur = rhs0(t, y)
        v = y[0]
        dy_a = list(dy) + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if "Na" in swaps:
            for i in range(9, 15):
                dy_a[i] = 0.0
        if "Ks" in swaps:
            dy_a[33] = 0.0
            dy_a[34] = 0.0
        if "Kr" in swaps:
            for i in range(39, 49):
                dy_a[i] = 0.0
        ki, nai = y[3], y[1]
        IKr_a = INa_a = IKs_a = 0.0
        if "Kr" in swaps:
            m, h = y[50], y[51]
            dy_a[50] = (tkr["m_ss"](v) - m) / tkr["tau_m"](v)
            dy_a[51] = (tkr["h_ss"](v) - h) / tkr["tau_h"](v)
            EK = RT_F * math.log(ko / ki)
            IKr_a = G["Kr"] * m * h * (v - EK)
        if "Na" in swaps:
            m, h = y[53], y[54]
            dy_a[53] = (tna["m_ss"](v) - m) / tna["tau_m"](v)
            dy_a[54] = (tna["h_ss"](v) - h) / tna["tau_h"](v)
            ENa = RT_F * math.log(NAO / nai)
            INa_a = G["Na"] * (m ** 3.0) * h * (v - ENa)
        if "Ks" in swaps:
            ad = y[55]
            ass = tks["a_ss"](v)
            dy_a[55] = ((1.0 - BETA) * ass - ad) / tks["tau_act"](v)
            EKs = RT_F * math.log((ko + PKNA * NAO) / (ki + PKNA * nai))
            IKs_a = G["Ks"] * (BETA * ass + ad) * (v - EKs)
        dy_a[0] -= (IKr_a + INa_a + IKs_a)
        dy_a[1] -= INa_a * Acap_Fvmyo
        dy_a[3] -= (IKr_a + IKs_a) * Acap_Fvmyo
        dy_a[49] += (IKr_a + IKs_a)
        INa_eff = INa_a if "Na" in swaps else cur[0]
        IKr_eff = IKr_a if "Kr" in swaps else cur[4]
        IKs_eff = IKs_a if "Ks" in swaps else cur[5]
        return dy_a, (INa_eff, cur[1], cur[2], cur[3], IKr_eff, IKs_eff, cur[6])

    return rhs


def wrap_ref(b0, P_ctrl):
    rhs0 = b0.build_model(P_ctrl)
    def rhs(t, y):
        dy, cur = rhs0(t, y)
        return list(dy), cur
    return rhs


# ----------------------------------------------------------------------------
# 跑拍
# ----------------------------------------------------------------------------
def run_beats(b0, rhs, y0, nbeats, cl, label="", keep_last=False, keep_currents=False):
    from scipy.integrate import solve_ivp
    f = lambda t, y: rhs(t, y)[0]
    t_rec = np.arange(0.0, cl + 1e-9, DT_REC)
    y = np.array(y0, dtype=float)
    apd = np.full(nbeats, np.nan)
    apa = np.full(nbeats, np.nan)
    qnet = np.full(nbeats, np.nan)
    out = {"t": None, "v": None, "cur": None, "y_end": None}
    t0 = time.time()
    nd = 0
    for b in range(nbeats):
        q0 = y[49]
        sol = solve_ivp(f, (0.0, cl), y, method="LSODA", t_eval=t_rec,
                        rtol=1e-7, atol=1e-9)
        if not sol.success:
            print(f"    {label} 拍 {b + 1}: LSODA 失败 {sol.message}", flush=True)
            break
        v = sol.y[0]
        y = sol.y[:, -1].copy()
        a90, apa_b, _, _ = b0.apd90(t_rec, v)
        apd[b] = a90
        apa[b] = apa_b
        qnet[b] = (y[49] - q0) * 1e-3
        nd = b + 1
        if b == nbeats - 1 and (keep_last or keep_currents):
            out["t"] = t_rec
            out["v"] = v.copy()
            if keep_currents:
                cur = np.empty((7, len(t_rec)))
                for k in range(len(t_rec)):
                    _, c7 = rhs(float(t_rec[k]), sol.y[:, k].tolist())
                    cur[:, k] = c7
                out["cur"] = cur
        if (b + 1) % 50 == 0 or b == 0:
            el = time.time() - t0
            print(f"    {label} 拍 {b + 1}/{nbeats}  APD90={apd[b]:7.2f}ms"
                  f"  qNet={qnet[b]:.4f}  用时{el:.0f}s", flush=True)
    out["y_end"] = y
    return apd[:nd], apa[:nd], qnet[:nd], out


def last_med(x, k=10):
    ok = x[~np.isnan(x)]
    return float(np.median(ok[-k:])) if len(ok) else float("nan")


def steady_range(x, k=10):
    ok = x[~np.isnan(x)]
    if len(ok) < k:
        return float("nan")
    return float(np.max(ok[-k:]) - np.min(ok[-k:]))


def dvdts_max(t, v):
    return float(np.max(np.diff(v) / (t[1] - t[0])))


def morph_check(apd_last, apa_last, rmp_last, qnet_last):
    return {
        "APD90_ms": {"value": apd_last, "pass": bool(abs(apd_last - B0_APD90) <= 0.02 * B0_APD90)},
        "APA_mV": {"value": apa_last, "pass": bool(124.4 <= apa_last <= 130.0)},
        "RMP_mV": {"value": rmp_last, "pass": bool(-95.0 <= rmp_last <= -85.0)},
        "qNet_uC_uF": {"value": qnet_last, "pass": bool(0.05 <= qnet_last <= 0.12)},
    }


def cal_kr_on_apd(b0, P_ctrl, tabs, G_fix, swaps, y0, target, label):
    """G_Kr 网格+二分锚 APD90（APD90 随 G_Kr 单调减）。"""
    kr_grid = [-1.0, -0.53, 0.0] if SMOKE else [-1.0, -0.70, -0.53, -0.22, 0.0]

    def probe(lg):
        G = dict(G_fix)
        G["Kr"] = 10.0 ** lg
        rhs = build_asm(b0, P_ctrl, tabs, G, swaps)
        apd, apa, _, _ = run_beats(b0, rhs, y0, CAL_BEATS, 1000.0, label=label)
        return last_med(apd), last_med(apa)

    pts = []
    for lg in kr_grid:
        a, pa = probe(lg)
        pts.append((lg, a, pa))
        print(f"  [{label}·网格] G_Kr=10^{lg:+.2f}={10 ** lg:.4g}  APD90={a:.2f}  APA={pa:.1f}"
              f"  {'健康' if pa >= 90 else '病理'}", flush=True)
    healthy = [(lg, a) for lg, a, pa in pts if pa >= 90 and not math.isnan(a)]
    bracket = None
    for (l1, a1), (l2, a2) in zip(healthy, healthy[1:]):
        if (a1 - target) * (a2 - target) <= 0 and a1 != a2:
            bracket = (l1, l2)
            break
    if bracket is None:
        if healthy:
            lg, a = min(healthy, key=lambda p: abs(p[1] - target))
            print(f"  [{label}] 无括弧，取最贴线健康点 G_Kr=10^{lg:+.2f}（贴线登记）", flush=True)
            return 10.0 ** lg, a, bool(abs(a - target) <= 0.02 * target)
        lg, a, pa = min(pts, key=lambda p: abs((p[1] if not math.isnan(p[1]) else 9e3) - target))
        print(f"  [{label}] 无健康点——照实登记", flush=True)
        return 10.0 ** lg, a, False
    lo, hi = bracket
    best = None
    for it in range(KR_ITERS):
        mid = 0.5 * (lo + hi)
        a, pa = probe(mid)
        print(f"  [{label}·二分 {it + 1}] G_Kr=10^{mid:.3f}={10 ** mid:.4g}  APD90={a:.2f}  APA={pa:.1f}", flush=True)
        if pa < 90.0 or math.isnan(a) or a > target:
            lo = mid           # APD 过长/不健康 -> G_Kr 太小，上移
        else:
            hi = mid
        best = (mid, a, pa)
        if pa >= 90 and not math.isnan(a) and abs(a - target) <= 0.02 * target:
            return 10.0 ** mid, a, True
    mid, a, pa = best
    return 10.0 ** mid, a, bool(pa >= 90 and not math.isnan(a) and abs(a - target) <= 0.02 * target)


def cal_ks_on_charge(b0, P_ctrl, tabs, y0, qKs_ref, label):
    """G_Ks 线性锚官方 IKs 电荷（swaps={Ks} 健康语境，IKs_α 对 G 精确线性）。"""
    rhs = build_asm(b0, P_ctrl, tabs, {"Ks": 1e-2}, ("Ks",))
    _, _, _, out_p = run_beats(b0, rhs, y0, CAL_BEATS, 1000.0, label=f"{label}-探针",
                               keep_last=True, keep_currents=True)
    qKs_probe = float(np.trapezoid(out_p["cur"][5], out_p["t"]))
    if qKs_probe <= 0 or qKs_ref <= 0:
        print(f"  [{label}] qKs 探针={qKs_probe:.3e} 非正——线性锚失效，G_Ks=1e-2 登记", flush=True)
        return 1e-2, qKs_probe, float("nan"), False
    G_Ks = 1e-2 * qKs_ref / qKs_probe
    rhs = build_asm(b0, P_ctrl, tabs, {"Ks": G_Ks}, ("Ks",))
    _, _, _, out_v = run_beats(b0, rhs, y0, CAL_BEATS, 1000.0, label=f"{label}-验证",
                               keep_last=True, keep_currents=True)
    qKs_v = float(np.trapezoid(out_v["cur"][5], out_v["t"]))
    ok = bool(abs(qKs_v / qKs_ref - 1.0) <= 0.05)
    print(f"  [{label}] 探针 qKs={qKs_probe:.3e} 靶 {qKs_ref:.3e} -> G_Ks={G_Ks:.4g}；"
          f"验证比={qKs_v / qKs_ref:.3f} {'过' if ok else '贴线登记'}", flush=True)
    return G_Ks, qKs_probe, qKs_v, ok


# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 76, flush=True)
    print(" ORd α 化组装 + AP 仿真（v1.1：Kr+Ks 主臂 + INa 登记失败臂）", flush=True)
    print(f" 模式: {'冒烟 SMOKE' if SMOKE else '正式'}  Q10: Kr {Q10_KR} Ks {Q10_KS}(÷{QFAC_KS:.3f})", flush=True)
    print(f" 拍数: 参考{REF_BEATS} 主臂{MAIN_BEATS} 标定{CAL_BEATS} J3 {J3_BEATS} N-arm {NARM_BEATS} J5 {J5_BEATS}", flush=True)
    print("=" * 76, flush=True)

    b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")
    eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")
    P_ctrl = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    with open(B0_SS, encoding="utf-8") as fh:
        y_ss = json.load(fh)["states50"]
    v0 = y_ss[0]

    tabs_kr = build_tabs_kr(eng)
    tabs_na = build_tabs_na()
    tabs_ks = build_tabs_ks()
    print(f"[表] hERG四表(16713003, ÷{QFAC_KR:.3f}) + Nav三表(N-arm专用) + IKs Fedida25点(÷{QFAC_KS:.3f}) 装载", flush=True)

    y0_alpha = list(y_ss) + [tabs_kr["m_ss"](v0), tabs_kr["h_ss"](v0), 0.0,
                             tabs_na["m_ss"](v0), tabs_na["h_ss"](v0),
                             (1.0 - BETA) * tabs_ks["a_ss"](v0)]
    print(f"[初值] α 态 @v={v0:.2f}mV: m_Kr={y0_alpha[50]:.4f} h_Kr={y0_alpha[51]:.4f}"
          f" h_Na={y0_alpha[54]:.4f} a_dyn={y0_alpha[55]:.5f}", flush=True)

    log = {"meta": {"smoke": SMOKE, "version": "v1.1", "t_start": time.strftime("%H:%M:%S"),
                    "Q10": {"Kr": Q10_KR, "Ks": Q10_KS},
                    "beats": {"ref": REF_BEATS, "main": MAIN_BEATS}}}

    # ===================== 参考臂 =====================
    print("\n[参考臂] 官方 ORd，同初值，CL=1000 ...", flush=True)
    rhs_ref = wrap_ref(b0, P_ctrl)
    apd_r, apa_r, qn_r, out_r = run_beats(b0, rhs_ref, y_ss, REF_BEATS, 1000.0,
                                          label="参考", keep_last=True, keep_currents=True)
    APD_ref = last_med(apd_r)
    APA_ref = last_med(apa_r)
    qNet_ref = last_med(qn_r)
    RMP_ref = float(np.min(out_r["v"]))
    DVDT_ref = dvdts_max(out_r["t"], out_r["v"])
    qKs_ref = float(np.trapezoid(out_r["cur"][5], out_r["t"]))
    qKr_ref = float(np.trapezoid(out_r["cur"][4], out_r["t"]))
    print(f"[参考臂] APD90={APD_ref:.3f}ms APA={APA_ref:.2f}mV RMP={RMP_ref:.2f}mV"
          f" qNet={qNet_ref:.4f} dVdt={DVDT_ref:.0f}V/s qKs={qKs_ref:.3e} qKr={qKr_ref:.3e}", flush=True)
    log["reference"] = {"APD90_ms": APD_ref, "APA_mV": APA_ref, "RMP_mV": RMP_ref,
                        "qNet": qNet_ref, "dVdt_Vs": DVDT_ref, "qKs": qKs_ref, "qKr": qKr_ref,
                        "apd_trace": [None if math.isnan(x) else x for x in apd_r],
                        "v_last": out_r["v"].tolist()}

    # ===================== 标定（v1.1 §5）=====================
    print("\n[标定 1/2] G_Ks（swaps={Ks}，线性锚 qKs_ref）...", flush=True)
    G_Ks, qKs_probe, qKs_v, ok_ks = cal_ks_on_charge(
        b0, P_ctrl, {"Ks": tabs_ks}, y0_alpha, qKs_ref, "标定Ks")

    print("\n[标定 2/2] G_Kr（swaps={Kr,Ks}，靶 APD90=%.3f ±2%%）..." % APD_ref, flush=True)
    tabs_main = {"Kr": tabs_kr, "Ks": tabs_ks}
    G_Kr, apd_cal, ok_kr = cal_kr_on_apd(
        b0, P_ctrl, tabs_main, {"Ks": G_Ks}, ("Kr", "Ks"), y0_alpha, APD_ref, "标定Kr")
    print(f"[标定] 终值 G_Kr={G_Kr:.4g}（{'过' if ok_kr else '贴线登记'}）  G_Ks={G_Ks:.4g}"
          f"（{'过' if ok_ks else '贴线登记'}）", flush=True)
    log["calibration"] = {"G_Kr": G_Kr, "G_Ks": G_Ks, "ok_kr": ok_kr, "ok_ks": ok_ks,
                          "qKs_probe": qKs_probe, "qKs_verify": qKs_v, "apd_cal": apd_cal}
    G_ALL = {"Kr": G_Kr, "Ks": G_Ks}

    # ===================== 主臂 =====================
    print("\n[主臂] Kr+Ks 双通道 α 化，500 拍 ...", flush=True)
    rhs_main = build_asm(b0, P_ctrl, tabs_main, G_ALL, ("Kr", "Ks"))
    apd_m, apa_m, qn_m, out_m = run_beats(b0, rhs_main, y0_alpha, MAIN_BEATS, 1000.0,
                                          label="主臂", keep_last=True, keep_currents=True)
    APD_m = last_med(apd_m)
    APA_m = last_med(apa_m)
    qNet_m = last_med(qn_m)
    RMP_m = float(np.min(out_m["v"]))
    DVDT_m = dvdts_max(out_m["t"], out_m["v"])
    J1_range = steady_range(apd_m)
    J1 = bool(J1_range < 0.5)
    crit2 = morph_check(APD_m, APA_m, RMP_m, qNet_m)
    J2 = all(c["pass"] for c in crit2.values())
    print(f"[主臂] APD90={APD_m:.3f} APA={APA_m:.2f} RMP={RMP_m:.2f} qNet={qNet_m:.4f}"
          f" dVdt={DVDT_m:.0f}  J1极差={J1_range:.3f}", flush=True)

    # ===================== J3 回溯 =====================
    print("\n[J3a] Kr 单换（G=0.2943，B2 锚，同 500 拍基准）...", flush=True)
    rhs = build_asm(b0, P_ctrl, {"Kr": tabs_kr}, {"Kr": B2_GKR}, ("Kr",))
    apd1, apa1, qn1, _ = run_beats(b0, rhs, y0_alpha, J3A_BEATS, 1000.0, label="J3a-Kr单换")
    a1 = last_med(apd1)
    dev1 = abs(a1 - B2_CTRL_APD90) / B2_CTRL_APD90
    j3a = {"APD90": a1, "target": B2_CTRL_APD90, "dev": dev1, "pass": bool(dev1 <= 0.01)}
    print(f"  APD90={a1:.2f} vs B2 {B2_CTRL_APD90:.2f} 偏={dev1 * 100:.2f}%  {'过' if j3a['pass'] else '不过'}", flush=True)

    print("\n[J3b] Ks 单换（G_Ks 标定值）...", flush=True)
    rhs = build_asm(b0, P_ctrl, {"Ks": tabs_ks}, {"Ks": G_Ks}, ("Ks",))
    apd3, apa3, qn3, out3 = run_beats(b0, rhs, y0_alpha, J3_BEATS, 1000.0, label="J3b-Ks单换", keep_last=True)
    c3 = morph_check(last_med(apd3), last_med(apa3), float(np.min(out3["v"])), last_med(qn3))
    j3b = {"crit": c3, "steady": steady_range(apd3),
           "pass": bool(all(c["pass"] for c in c3.values()) and steady_range(apd3) < 0.5)}
    print(f"  APD90={last_med(apd3):.2f} APA={last_med(apa3):.2f} qNet={last_med(qn3):.4f}"
          f"  {'过' if j3b['pass'] else '不过'}", flush=True)

    print("\n[J3c] N-arm：INa 换芯登记臂（G_Na∈{1,16}，预期失败签名）...", flush=True)
    narm = {}
    for gna in (1.0, 16.0):
        rhs = build_asm(b0, P_ctrl, {"Na": tabs_na}, {"Na": gna}, ("Na",))
        apdn, apan, qnn, outn = run_beats(b0, rhs, y0_alpha, NARM_BEATS, 1000.0,
                                          label=f"N-arm-G{gna:g}", keep_last=True)
        dv = dvdts_max(outn["t"], outn["v"]) if outn["v"] is not None else float("nan")
        narm[f"G{gna:g}"] = {"APD90": last_med(apdn), "APA": last_med(apan),
                             "dVdt": dv, "qNet": last_med(qnn)}
        print(f"  G_Na={gna:g}: APA={last_med(apan):.2f}mV  APD90={last_med(apdn):.2f}"
              f"  dVdt={dv:.0f}V/s", flush=True)
    dAPA = abs(narm["G16"]["APA"] - narm["G1"]["APA"])
    j3c_expected_fail = bool(narm["G1"]["APA"] < 90.0 and narm["G16"]["APA"] < 90.0 and dAPA < 5.0)
    narm["delta_APA_G16vsG1"] = dAPA
    narm["registration"] = ("表观 m∞（Rs 伪影坐标）细胞级不可携带：AP 脚部 m³·h 窗口关闭，"
                            "无再生性上冲；G×16 饱和不变 -> 非电导问题，与 Nav15 模型卡 §3.4 登记一致")
    narm["pass"] = j3c_expected_fail
    print(f"  ΔAPA(G16 vs G1)={dAPA:.2f}mV  饱和签名 {'成立' if j3c_expected_fail else '不成立——意外，升级重议'}", flush=True)

    J3 = bool(j3a["pass"] and j3b["pass"] and narm["pass"])
    log["J3"] = {"J3a_Kr_only": j3a, "J3b_Ks_only": j3b, "J3c_N_arm": narm}

    # ===================== J4 =====================
    J4 = bool(50.0 <= DVDT_m <= 500.0)
    log["J4"] = {"dVdt_main": DVDT_m, "dVdt_ref": DVDT_ref,
                 "ratio": DVDT_m / DVDT_ref if DVDT_ref else None, "pass": J4}

    # ===================== J5 频率 =====================
    print("\n[J5 频率] 两臂 × CL ...", flush=True)
    j5 = {"ref": {}, "main": {}}
    for arm, rhs_a, y0_a in (("ref", rhs_ref, y_ss), ("main", rhs_main, y0_alpha)):
        for cl in J5_CLS:
            apd_c, apa_c, _, _ = run_beats(b0, rhs_a, y0_a, J5_BEATS, cl, label=f"J5-{arm}-CL{cl:.0f}")
            j5[arm][f"{cl:.0f}"] = {"APD90": last_med(apd_c), "APA_min": float(np.nanmin(apa_c)),
                                    "steady": steady_range(apd_c)}
            print(f"  {arm} CL={cl:.0f}: APD90={last_med(apd_c):.2f}  APA_min={np.nanmin(apa_c):.1f}", flush=True)
    def mono(d):
        vs = [d[f"{cl:.0f}"]["APD90"] for cl in sorted(J5_CLS, reverse=True)]
        return all(not math.isnan(x) for x in vs) and all(x >= y for x, y in zip(vs, vs[1:]))
    def capture(d):
        return all(d[f"{cl:.0f}"]["APA_min"] >= 90.0 for cl in J5_CLS)
    j5["pass"] = bool(mono(j5["ref"]) and mono(j5["main"]) and capture(j5["ref"]) and capture(j5["main"]))
    log["J5"] = j5

    # ===================== J6 电流量级 =====================
    j6 = {}
    for i, nm in enumerate(("INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1")):
        pk_r = float(np.max(np.abs(out_r["cur"][i])))
        pk_m = float(np.max(np.abs(out_m["cur"][i])))
        ratio = pk_m / pk_r if pk_r > 0 else float("nan")
        j6[nm] = {"pk_ref": pk_r, "pk_main": pk_m, "ratio": ratio,
                  "pass": bool(0.2 <= ratio <= 5.0) if pk_r > 0 else None}
    J6 = bool(j6["IKr"]["pass"] and j6["IKs"]["pass"])
    log["J6"] = j6

    # ===================== 总判 =====================
    overall = bool(J1 and J2 and J3 and J4)
    print("\n" + "=" * 76, flush=True)
    print(f" J1 稳态: 极差 {J1_range:.3f}ms (<0.5) {'✓' if J1 else '✗'}", flush=True)
    print(f" J2 形态: APD90 {APD_m:.2f}({'✓' if crit2['APD90_ms']['pass'] else '✗'})"
          f" APA {APA_m:.1f}({'✓' if crit2['APA_mV']['pass'] else '✗'})"
          f" RMP {RMP_m:.1f}({'✓' if crit2['RMP_mV']['pass'] else '✗'})"
          f" qNet {qNet_m:.3f}({'✓' if crit2['qNet_uC_uF']['pass'] else '✗'}) -> {'过' if J2 else '不过'}", flush=True)
    print(f" J3a Kr单换偏 {j3a['dev'] * 100:.2f}%({'✓' if j3a['pass'] else '✗'})"
          f"  J3b Ks单换({'✓' if j3b['pass'] else '✗'})"
          f"  J3c N-arm 饱和签名({'✓' if narm['pass'] else '✗'}) -> {'过' if J3 else '不过'}", flush=True)
    print(f" J4 上冲: {DVDT_m:.0f} V/s∈[50,500] {'✓' if J4 else '✗'}（参考 {DVDT_ref:.0f}，比 {DVDT_m / DVDT_ref:.2f}）", flush=True)
    print(f" J5 频率: {'过' if j5['pass'] else '不过'}（登记判）  J6 IKr/IKs 量级: {'过' if J6 else '带外登记'}（登记判）", flush=True)
    print(f" 【总判 {'过线 —— Kr+Ks 双通道 α 化组装封卷 + INa 不可携带判词' if overall else '不过线 —— 按预注册 §六 登记'}】", flush=True)
    print("=" * 76, flush=True)

    log["main"] = {"APD90_ms": APD_m, "APA_mV": APA_m, "RMP_mV": RMP_m, "qNet": qNet_m,
                   "dVdt_Vs": DVDT_m, "J1_range": J1_range, "crit2": crit2,
                   "apd_trace": [None if math.isnan(x) else x for x in apd_m],
                   "v_last": out_m["v"].tolist()}
    log["verdict"] = {"J1": J1, "J2": J2, "J3": J3, "J4": J4, "J5": j5["pass"], "J6": J6,
                      "overall": overall}

    # ===================== 敏感性臂（正式）=====================
    sens = {}
    if RUN_SENS:
        for arm_name, q10ks, tscale in (("S1_Q10Ks2.0", 2.0, 1.0),
                                        ("S2_Q10Ks3.0", 3.0, 1.0),
                                        ("S3_tauKs0.5", 2.5, 0.5)):
            print(f"\n[敏感性臂 {arm_name}] Q10_Ks={q10ks} τ×{tscale} ...", flush=True)
            tabs_ks_a = build_tabs_ks_scaled(q10ks, tscale)
            G_Ks_a, _, _, _ = cal_ks_on_charge(b0, P_ctrl, {"Ks": tabs_ks_a}, y0_alpha,
                                               qKs_ref, f"{arm_name}-标定")
            tabs_a = {"Kr": tabs_kr, "Ks": tabs_ks_a}
            rhs_a = build_asm(b0, P_ctrl, tabs_a, {"Kr": G_Kr, "Ks": G_Ks_a}, ("Kr", "Ks"))
            apd_a, apa_a, qn_a, out_a = run_beats(b0, rhs_a, y0_alpha, S_BEATS, 1000.0,
                                                  label=arm_name, keep_last=True)
            c_a = morph_check(last_med(apd_a), last_med(apa_a), float(np.min(out_a["v"])), last_med(qn_a))
            sens[arm_name] = {"G_Ks": G_Ks_a, "APD90": last_med(apd_a), "APA": last_med(apa_a),
                              "qNet": last_med(qn_a), "steady": steady_range(apd_a), "crit": c_a,
                              "J2_pass": all(c["pass"] for c in c_a.values()),
                              "v_last": out_a["v"].tolist()}
            print(f"  {arm_name}: G_Ks={G_Ks_a:.4g} APD90={last_med(apd_a):.2f}"
                  f"  J2 {'过' if sens[arm_name]['J2_pass'] else '不过'}", flush=True)
    log["sensitivity"] = sens

    # ===================== 落盘 =====================
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(BASE, f"2026-09-15_ORd三通道α化组装{tag}.json")
    log["meta"]["runtime_s"] = time.time() - t_start
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(log, fh, ensure_ascii=False)
    print(f"\n 结果落盘: {fjson}", flush=True)

    # ===================== 图 =====================
    import matplotlib
    matplotlib.use("Agg")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(f"ORd α 化组装 v1.1（Kr+Ks 主臂 + INa 登记臂）· {'冒烟' if SMOKE else '正式'}"
                 f" · 总判 {'过' if overall else '不过'}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(out_r["t"], out_r["v"], "k-", lw=1.4, label=f"官方 ORd（APD90 {APD_ref:.1f}）")
    ax.plot(out_m["t"], out_m["v"], "r--", lw=1.2, label=f"α Kr+Ks（APD90 {APD_m:.1f}）")
    ax.set_title("末拍 AP 对拍（主臂）")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("v (mV)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(out_r["t"], out_r["v"], "k-", lw=1.2, label="官方")
    for gna, col in (("G1", "tab:orange"), ("G16", "tab:red")):
        pass
    ax.set_title("N-arm：INa 换芯失败签名（见 J3c）")
    ax.set_xlabel("t (ms)")
    for gna, col in ((1.0, "tab:orange"), (16.0, "tab:red")):
        rhs = build_asm(b0, P_ctrl, {"Na": tabs_na}, {"Na": gna}, ("Na",))
        from scipy.integrate import solve_ivp
        f = lambda t, y: rhs(t, y)[0]
        sol = solve_ivp(f, (0.0, 1000.0), np.array(y0_alpha, dtype=float), method="LSODA",
                        t_eval=np.arange(0.0, 1000.0 + 1e-9, DT_REC), rtol=1e-7, atol=1e-9)
        ax.plot(sol.t, sol.y[0], "--", color=col, lw=1.0,
                label=f"α INa G={gna:g}（APA≈{narm[f'G{gna:g}']['APA']:.0f}）")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(np.arange(1, len(apd_r) + 1), apd_r, "k.-", ms=3, lw=0.6, label="官方")
    ax.plot(np.arange(1, len(apd_m) + 1), apd_m, "r.-", ms=3, lw=0.6, label="α Kr+Ks")
    ax.set_title("APD90 逐拍轨迹")
    ax.set_xlabel("拍"); ax.set_ylabel("APD90 (ms)"); ax.legend(fontsize=8)

    for k, (nm, idx) in enumerate((("IKr", 4), ("IKs", 5), ("ICaL", 3))):
        ax = axes[1, k]
        ax.plot(out_r["t"], out_r["cur"][idx], "k-", lw=1.0, label="官方")
        ax.plot(out_m["t"], out_m["cur"][idx], "r--", lw=1.0, label="α")
        ax.set_title(f"{nm} 末拍（峰比 {j6[nm]['ratio']:.2f}）")
        ax.set_xlabel("t (ms)"); ax.set_ylabel("pA/pF"); ax.legend(fontsize=8)

    ax = axes[2, 0]
    msk = out_r["t"] <= 30.0
    ax.plot(out_r["t"][msk], out_r["v"][msk], "k-", lw=1.4, label="官方")
    ax.plot(out_m["t"][msk], out_m["v"][msk], "r--", lw=1.2, label="α Kr+Ks")
    ax.set_title(f"上冲前 30ms（dV/dt {DVDT_m:.0f} vs {DVDT_ref:.0f} V/s）")
    ax.set_xlabel("t (ms)"); ax.legend(fontsize=8)

    ax = axes[2, 1]
    cls_sorted = sorted(J5_CLS, reverse=True)
    ax.plot([1000.0 / c for c in cls_sorted], [j5["ref"][f"{c:.0f}"]["APD90"] for c in cls_sorted],
            "ko-", ms=5, label="官方")
    ax.plot([1000.0 / c for c in cls_sorted], [j5["main"][f"{c:.0f}"]["APD90"] for c in cls_sorted],
            "r^--", ms=5, label="α Kr+Ks")
    ax.set_title("频率恢复（APD90 vs Hz）")
    ax.set_xlabel("Hz"); ax.set_ylabel("APD90 (ms)"); ax.legend(fontsize=8)

    ax = axes[2, 2]
    ax.axis("off")
    txt = (f"J1 稳态极差 {J1_range:.3f}ms {'✓' if J1 else '✗'}\n"
           f"J2 APD90 {APD_m:.1f}({'✓' if crit2['APD90_ms']['pass'] else '✗'}) "
           f"APA {APA_m:.1f}({'✓' if crit2['APA_mV']['pass'] else '✗'})\n"
           f"   RMP {RMP_m:.1f}({'✓' if crit2['RMP_mV']['pass'] else '✗'}) "
           f"qNet {qNet_m:.3f}({'✓' if crit2['qNet_uC_uF']['pass'] else '✗'})\n"
           f"J3a Kr偏 {j3a['dev'] * 100:.2f}%({'✓' if j3a['pass'] else '✗'}) "
           f"J3b {'✓' if j3b['pass'] else '✗'} J3c {'✓' if narm['pass'] else '✗'}\n"
           f"J4 dV/dt {DVDT_m:.0f}V/s({'✓' if J4 else '✗'})  J5 {'✓' if j5['pass'] else '✗'}  J6 {'✓' if J6 else '✗'}\n"
           f"G: Kr {G_Kr:.3g} Ks {G_Ks:.3g}\n"
           f"用时 {time.time() - t_start:.0f}s\n"
           f"【总判 {'过线' if overall else '不过线'}】")
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, va="top", fontsize=10)

    fpng = os.path.join(BASE, f"2026-09-15_ORd三通道α化组装{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" 图落盘: {fpng}", flush=True)

    if SMOKE:
        print("\n[冒烟完] 正式跑指令（Spyder）：\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-15_ORd三通道α化组装.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
