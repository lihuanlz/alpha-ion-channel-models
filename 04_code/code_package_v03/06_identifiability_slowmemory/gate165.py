# -*- coding: utf-8 -*-
"""gate165.py：代码165 门控侧指标单一实现源（mk165/run_all165/judge165 三处复用，防口径漂移）。
与 gate164 差异（预注册_代码165 §判官差异声明 冻结，164 判词不回溯）：
  ① 仿真通道 sim_gate41→sim_gate148(·,·,·,44)（model 44 = K6a9Ji：Lu 式外加 C_4⇌I_c，
     fi_c=p[24]·e^(ZI·v)、bi_c=p[25]·e^(−ZI·v)，ZI=0.6/25.693；26 参）；
  ② envelope_fit 多初值稳健化：p0 网格 a2∈{0.3,0.6,0.85}×t1∈{0.002,0.005}×t2∈{0.02,0.036,0.08}
     共 18 起点，同 bounds 同 maxfev，取残差平方和最小者（治补记37 门控双盆病理；
     口径变更仅前向适用于 165 起）；
  ③ Q–V 解析平衡 eq() 扩一项 I_c：model 44 拓扑为树（C_4⇌I_c 为悬挂边，无循环流），
     细致平衡成立，平衡权重 w[I_c]=w[C_4]·(p24/p25)·e^(2·ZI·v)，I_c 电荷层位=4
     （C_4⇌I_c 不跨 VSD 层、Ig 口径不变——仅 VSD 跃迁净通量计入）；
     eq44(v)=[Σk·(wC_k+wO_k)+4·wC_4·r(v)]/[Σ(wC+wO)+wC_4·r(v)]/4，r(v)=(p24/p25)·e^(2·ZI·v)。
     冻结前验证（2026-09-05，165A/165B 起跑点=s2/A 24参+[1,3]/[30,10]）：
       · eq44 网格稳定性：−100..80/10、−100..40/20、−100..40/5、−140..100/10 四网格
         中点 −53.2019~−53.2166，散布 ≤0.02mV；
       · 协议仿真 qv_sim（3s@−100→3s@V 阶积分）与 eq44 差 1.2mV，且随沉降时间漂移
         （T=3s:−54.45 → T=10s:−56.28）——病源=s 门有效电压耦合 ve_a=v+p16·s，
         s 缓变使 ∫Ig 含 s 驱动慢电荷，协议输出沉降依赖、不可作冻结口径；
         eq44=解析骨架平衡（与 164 冻结 eq() 同族、仅加 I_c 精确权重），定为 KG③ 口径；
       · reach165 四个全过点在 eq44 下复核 KG 仍全过（QV=−53.21~−53.33、
         左移 36.2~37.0mV）——可达性判决对口径切换稳健。
判窗/锚点/半宽/GATE_FAIL 与 gate164 逐字相同（判线全同 164）。
"""
import numpy as np
from scipy.optimize import curve_fit
from math import comb

DT = 1e-4
# 判窗（预注册冻结，全同 164）
WIN_T0 = (28.1, 44.3); WIN_T60 = (3.7, 4.9)
WIN_QV = (-58.0, -48.0); WIN_Q2 = (0.55, 0.78)
# 锚点与窗半宽（目标函数归一化用，全同 164）
A_T0, H_T0 = 36.2, 8.1
A_T60, H_T60 = 4.3, 0.6
A_QV, H_QV = -53.1, 5.0
A_Q2, H_Q2 = 0.665, 0.115
GATE_FAIL = 1e4   # 仿真/拟合失败固定大惩罚（预注册冻结口径）
ZI = 0.6 / 25.693  # 与 runner_v148 内核逐字一致

# 多初值网格（预注册冻结：18 起点）
P0_GRID = [(a2, t1, t2) for a2 in (0.3, 0.6, 0.85)
           for t1 in (0.002, 0.005) for t2 in (0.02, 0.036, 0.08)]

def envelope_fit(ig_step, dt=DT):
    """envelope 段 Q(t)=∫Ig 双指数拟合（多初值稳健化版）→ (τ1,τ2,a2frac,qinf)；失败 None。
    18 起点同 bounds 各拟合一次，取 SSR 最小者；全部失败返回 None。"""
    q = np.cumsum(ig_step) * dt
    qinf = q[-1]
    if not np.isfinite(qinf) or qinf <= 0.05:
        return None
    qn = q / qinf
    t = np.arange(len(ig_step)) * dt
    def bi(t, a2, t1, t2):
        return (1 - a2) * (1 - np.exp(-t / t1)) + a2 * (1 - np.exp(-t / t2))
    best = None
    for p0 in P0_GRID:
        try:
            popt, _ = curve_fit(bi, t, qn, p0=list(p0),
                                bounds=([0.0, 1e-4, 1e-4], [1.0, 10.0, 10.0]), maxfev=20000)
            ssr = float(np.sum((bi(t, *popt) - qn) ** 2))
            if best is None or ssr < best[0]:
                best = (ssr, popt)
        except Exception:
            continue
    if best is None:
        return None
    a2, t1, t2 = best[1]
    if t1 > t2:
        t1, t2 = t2, t1
    return t1 * 1e3, t2 * 1e3, a2, qinf

def eq44(v, p):
    """model 44 解析平衡平均层位（归一化到 [0,1] 再 /4 由调用方不做——此处直接返回 /4 口径，
    与 gate164.eq 输出同口径）。树拓扑细致平衡：wC_k=comb(4,k)(a/b)^k，
    wO_k=p1·p2^k·wC_k，wI_c=wC_4·r(v)。"""
    a = p[19]*np.exp(p[20]*v); b = p[21]*np.exp(-p[20]*v)
    wC = np.array([comb(4, k)*(a/b)**k for k in range(5)])
    wO = p[1]*np.array([p[2]**k for k in range(5)])*wC
    r = (p[24]/p[25])*np.exp(2.0*ZI*v)
    num = float(np.sum((wC+wO)*np.arange(5))) + 4.0*wC[4]*r
    den = float(np.sum(wC+wO)) + wC[4]*r
    return num/den/4.0

def gate_metrics(R, p):
    """四指标：(τ_Q2(0), τ_Q2(+60), Q–V 中点, Q2 电荷比@0mV)；任一失败返回 None。
    R 为 runner 模块（显式注入，防 import 漂移）。p 为 26 参。"""
    # envelope 0mV 与 +60mV（2s 预平衡；τ(0)=36ms→0.6s=17τ 足量）
    n0 = int(2.0 / DT)
    V0 = np.r_[np.full(n0, -100.0), np.full(int(0.6 / DT), 0.0)]
    ig0 = R.sim_gate148(p, V0, DT, 44)
    if not np.all(np.isfinite(ig0)):
        return None
    r0 = envelope_fit(ig0[n0:])
    V60 = np.r_[np.full(n0, -100.0), np.full(int(0.06 / DT), 60.0)]
    ig60 = R.sim_gate148(p, V60, DT, 44)
    if not np.all(np.isfinite(ig60)):
        return None
    r60 = envelope_fit(ig60[n0:])
    if r0 is None or r60 is None:
        return None
    # Q–V 解析平衡（model 44 树拓扑扩展式）
    vs = np.arange(-100, 81, 10.0)
    qs = np.array([eq44(v, p) for v in vs])
    try:
        popt, _ = curve_fit(lambda v, vh, k: 1.0/(1.0+np.exp(-(v-vh)/k)), vs, qs,
                            p0=[-55.0, 15.0])
        vh = popt[0]
    except Exception:
        return None
    return r0[1], r60[1], vh, r0[2]   # τ_Q2(0), τ_Q2(+60), QV中点, Q2比

def gate_pen(R, p):
    """目标函数门控项：Σ ((指标−锚)/窗半宽)²；失败返回 GATE_FAIL。"""
    m = gate_metrics(R, p)
    if m is None:
        return GATE_FAIL
    t0, t60, vh, q2 = m
    return ((t0-A_T0)/H_T0)**2 + ((t60-A_T60)/H_T60)**2 \
         + ((vh-A_QV)/H_QV)**2 + ((q2-A_Q2)/H_Q2)**2

def kg_verdict(m, gv_mid=None):
    """KG 五条款判定（m=gate_metrics 输出；gv_mid=模型 G–V 中点，判官侧传入）。
    返回 (过/灭, 明细串)。KG⑤ 含"较 G–V 左移≥30mV"（预注册冻结口径，全同 164）。"""
    t0, t60, vh, q2 = m
    ok = (WIN_T0[0] <= t0 <= WIN_T0[1]) and (WIN_T60[0] <= t60 <= WIN_T60[1]) \
         and (WIN_QV[0] <= vh <= WIN_QV[1]) and (WIN_Q2[0] <= q2 <= WIN_Q2[1])
    det = (f"τ_Q2(0)={t0:.2f}∈[28.1,44.3]{'✓' if WIN_T0[0]<=t0<=WIN_T0[1] else '✗'} "
           f"τ_Q2(+60)={t60:.2f}∈[3.7,4.9]{'✓' if WIN_T60[0]<=t60<=WIN_T60[1] else '✗'} "
           f"QV={vh:.2f}∈[−58,−48]{'✓' if WIN_QV[0]<=vh<=WIN_QV[1] else '✗'} "
           f"Q2比={q2:.3f}∈[0.55,0.78]{'✓' if WIN_Q2[0]<=q2<=WIN_Q2[1] else '✗'}")
    if gv_mid is not None:
        shift = gv_mid - vh   # Q–V 比 G–V 更负=左移量
        ok3 = shift >= 30.0
        ok = ok and ok3
        det += f" 左移={shift:.1f}mV≥30{'✓' if ok3 else '✗'}(GV中点={gv_mid:.2f})"
    return ok, det
