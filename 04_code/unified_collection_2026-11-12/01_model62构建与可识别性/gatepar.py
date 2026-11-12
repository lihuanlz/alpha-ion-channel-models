# -*- coding: utf-8 -*-
"""gatepar.py：门控包络拟合进程池后端（引擎 v2 组件，2026-09-06）。
纪律：
  ① 18 起点拟合逐位镜像 gate165.envelope_fit——同 P0_GRID、同 bounds([0,1e-4,1e-4],[1,10,10])、
     同 maxfev=20000、同 ssr=np.sum((bi-qn)**2)、同"首优保序"选取（best is None or ssr<best[0]）；
     仅分发方式由串行改进程池，pool.map 保 P0_GRID 顺序 → min-SSR 选择与串行逐位一致。
  ② pool=None 时回退 gate165 串行原函数（零差异保险）。
  ③ 依赖注入 R（runner 模块）防 import 漂移，同 gate165 惯例。
"""
import numpy as np
from scipy.optimize import curve_fit
import gate165 as G


def _env_one(args):
    """进程池工人：单起点双指数拟合。返回 (ssr, a2, t1, t2) 或 None。顶级定义（可 pickle）。"""
    qn, dt, p0 = args
    t = np.arange(len(qn)) * dt
    def bi(t, a2, t1, t2):
        return (1 - a2) * (1 - np.exp(-t / t1)) + a2 * (1 - np.exp(-t / t2))
    try:
        popt, _ = curve_fit(bi, t, qn, p0=list(p0),
                            bounds=([0.0, 1e-4, 1e-4], [1.0, 10.0, 10.0]), maxfev=20000)
        ssr = float(np.sum((bi(t, *popt) - qn) ** 2))
        return (ssr, float(popt[0]), float(popt[1]), float(popt[2]))
    except Exception:
        return None


def envelope_fit_par(ig_step, pool, dt=G.DT):
    """envelope_fit 的进程池版：返回 (τ1,τ2,a2frac,qinf)（ms 口径同 gate165）；失败 None。"""
    if pool is None:
        return G.envelope_fit(ig_step, dt)
    q = np.cumsum(ig_step) * dt
    qinf = q[-1]
    if not np.isfinite(qinf) or qinf <= 0.05:
        return None
    qn = q / qinf
    rs = list(pool.map(_env_one, [(qn, dt, p0) for p0 in G.P0_GRID]))
    best = None
    for r in rs:                       # 按 P0_GRID 原序，首优保序——与串行同语义
        if r is None:
            continue
        if best is None or r[0] < best[0]:
            best = r
    if best is None:
        return None
    a2, t1, t2 = best[1], best[2], best[3]
    if t1 > t2:
        t1, t2 = t2, t1
    return t1 * 1e3, t2 * 1e3, a2, qinf


def gate_metrics_par(R, p, pool):
    """gate165.gate_metrics 的并行后端版：四指标 (τ_Q2(0), τ_Q2(+60), QV中点, Q2比)；失败 None。"""
    n0 = int(2.0 / G.DT)
    V0 = np.r_[np.full(n0, -100.0), np.full(int(0.6 / G.DT), 0.0)]
    ig0 = R.sim_gate148(p, V0, G.DT, 44)
    if not np.all(np.isfinite(ig0)):
        return None
    r0 = envelope_fit_par(ig0[n0:], pool)
    V60 = np.r_[np.full(n0, -100.0), np.full(int(0.06 / G.DT), 60.0)]
    ig60 = R.sim_gate148(p, V60, G.DT, 44)
    if not np.all(np.isfinite(ig60)):
        return None
    r60 = envelope_fit_par(ig60[n0:], pool)
    if r0 is None or r60 is None:
        return None
    vs = np.arange(-100, 81, 10.0)
    qs = np.array([G.eq44(v, p) for v in vs])
    try:
        popt, _ = curve_fit(lambda v, vh, k: 1.0 / (1.0 + np.exp(-(v - vh) / k)), vs, qs,
                            p0=[-55.0, 15.0])
        vh = popt[0]
    except Exception:
        return None
    return r0[1], r60[1], vh, r0[2]


def gate_pen_par(R, p, pool):
    """并行后端门控惩罚；失败返回 gate165.GATE_FAIL。数值与 gate165.gate_pen 逐位一致。"""
    m = gate_metrics_par(R, p, pool)
    if m is None:
        return G.GATE_FAIL
    t0, t60, vh, q2 = m
    return ((t0 - G.A_T0) / G.H_T0) ** 2 + ((t60 - G.A_T60) / G.H_T60) ** 2 \
         + ((vh - G.A_QV) / G.H_QV) ** 2 + ((q2 - G.A_Q2) / G.H_Q2) ** 2
