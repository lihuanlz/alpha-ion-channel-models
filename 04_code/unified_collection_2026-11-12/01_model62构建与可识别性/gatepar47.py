# -*- coding: utf-8 -*-
"""gatepar47.py：model 47 门控并行后端（代码167，2026-09-08）。
镜像 gatepar.py（纪律①②③逐字沿用：同 P0_GRID/bounds/maxfev/ssr/首优保序、pool=None 回退串行、
依赖注入 R），唯一差异：sim_gate148 的 mid=47（κ(s) 走本通道，KG 时标分离保护见预注册 §1）。
eq44 直接复用 gate165（κ 不动平衡比，解析平衡式零改动——预注册 §1）。
"""
import numpy as np
from scipy.optimize import curve_fit
import gate165 as G
from gatepar import envelope_fit_par, _env_one   # 进程池工人与并行 envelope 原样复用

MID47 = 47


def gate_metrics_par47(R, p, pool):
    """gatepar.gate_metrics_par 的 mid=47 版：四指标 (τ_Q2(0), τ_Q2(+60), QV中点, Q2比)；失败 None。"""
    n0 = int(2.0 / G.DT)
    V0 = np.r_[np.full(n0, -100.0), np.full(int(0.6 / G.DT), 0.0)]
    ig0 = R.sim_gate148(p, V0, G.DT, MID47)
    if not np.all(np.isfinite(ig0)):
        return None
    r0 = envelope_fit_par(ig0[n0:], pool)
    V60 = np.r_[np.full(n0, -100.0), np.full(int(0.06 / G.DT), 60.0)]
    ig60 = R.sim_gate148(p, V60, G.DT, MID47)
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


def gate_pen_par47(R, p, pool):
    """并行后端门控惩罚（mid=47）；失败返回 gate165.GATE_FAIL。"""
    m = gate_metrics_par47(R, p, pool)
    if m is None:
        return G.GATE_FAIL
    t0, t60, vh, q2 = m
    return ((t0 - G.A_T0) / G.H_T0) ** 2 + ((t60 - G.A_T60) / G.H_T60) ** 2 \
         + ((vh - G.A_QV) / G.H_QV) ** 2 + ((q2 - G.A_Q2) / G.H_Q2) ** 2


def gate_metrics47(R, p):
    """串行版（判官/锚注册用；数值与并行版逐位同值——pool=None 回退路径）。"""
    return gate_metrics_par47(R, p, None)


def gate_pen47(R, p):
    m = gate_metrics47(R, p)
    if m is None:
        return G.GATE_FAIL
    t0, t60, vh, q2 = m
    return ((t0 - G.A_T0) / G.H_T0) ** 2 + ((t60 - G.A_T60) / G.H_T60) ** 2 \
         + ((vh - G.A_QV) / G.H_QV) ** 2 + ((q2 - G.A_Q2) / G.H_Q2) ** 2
