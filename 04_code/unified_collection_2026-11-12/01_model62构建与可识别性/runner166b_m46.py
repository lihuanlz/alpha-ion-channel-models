# runner166b_m46.py — 代码166b（R2 锚定版）：τ_u(−120) 锚升格进目标函数，k_u 下界抬高
# 与 166 的全部差异（预注册_代码166b §3 已声明）：
#   ① 目标函数加 u 锚罚项 λ·((τ_u(−120)−1.59)/0.5)²（τ 锚从判线退位进目标，防双重计数——顾问修正①）；
#   ② walls：k_u 下界 1.0→4.0（参数侧防阶梯逃逸；判线侧独立检查=KC-U③，顾问修正②）；
#   ③ 起点 u 五槽重置于守锚点（τ_u(−120)=1.59s 恰中锚心、罚项为零）。
# 其余（内核/sse 口径/判窗/λ=1e4）逐字沿用 runner166_m46，一字未动。
import numpy as np
import sys, os
sys.path.insert(0, '/mnt/agents/output/04_细胞线4/结果/代码165_失活恢复拓扑卡_2026-09-05/work')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runner166_m46 import (R, RA, sim46, sse_of46, unpack46, x_from_params46, u_dyn,
                           MODEL_NAME, DT, XC_LO, XC_HI, XK_LO)

TAU120_ANCHOR = 1.59   # 数据锚（补记53 §1 直测）
TAU120_WIDTH = 0.5     # 罚项宽度（预注册冻结）

def walls46b(xi):
    """walls46 唯一改动：k_u∈[4,40]。"""
    if not (XC_LO <= xi[0] <= XC_HI):
        return False
    for i in (22, 23, 24, 25):
        if xi[i] < XK_LO:
            return False
    p26 = 10.0 ** xi[26]
    if not (0.05 <= p26 <= 200.0):
        return False
    if not (-0.05 <= xi[27] <= 0.05):
        return False
    if not (1.0 <= xi[28] <= 50.0):
        return False
    if not (-80.0 <= xi[29] <= 40.0):
        return False
    if not (4.0 <= xi[30] <= 40.0):
        return False
    return True

def u_anchor_pen(p):
    """τ_u(−120mV) 锚罚项（无量纲平方，与门控罚同待遇）。"""
    tau120 = p[26] * np.exp(p[27] * (-120.0))
    return ((tau120 - TAU120_ANCHOR) / TAU120_WIDTH) ** 2

def obj_par_factory46b(gpool):
    """sse + λ·(门控罚 + u 锚罚)。"""
    import gatepar
    def obj_par(x, data, keep, lam, decim, model):
        assert model == MODEL_NAME
        if not walls46b(x[:-1]):
            return 1e12
        p = unpack46(x[:-1])
        si = sse_of46(p, data, keep=keep, ek=x[-1], decim=decim)
        if not np.isfinite(si):
            return 1e12
        return si + lam * (gatepar.gate_pen_par(R, p, gpool) + u_anchor_pen(p))
    return obj_par
