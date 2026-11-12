# runner166_m46.py — 代码166 慢可用度加边卡：model 46 "K6a9Ju"（31参）
# = model 44（K6a9Ji，26参，冻结不动）+ u 态（p26..p30）。
# 结构原理（预注册 §1）：u 是电压的自治函数 du/dt=(u∞(v)−u)/τ_u(v)，不读马氏态
#   → 不必动内核：I46 = sim_kernel44(v) × u(v)。u≡1 时 IEEE 精确退化（x*1.0=x），
#   J0 锚=165B 终卡点 J=29485.2161 / sse=29460.8144（预注册 §3）。
# 布局：p[0..25] 同 model 44；p26=τ0u(log, s@0mV)、p27=zτ(线性)、p28=M(线性≥1)、
#   p29=vh_u(线性 mV)、p30=k_u(线性 mV，|k|<1→NaN 守卫，镜像 s 守卫纪律)。
# τ_u(v)=p26·e^(p27·v)；u∞(v)=1+(p28−1)·logistic((v−p29)/p30)；u(0)=u∞(V[0])（平衡初态）。
# 守卫：p28<1→NaN；τ_u(v)<0.05→NaN（预注册界 p26∈[0.05,200] 的途中保险）；|p30|<1→NaN。
import numpy as np
from numba import njit
import sys, os
sys.path.insert(0, '/mnt/agents/output/04_细胞线4/结果/代码165_失活恢复拓扑卡_2026-09-05/work')
import runner_v148 as R

MODEL_NAME = "K6a9Ju"
MID44 = 44
DT = 1e-4
EK = -85.0  # 默认；OBJ 由 x[-1] 传入

@njit(cache=True, fastmath=True)
def u_dyn(p, V, dt):
    """u 轨迹（精确弛豫子步，与内核失活/C4⇌Ic 子步同款）。"""
    n = len(V)
    u = np.empty(n)
    if abs(p[30]) < 1.0 or p[28] < 1.0:
        for i in range(n):
            u[i] = np.nan
        return u
    ui = 0.0
    for i in range(n):
        v = V[i]
        tv = p[26] * np.exp(p[27] * v)
        if tv < 0.05:
            for j in range(n):
                u[j] = np.nan
            return u
        uinf = 1.0 + (p[28] - 1.0) / (1.0 + np.exp(-(v - p[29]) / p[30]))
        if i == 0:
            ui = uinf
        else:
            ui = uinf + (ui - uinf) * np.exp(-dt / tv)
        u[i] = ui
    return u

def sim46(p, V, dt, ek):
    """model 46 输出 = 内核 44 输出 × u(v)。p 长度 31（内核只读前 26 槽）。"""
    m = R.sim_kernel(np.asarray(p, float), V, MID44, dt, ek)
    if not np.all(np.isfinite(m)):
        return m
    u = u_dyn(np.asarray(p, float), V, dt)
    return m * u

def _sse_proto46(args):
    p, v, c, ek, dt = args
    try:
        m = sim46(p, v, DT if dt is None else dt, EK if ek is None else ek)
    except (ZeroDivisionError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(m)):
        return None
    return m - c

def sse_of46(p, data, keep=None, ek=None, decim=None, dt=None):
    """逐字镜像 R.sse_of 的累加顺序（协议顺序=data.items()，r@r 串行求和），
    仅把内核换成 sim46。u≡1 时与 R.sse_of(p,data,'K6a9Ji') 逐位同值。"""
    tot = 0.0
    prots = list(data.items())
    def _dt_of(pr):
        return dt.get(pr, None) if isinstance(dt, dict) else dt
    rs = list(R._tpe().map(_sse_proto46, [(p, v, c, ek, _dt_of(pr)) for pr, (v, c) in prots])) \
        if len(prots) > 1 else [_sse_proto46((p, prots[0][1][0], prots[0][1][1], ek, _dt_of(prots[0][0])))]
    for (proto, (v, c)), r in zip(prots, rs):
        if r is None:
            return 1e30
        if keep is not None and proto in keep:
            r = r[keep[proto]]
        r = r[np.isfinite(r)]
        if decim is not None and proto in decim:
            ds = int(decim[proto])
            r = r[::ds] * np.sqrt(ds)
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot

# ---- 优化坐标（镜像 runner_v148 的 K6a9Ji 布局 + 5 新槽） ----
LG44 = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21, 22, 23, 24, 25]
LG46 = LG44 + [26]           # τ0u 恒正→log10 槽

def unpack46(x):
    x = np.asarray(x, float)
    p = np.zeros(31)
    p[LG46] = 10.0 ** x[LG46]
    ln = [i for i in range(31) if i not in LG46]
    p[ln] = x[ln]
    return p

def x_from_params46(p):
    xr = np.asarray(p, float).copy()
    xr[LG46] = np.log10(xr[LG46])
    return xr

# 预注册 §6 参数界（物理口径）：p26∈[0.05,200]s、p27∈[−0.05,0.05]、p28∈[1,50]、
# p29∈[−80,40]mV、p30∈[1,40]mV；结构墙（c/KI/KB/KIC/KBC）逐字沿用 RA.XB 口径。
import run_all165 as RA
XC_LO, XC_HI, XK_LO = RA.XC_LO, RA.XC_HI, RA.XK_LO

def walls46(xi):
    """xi=优化坐标（不含 EK）。结构墙 + u 五槽界。出界→OBJ 返 1e12。"""
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
    if not (1.0 <= xi[30] <= 40.0):
        return False
    return True

def obj_par_factory46(gpool):
    """逐字镜像 engine_v2.obj_par_factory，仅换 sse/unpack/walls。λ 冻结 1e4 由卡面带入。"""
    import gatepar
    def obj_par(x, data, keep, lam, decim, model):
        assert model == MODEL_NAME
        if not walls46(x[:-1]):
            return 1e12
        p = unpack46(x[:-1])
        si = sse_of46(p, data, keep=keep, ek=x[-1], decim=decim)
        if not np.isfinite(si):
            return 1e12
        return si + lam * gatepar.gate_pen_par(R, p, gpool)
    return obj_par
