# de169.py — 代码169 model 50 结构重修卡（预注册 rev A2 §6 执行件，本地机 38 维拟合器）
# 用法：python de169.py [de_workers] [gate_workers] [card_path]
# model 50 = 167 核（32 参全解冻）+ S1 漏电 g_L·(v−EK) + S2 失活价数 p32 释放
#            + S3′ Ic⇌O4 恢复直边（p33/p34/p35）+ S4 链平衡解冻 + X 深井 168 值冻结继承 + EK
# 优化坐标 38 维：x[0:32]=核（LG48 log10 槽，逐字沿用 runner167_m48），x[32]=p32 线性，
#   x[33]=log10 p33，x[34]=p34 线性，x[35]=log10 p35，x[36]=log10 g_L，x[37]=EK。
# 物理向量 p 长度 37（内核 len(p)>35 守卫触发直边；p[36]=g_L 不进内核、在 sim50 整机装配层加）。
# 判线与目标函数分离（老规矩）；判官全件在 _dump_verdict50 独立全量复算。
# 依赖（同包浅层文件夹）：runner_v148.py / run_all165.py / gate165.py / gatepar.py /
#   gatepar47.py / runner166_m46.py / runner166b_m46.py / traj167_轨迹仪器化_2026-09-08.py /
#   counter_167_K6a9Jub.json / counter_168_K6a9JubX.json / data/cipa/007.npz / job_169 卡 json。
import json, math, os, sys, time
import io as _io
import contextlib as _cl
import numpy as np
from numba import njit

HERE169 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE169)
import runner_v148 as R
import run_all165 as RA
import gate165
import gatepar47
from runner166_m46 import u_dyn, DT
from runner166b_m46 import u_anchor_pen

MODEL_NAME = "model50"
SHIFT = 0.1318          # nA（补记49 基线平移差 −131.8 pA 的反号修正，de167 逐字沿用）
W_TAIL = 4.0            # 尾流保护权重（rev A2 §3 冻结，护 168 战果）
LAM = 1e4               # 窗项/罚项统一权重（λ=1e4 体系沿用 167）

# ---- 判线函数内联（逐字拷贝 counter165_offset.py，免拖 engine_v2 依赖链进包）----
def find_step(v, t_nom):
    i0 = int((t_nom - 0.05) / DT); i1 = int((t_nom + 0.05) / DT)
    j = int(np.argmax(np.abs(np.diff(v[i0:i1])))); return t_nom - 0.05 + (j + 1) * DT

def halfrise(tr, t0, skip):
    i0 = int(round(t0 / DT)); seg = tr[i0:i0 + 50000]
    lo = np.mean(seg[int(skip / DT):int((skip + 0.1) / DT)]); hi = np.mean(seg[40000:])
    thr = lo + 0.5 * (hi - lo); isk = int(skip / DT)
    idx = np.nonzero(seg[isk:] >= thr)[0]
    return (idx[0] + isk) * DT * 1e3 if len(idx) else np.nan

def creep(tr, t0):
    i0 = int(round(t0 / DT))
    mid_ = np.mean(tr[i0 + 20000:i0 + 30000]); end = np.mean(tr[i0 + 40000:i0 + 50000])
    return (end / mid_ - 1) * 100 if abs(mid_) > 1e-9 else np.nan

def tail_rise(tr, t40):
    i0 = int(round((t40 + 5.0) / DT)); seg = tr[i0:i0 + int(0.5 / DT)]
    return float(np.argmax(np.abs(seg)) * DT * 1e3)

def _ssv(tr, t0, t1):
    return float(tr[int(t0 / DT):int(t1 / DT)].mean())

def steady_ratios(tr):
    i40 = _ssv(tr, 45.919, 46.919); i00 = _ssv(tr, 29.403, 30.403); i60 = _ssv(tr, 54.177, 55.177)
    return i40 / i00, i60 / i00, i40, i00, i60

def escape_check32(p):
    """结构逃逸监测（逐字拷贝 judge165/counter165_offset，作用核 32 参）。"""
    esc = []
    if not (0.01 <= p[0] <= 1e6): esc.append(f"c={p[0]:.3g} 越界")
    elif p[0] >= 0.999e6: esc.append(f"c={p[0]:.3g} 顶天花板（c→∞ 逃逸未遂）")
    elif p[0] <= 0.0101: esc.append(f"c={p[0]:.3g} 贴地板（c→0 逃逸未遂）")
    if p[22] < 0.5: esc.append(f"KI={p[22]:.3g} 破地板（KI→0 逃逸）")
    elif p[22] <= 0.505: esc.append(f"KI={p[22]:.3g} 贴地板（KI→0 逃逸未遂）")
    if p[23] < 0.5: esc.append(f"KB={p[23]:.3g} 破地板（KB→0 逃逸）")
    elif p[23] <= 0.505: esc.append(f"KB={p[23]:.3g} 贴地板（KB→0 逃逸未遂）")
    if p[24] < 0.5: esc.append(f"KIC={p[24]:.3g} 破地板（KIC→0 逃逸）")
    elif p[24] <= 0.505: esc.append(f"KIC={p[24]:.3g} 贴地板（KIC→0 逃逸未遂）")
    if p[25] < 0.5: esc.append(f"KBC={p[25]:.3g} 破地板（KBC→0 逃逸）")
    elif p[25] <= 0.505: esc.append(f"KBC={p[25]:.3g} 贴地板（KBC→0 逃逸未遂）")
    return esc

# ---- X 深井：168 拟合值冻结继承（预注册 §1；evolve_x 内联，不 import de168——
#      de168 模块级 CORE_JSON 是沙箱绝对路径，Windows 本地机 import 即炸，教训在案）----
@njit(cache=True, fastmath=True)
def evolve_x(V, dt, XMAX, VH, KX, TAU0, ZX):
    n = len(V); xs = np.empty(n)
    X = XMAX / (1.0 + np.exp(-(V[0] - VH) / KX))
    for i in range(n):
        v = V[i]
        xi = XMAX / (1.0 + np.exp(-(v - VH) / KX))
        tv = TAU0 * np.exp(ZX * v)
        if tv < 0.05:
            xs[i] = np.nan; continue
        X = xi + (X - xi) * np.exp(-dt / tv)
        xs[i] = X
    return xs

GX_MID, GX_K = -60.0, 10.0     # X 门控（de168 冻结常量逐字）

_C168 = json.load(open(os.path.join(HERE169, 'counter_168_K6a9JubX.json'), encoding='utf-8'))
P6X = np.asarray(_C168['params6'], float)   # [XMAX,VH,K,TAU0,Z,ALPHA]
assert len(P6X) == 6

# ---- 内核：traj167 源码补丁版 exec（probe169/scan169b/scan169c 已验证路径逐字沿用）----
#   补丁1：numba cache=True→False（exec 动态模块无 locator，缓存反序列化必炸——工程教训在案）
#   补丁2：失活价数硬编码 (0.6 / 25.693)×4 处（fi7/bi7/fic/bic）→ (p[32])（S2）
#   补丁3：C4⇌Ic 弛豫块后锚点注入 Ic⇌O4 直边精确弛豫子步（S3′；scan169b 模板逐字）
#   补丁4：traj167 顶层读 167 判决点的 /mnt 沙箱硬路径 → HERE 相对（Windows 本地机兼容）
_src = open(os.path.join(HERE169, 'traj167_轨迹仪器化_2026-09-08.py'), encoding='utf-8') \
    .read().replace('cache=True', 'cache=False')
assert _src.count('(0.6 / 25.693)') == 4, '价数硬编码点应为 4 处（fi7/bi7/fic/bic）'
_src = _src.replace('(0.6 / 25.693)', '(p[32])')
_ANCHOR = """            Tc = x7[8] + xiC
            x7[8] = Tc * frc + (x7[8] - Tc * frc) * eic
            xiC = Tc - x7[8]"""
_INJECT = _ANCHOR + """
        fio = p[33] * np.exp(p[34] * v) if len(p) > 35 else 0.0
        bio = p[35] * np.exp(-p[34] * v) if len(p) > 35 else 0.0
        lamo = fio + bio
        if lamo > 0.0:
            eio = np.exp(-lamo * dt)
            fro2 = bio / lamo
            To = x7[9] + xiC
            x7[9] = To * fro2 + (x7[9] - To * fro2) * eio
            xiC = To - x7[9]"""
assert _src.count(_ANCHOR) == 1, 'C4⇌Ic 锚点不唯一'
_src = _src.replace(_ANCHOR, _INJECT)
_HARD167 = "'/mnt/agents/output/04_细胞线4/结果/代码167_垂直时标卡_2026-09-08/work/counter_167_K6a9Jub.json'"
assert _src.count(_HARD167) == 1, 'traj167 顶层 167 判决点硬路径未找到'
_src = _src.replace(_HARD167, "os.path.join(HERE, 'counter_167_K6a9Jub.json')")
_ns = {'__name__': 'traj50', '__file__': os.path.join(HERE169, 'traj167_轨迹仪器化_2026-09-08.py')}
with _cl.redirect_stdout(_io.StringIO()):
    exec(compile(_src, 'traj50_src', 'exec'), _ns)
sim50_trace = _ns['sim47_trace']      # out(n,7)：[I, s, o7, Ic, Isum, r1, r2]
# traj167 顶层分析副作用落盘（仪器化脚本固有成瘾），进包运行即清
try:
    os.remove(os.path.join(HERE169, 'traj167_deact_seg1.npz'))
except OSError:
    pass

# ---- model 50 整机装配（scan169c 已验证模板逐字：核迹线×u + X 电流 + 漏电）----
def sim50(p, V, dt, ek):
    """p 长度 37：[0:32]=167 核，p[32]=失活价数，p[33..35]=Ic⇌O4 直边，p[36]=g_L。
    输出 = 内核47补丁版电流 × u(v) + g·ALPHA·x·gX·(v−EK)（X 冻结）+ g_L·(v−EK)（S1，E_L=EK 双录）。"""
    p = np.asarray(p, float)
    tr = sim50_trace(p, V, dt, ek)
    m = tr[:, 0]
    if not np.all(np.isfinite(m)):
        return m
    u = u_dyn(p, V, dt)
    xs = evolve_x(V, dt, P6X[0], P6X[1], P6X[2], P6X[3], P6X[4])
    gx = 1.0 / (1.0 + np.exp((V - GX_MID) / GX_K))
    return m * u + p[7] * P6X[5] * xs * gx * (V - ek) + p[36] * (V - ek)


def _sse_proto50(args):
    p, v, c, ek, dt = args
    try:
        m = sim50(p, v, DT if dt is None else dt, ek)
    except (ZeroDivisionError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(m)):
        return None
    return m - c


def resid_of50(p, data, keep=None, ek=None, decim=None, serial=True):
    """四协议残差字典（keep 掩码+decim 抽稀口径逐字镜像 sse_of48；serial=DE 段协议串行防超订）。
    返回 None=非有限（OBJ 返 1e12）；窗项在 OBJ 层从 deact 残差现场取（r 在手，免二次模拟）。"""
    prots = list(data.items())
    if serial or len(prots) <= 1:
        rs = [_sse_proto50((p, np.asarray(v, float), np.asarray(c, float), ek, None))
              for pr, (v, c) in prots]
    else:
        rs = list(R._tpe().map(_sse_proto50,
                               [(p, np.asarray(v, float), np.asarray(c, float), ek, None)
                                for pr, (v, c) in prots]))
    out = {}
    for (proto, (v, c)), r in zip(prots, rs):
        if r is None:
            return None
        if keep is not None and proto in keep:
            r = r[keep[proto]]
        r = r[np.isfinite(r)]
        if decim is not None and proto in decim:
            ds = int(decim[proto])
            r = r[::ds] * np.sqrt(ds)
        out[proto] = r
    return out


def sse_of50(p, data, keep=None, ek=None, decim=None, serial=True):
    rd = resid_of50(p, data, keep=keep, ek=ek, decim=decim, serial=serial)
    if rd is None:
        return 1e30
    tot = 0.0
    for pr, r in rd.items():
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


# ---- 优化坐标（镜像 48 布局 + 新槽）----
LG48 = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21, 22, 23, 24, 25, 26]   # log10 槽（β=p31 线性）
LG50_EXTRA = [33, 35, 36]          # p33/p35/g_L 走 log10；p32/p34/EK 线性
XC_LO, XC_HI, XK_LO = RA.XC_LO, RA.XC_HI, RA.XK_LO


def unpack50(x):
    """38 维优化坐标 → 37 维物理向量 + EK。"""
    x = np.asarray(x, float)
    p = np.zeros(37)
    p[LG48] = 10.0 ** x[LG48]
    ln32 = [i for i in range(32) if i not in LG48]
    p[ln32] = x[ln32]
    p[32] = x[32]                          # p32 失活价数（线性）
    p[33] = 10.0 ** x[33]                  # p33 O4→Ic 指前
    p[34] = x[34]                          # p34 直边电压斜率（线性）
    p[35] = 10.0 ** x[35]                  # p35 Ic→O4 指前
    p[36] = 10.0 ** x[36]                  # g_L
    return p, float(x[37])


def x_from_params50(p37, ek):
    x = np.zeros(38)
    p37 = np.asarray(p37, float)
    x[:32] = p37[:32]
    x[LG48] = np.log10(p37[LG48])
    x[32] = p37[32]; x[33] = np.log10(p37[33]); x[34] = p37[34]
    x[35] = np.log10(p37[35]); x[36] = np.log10(p37[36]); x[37] = ek
    return x


def walls50(x):
    """38 维结构墙（walls48 口径逐字 + 新槽物理界）。出界→OBJ 返 1e12。"""
    if not (XC_LO <= x[0] <= XC_HI):
        return False
    for i in (22, 23, 24, 25):
        if x[i] < XK_LO:
            return False
    if not (0.05 <= 10.0 ** x[26] <= 200.0):
        return False
    if not (-0.05 <= x[27] <= 0.05):
        return False
    if not (1.0 <= x[28] <= 50.0):
        return False
    if not (-80.0 <= x[29] <= 40.0):
        return False
    if not (4.0 <= x[30] <= 40.0):
        return False
    if not (0.0 <= x[31] <= 5.0):
        return False
    if not (0.005 <= x[32] <= 0.12):           # p32 失活价数
        return False
    if not (0.1 <= 10.0 ** x[33] <= 300.0):    # p33
        return False
    if not (-0.10 <= x[34] <= 0.05):           # p34
        return False
    if not (0.05 <= 10.0 ** x[35] <= 100.0):   # p35
        return False
    if not (1e-4 <= 10.0 ** x[36] <= 0.03):    # g_L
        return False
    if not (-95.0 <= x[37] <= -75.0):          # EK
        return False
    return True


# ---- DE 界盒（全物理界盒，de167 de_bounds48 口径 + 新槽；起跑点必须在盒内）----
def de_bounds50(x0):
    b = [(0.0, 0.0)] * 38
    for i in [1, 2, 4, 6, 7, 8, 10, 19, 21]:
        b[i] = (-4.0, 4.0)
    b[1] = (-6.0, 4.0)   # L7 槽下界放宽（rev A3 双录⑦：起点 L7×0.04=6.4e-5→log10=−4.19，
                         # 167 旧盒 −4 装不下；walls48 本无 L7 下界，仅 DE 探索盒）
    b[0] = (-2.0, 6.0)
    b[14] = (math.log10(0.3), math.log10(2000.0))   # τ_s 指前上界放宽（rev A3 双录⑦：167 判决点
                                                    # p14=393.9 系当年 NM 无界段合法走出旧盒 200；
                                                    # walls48 本无此槽界，仅 DE 探索盒含起点）
    for i in (22, 23, 24, 25):
        b[i] = (XK_LO, 4.0)
    b[26] = (math.log10(0.05), math.log10(200.0))
    for i in (3, 5, 9, 11):
        b[i] = (-0.3, 0.3)
    b[12] = (-35.0, 35.0); b[13] = (-1.0, 1.0)
    b[15] = (-140.0, 0.0); b[16] = (0.0, 80.0)
    b[17] = (1.0, 20.0)
    b[18] = (-0.1, 0.0); b[20] = (0.001, 0.1)
    b[27] = (-0.05, 0.05); b[28] = (1.0, 50.0)
    b[29] = (-80.0, 40.0); b[30] = (4.0, 40.0)
    b[31] = (0.0, 5.0)
    b[32] = (0.005, 0.12)
    b[33] = (-1.0, math.log10(300.0))
    b[34] = (-0.10, 0.05)
    b[35] = (math.log10(0.05), 2.0)
    b[36] = (-4.0, math.log10(0.03))
    b[37] = (-95.0, -75.0)
    for i, (lo, hi) in enumerate(b):
        assert lo <= x0[i] <= hi, f"DE 盒不含起跑点 @槽{i}（{x0[i]}∉[{lo},{hi}]）"
    return b


NPOP = 228          # popsize=6 × 38 维（预注册 §6 冻结）
DE_SEED = 20260909  # 预注册 §6 冻结


def init_pop50(x0, bounds):
    """228 种群（预注册 §6+rev A2 变更⑤）：个体0=起跑点保底种子；
    A 组 114=起跑点逐槽抖动（log 槽 ±0.1dex / 线性槽 ×(1+10%)，de167 同款）；
    B 组 114=手工候选 A 方向播种（L7×0.04/直边开/g_L=0.00313 邻域对数抖动）。"""
    rng = np.random.default_rng(DE_SEED)
    lo = np.array([bb[0] for bb in bounds]); hi = np.array([bb[1] for bb in bounds])
    pop = np.empty((NPOP, 38))
    pop[0] = x0
    for j in range(1, NPOP):
        x = x0.copy()
        if j > NPOP // 2:
            # B 组：候选 A 方向（rev A2 变更④⑤：L7×0.04、直边 3/−0.03/1、g_L=0.00313 邻域）
            x[1] = math.log10(10.0 ** x0[1] * 0.04) + rng.normal(0, 0.3)
            x[33] = math.log10(3.0) + rng.normal(0, 0.5)
            x[34] = -0.03 + rng.normal(0, 0.01)
            x[35] = math.log10(1.0) + rng.normal(0, 0.5)
            x[36] = math.log10(0.00313) + rng.normal(0, 0.3)
        for i in range(38):
            if j > NPOP // 2 and i in (1, 33, 34, 35, 36):
                continue                       # B 组候选槽已置位，不再覆盖
            if i == 31:
                x[i] = rng.uniform(0.0, 3.0) if j <= NPOP // 2 else x[i]
            elif i == 37:
                x[i] = x0[i] + rng.normal(0, 0.5)
            elif i in LG48 or i in LG50_EXTRA:
                if not (j > NPOP // 2 and i in (33, 35, 36)):
                    x[i] = x[i] + rng.normal(0, 0.1)
            elif i in (32, 34):
                if not (j > NPOP // 2 and i == 34):
                    x[i] = x[i] * (1.0 + rng.normal(0, 0.1))
            else:
                x[i] = x[i] * (1.0 + rng.normal(0, 0.1))
        pop[j] = np.clip(x, lo, hi)
    return pop

# ---- 目标函数（rev A2 §3 冻结文本逐字落码）----
# OBJ = sse_full(keep 口径, sa+SHIFT) + w_tail·sse_tail[0.5,6.0]s
#       + 1e4·[ 短步窗 sse + 跳回峰窗 sse + (保持流均值−0.0134)²/0.005² + u锚(166b) + gate_pen(47) ]
# 窗项从 deact 全分辨率残差现场取（残差在手免二次模拟；mean(M)=mean(C)+mean(r) 恒等式）。
# 探针教训落位（probe169 两轮）：窗项无 1e4 量级权重时优化器无视 45ms 窗（450 点 vs 91k 点段）。
HOLD_TARGET, HOLD_WIDTH = -0.005027, 0.005   # 保持流锚（rev A3 修订：数据 [0.48,0.60]s 窗均值
                                             # 实测 −0.005027nA；原 0.0134 系 diag169 单点 t=0.5s
                                             # 读数误作窗均值——依据数字错误，双录在案）


def resid50_pair(p, data, keep, ek, decim, serial):
    """返回 (rd, r_deact_raw)：rd=各协议抽稀后残差（sse_full 累加用），
    r_deact_raw=deactivation 协议未抽稀全分辨率残差（窗项用）。非有限→None。"""
    prots = list(data.items())
    if serial or len(prots) <= 1:
        rs = [_sse_proto50((p, np.asarray(v, float), np.asarray(c, float), ek, None))
              for pr, (v, c) in prots]
    else:
        rs = list(R._tpe().map(_sse_proto50,
                               [(p, np.asarray(v, float), np.asarray(c, float), ek, None)
                                for pr, (v, c) in prots]))
    rd, raw = {}, None
    for (proto, (v, c)), r in zip(prots, rs):
        if r is None:
            return None
        if keep is not None and proto in keep:
            r = r[keep[proto]]
        r = r[np.isfinite(r)]
        if proto == 'deactivation':
            raw = r
        if decim is not None and proto in decim:
            ds = int(decim[proto])
            r = r[::ds] * np.sqrt(ds)
        rd[proto] = r
    return rd, raw


def obj50_factory(gpool, serial, wins):
    """wins=dict(ISH, IPK, IHD, IWT, HOLD_C)。DE 段 serial=True 防嵌套超订；NM 段协议线程池。"""
    ISH, IPK, IHD, IWT, HOLD_C = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT'], wins['HOLD_C']

    def obj(x, data, keep, lam, decim, model):
        assert model == MODEL_NAME
        x = np.asarray(x, float)
        if not walls50(x):
            return 1e12
        p, ek = unpack50(x)
        out = resid50_pair(p, data, keep, ek, decim, serial)
        if out is None:
            return 1e12
        rd, raw = out
        sse_full = 0.0
        for pr, r in rd.items():
            sse_full += float(r @ r)
        if not np.isfinite(sse_full):
            return 1e12
        sse_tail = float(raw[IWT[0]:IWT[1]] @ raw[IWT[0]:IWT[1]])
        sse_short = float(raw[ISH[0]:ISH[1]] @ raw[ISH[0]:ISH[1]])
        sse_peak = float(raw[IPK[0]:IPK[1]] @ raw[IPK[0]:IPK[1]])
        hold_m = HOLD_C + float(np.mean(raw[IHD[0]:IHD[1]]))
        hold_pen = ((hold_m - HOLD_TARGET) / HOLD_WIDTH) ** 2
        gp = gatepar47.gate_pen_par47(R, p[:32], gpool)
        up = u_anchor_pen(p[:32])
        return sse_full + W_TAIL * sse_tail + lam * (sse_short + sse_peak + hold_pen + up + gp)
    return obj


# ---- 冻结锚（预注册 §6+纪律：沙箱先行注册后填入；None=拒跑）----
J0_ANCHOR = 5301109.519743     # anchor169 沙箱注册 2026-09-08（rev A3 口径；起点=候选A邻域
                               # 开放账单原样在案：L7×0.04 打乱 sa 段，sse0 大属预期，DE 赎回）
SSE0_ANCHOR = 509498.042045    # 起点 sse_full（keep 口径，sa+SHIFT）

# ---- M4 尾流判官（168 KT1/KT2 逐字；tail_fit_kernel 逐字拷贝 de167）----
TAIL_WIN = (0.5, 6.0)          # sse_tail 窗（跳回后秒；de168 WIN 逐字）
TAIL_SSE_MAX = 87.0            # 168 KT1 冻结线
KCT_TAIL_WIN = (1.0, 2.5)      # τ 窗（168 KT2 逐字）
KCT_AMIN = 0.08                # |a| 下限（168 KT2 逐字）
KCT_T0NOM = 2.66               # deact 首段 +50→−120 标称时刻（find_step ±0.05s 窗）
M1_WIN = (-4.45, -3.64)        # M1 跳回峰窗 min∈（数据 −4.042±10%）
M2_WIN = (-0.20, 0.02)         # M2 短步窗 min∈
M3_WIN = (-0.015, 0.005)       # M3 −80 保持流均值∈（rev A3：中心=窗均值实测−0.005027，半宽0.01）
HRS1_M5_MAX = 1937.8 * 1.05    # M5：hrs[1] 继承灭允许恶化 ≤5%（vs 168 判官值）


def tail_fit_kernel(tr, i0, i1, dt):
    """多起点单指数拟合取 SSE 最小；全失败=None（快模语义，de167 逐字）。"""
    from scipy.optimize import curve_fit
    tt = np.arange(i0, i1) * dt
    tt = tt - tt[0]
    y = tr[i0:i1]
    best = None
    for t0_ in (0.1, 0.3, 1.0, 2.5):
        try:
            popt, _ = curve_fit(lambda t, c, a, tau: c + a * np.exp(-t / tau),
                                tt, y, p0=[y[-1], y[0] - y[-1], t0_], maxfev=20000)
            ss = float(np.sum((y - (popt[0] + popt[1] * np.exp(-tt / popt[2]))) ** 2))
            if best is None or ss < best[0]:
                best = (ss, popt[1], popt[2])
        except Exception:
            pass
    return (best[1], best[2]) if best else (None, None)


def gv_midpoint50(p, ek):
    """宏观 G–V 中点（含 u 表观，de167 gv_midpoint48 口径，sim48→sim50）。"""
    from scipy.optimize import curve_fit
    vs = np.arange(-90, 51, 10.0)
    n_pre = int(2.0 / DT); n_hold = int(4.0 / DT); n_tl = int(0.3 / DT)
    pk = []
    for v in vs:
        vp = np.r_[np.full(n_pre, -100.0), np.full(n_hold, v), np.full(n_tl, -120.0)]
        m = sim50(p, vp, DT, ek)
        if not np.all(np.isfinite(m)):
            return None
        pk.append(float(np.max(np.abs(m[n_pre + n_hold:]))))
    pk = np.asarray(pk, float)
    span = pk.max() - pk.min()
    if not np.isfinite(span) or pk.max() <= 0 or span / pk.max() < 0.30:
        return None
    gn = (pk - pk.min()) / span
    try:
        popt, _ = curve_fit(lambda v, vh, k: 1.0 / (1.0 + np.exp(-(v - vh) / k)), vs, gn,
                            p0=[0.0, 10.0], maxfev=8000)
        return float(popt[0])
    except Exception:
        return None

# ---- 主流程（镜像 de167.fit167 骨架；单段 DE(dec5,150it) → NM(full,3000it)，预注册 §6）----
WALL_BUDGET = 4.0 * 3600.0        # 预注册 §6 wall 预算 4h
WALL_STOP = WALL_BUDGET * 1.18    # 超 18% 自动停并双录


def fit169(card_path, de_workers, gate_workers):
    if J0_ANCHOR is None or SSE0_ANCHOR is None:
        print("[169] J0/SSE0 锚未注册（常量 None），按纪律拒跑——先在沙箱注册", flush=True)
        return
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from scipy.optimize import minimize, differential_evolution

    card = json.load(open(card_path, encoding="utf-8"))
    assert card["model"] == MODEL_NAME, f"卡面 model={card['model']} ≠ {MODEL_NAME}"
    lam = float(card["lambda"]); assert lam == 1e4, "λ 冻结 1e4，卡面不符拒跑"
    assert float(card.get("w_tail", 0)) == W_TAIL, "w_tail 冻结 4，卡面不符拒跑"
    p_start = np.array(card["start_params"], float); assert len(p_start) == 37
    if "params6" in card:
        assert np.allclose(np.asarray(card["params6"], float), P6X, rtol=0, atol=1e-9), \
            "卡面 params6 与 counter_168 冻结值不符——X 继承链断裂，拒跑"
    data, keep = RA.load_keep(card["protos"])
    vsa, csa = data["steady_activation"]
    vd, cd = data["deactivation"]
    data2 = dict(data); data2["steady_activation"] = (vsa, np.asarray(csa, float) + SHIFT)
    # 窗索引（判官同式锚：find_step(vd, 2.66) 现场定跳回时刻）
    t0 = find_step(np.asarray(vd, float), KCT_T0NOM)
    ISH = (int(0.415 / DT), int(0.460 / DT))
    IPK = (int(round((t0 + 0.002) / DT)), int(round((t0 + 0.080) / DT)))
    IHD = (int(0.48 / DT), int(0.60 / DT))
    IWT = (int(round((t0 + TAIL_WIN[0]) / DT)), int(round((t0 + TAIL_WIN[1]) / DT)))
    HOLD_C = float(np.mean(np.asarray(cd, float)[IHD[0]:IHD[1]]))
    wins = dict(ISH=ISH, IPK=IPK, IHD=IHD, IWT=IWT, HOLD_C=HOLD_C)
    print(f"[169] 窗索引：t0={t0:.4f}s IPK={IPK} ISH={ISH} IHD={IHD} IWT={IWT} "
          f"数据保持流={HOLD_C:.4f}nA", flush=True)
    ek_start = float(card["EK_start"])
    x0 = x_from_params50(p_start, ek_start)
    decim = card.get("decimate", None)
    de_maxiter = int(card.get("de_maxiter", 150))
    nm_maxiter = int(card.get("nm_maxiter", 3000))
    tt0 = time.time()
    print(f"[169] {card.get('tag')} de_workers={de_workers} gate_workers={gate_workers} "
          f"DE(dec5,{de_maxiter}it,NPOP={NPOP})→NM(full,{nm_maxiter}it) seed={DE_SEED}", flush=True)

    gpool = ProcessPoolExecutor(max_workers=gate_workers) if gate_workers > 1 else None
    obj_de = obj50_factory(gpool, True, wins)
    obj_nm = obj50_factory(gpool, False, wins)

    # —— 校验锚（裸调用，不计 nfev）：169 新目标函数在起点须中注册锚 ——
    J0 = obj_nm(x0, data2, keep, lam, None, MODEL_NAME)
    sse0 = sse_of50(p_start, data2, keep=keep, ek=ek_start, serial=False)
    print(f"[169] 校验锚：sse0={sse0:.4f}（锚 {SSE0_ANCHOR}）、J0={J0:.4f}（锚 {J0_ANCHOR}）",
          flush=True)
    if abs(sse0 - SSE0_ANCHOR) > 1e-6 * SSE0_ANCHOR or abs(J0 - J0_ANCHOR) > 1e-3:
        print("[169] 锚未中，拒跑（内核/坐标一致性事故，原样记录）", flush=True)
        gpool and gpool.shutdown()
        return

    bounds = de_bounds50(x0)
    pop = init_pop50(x0, bounds)
    tp = ThreadPoolExecutor(de_workers)
    nfev_de = [0]
    state = {"gen": [0]}

    def f_de(x):
        nfev_de[0] += 1
        return obj_de(x, data2, keep, lam, decim, MODEL_NAME)

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        print(f"  [169] DE gen {state['gen'][0]} done ({el:.0f}s)", flush=True)
        if el > WALL_BUDGET:
            print(f"  [169] wall 超预算 {WALL_BUDGET / 3600:.1f}h（DE 段不可优雅中断，双录在案）",
                  flush=True)

    rde = differential_evolution(
        f_de, bounds, maxiter=de_maxiter, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    print(f"[169] DE 完：J={rde.fun:.1f} nfev={rde.nfev} ({time.time() - tt0:.0f}s) "
          f"msg={rde.message}", flush=True)
    de_rec = {"maxiter": de_maxiter, "npop": NPOP, "seed": DE_SEED, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message)}
    tp.shutdown()
    if rde.fun >= 1e11:
        print("[169] DE 末代最佳 J≥1e11（全墙）——早停条款判死记录，不进 NM", flush=True)
        _dump_verdict50(card, lam, data2, keep, np.asarray(rde.x, float), nfev_de[0],
                        de_rec, None, "de_allwall|" + str(rde.message), tt0, stage="de_allwall")
        gpool and gpool.shutdown()
        return
    # 防超时丢点（probe169 教训落位）：DE 完先落盘中间件，再进 NM
    _dump_verdict50(card, lam, data2, keep, np.asarray(rde.x, float), nfev_de[0],
                    de_rec, None, "de_only", tt0, stage="de_only")

    # —— NM polish（全数据，早停沿用 167 逻辑 + wall 自动停） ——
    nfev_nm = [0]; best = {"f": None, "x": np.asarray(rde.x, float)}
    state2 = {"it": 0, "fchk": [None, 0]}

    class _EarlyStop(Exception):
        pass

    def cb_nm(xk):
        state2["it"] += 1
        if state2["it"] % 200 == 0:
            print(f"  [169] NM iter {state2['it']} bestJ={best['f']:.2f} "
                  f"({time.time() - tt0:.0f}s)", flush=True)
        if state2["it"] % 1000 == 0:
            f_now = best["f"]
            if state2["fchk"][0] is not None and f_now is not None:
                gain = (state2["fchk"][0] - f_now) / max(abs(state2["fchk"][0]), 1e-12)
                state2["fchk"][1] = state2["fchk"][1] + 1 if gain < 1e-3 else 0
                if state2["fchk"][1] >= 2:
                    raise _EarlyStop
            if f_now is not None:
                state2["fchk"][0] = f_now
        if time.time() - tt0 > WALL_STOP:
            print(f"  [169] wall 超 {WALL_STOP / 3600:.2f}h（预算 4h+18%），自动停并双录", flush=True)
            raise _EarlyStop

    def f_nm(x):
        nfev_nm[0] += 1
        v = obj_nm(x, data2, keep, lam, None, MODEL_NAME)
        if best["f"] is None or v < best["f"]:
            best["f"] = v; best["x"] = np.array(x).copy()
        return v

    msg, stopped = "", False
    try:
        rnm = minimize(f_nm, np.asarray(rde.x, float), method="Nelder-Mead", callback=cb_nm,
                       options={"maxiter": nm_maxiter, "maxfev": nm_maxiter + 200,
                                "xatol": 1e-6, "fatol": 1e-3})
        x_stage, msg = rnm.x, str(rnm.message)
    except _EarlyStop:
        x_stage, msg, stopped = best["x"], "early_stop(nm|wall)", True
    nm_rec = {"maxiter": nm_maxiter, "nfev": nfev_nm[0], "J_best": best["f"],
              "early_stopped": stopped, "message": msg}
    print(f"[169] NM 完：bestJ={best['f']:.2f} nfev={nfev_nm[0]} ({time.time() - tt0:.0f}s) {msg}",
          flush=True)
    gpool and gpool.shutdown()

    _dump_verdict50(card, lam, data2, keep, np.asarray(x_stage, float),
                    nfev_de[0] + nfev_nm[0], de_rec, nm_rec, msg, tt0, stage="final")


def _dump_verdict50(card, lam, data2, keep, x_stage, nfev, de_rec, nm_rec, msg, tt0,
                    stage="final"):
    """判官（判线 M1-M6 全冻结，预注册 §2）。判线与目标函数分离：此处独立全量复算。
    quick（stage≠final 的中间件落盘）：只落快账（防超时丢点条款），判官重件留待 final。"""
    quick = (stage != "final")
    data, _ = RA.load_keep(card["protos"])
    vsa, csa_raw = data["steady_activation"]
    vd, cd = data["deactivation"]
    p, ek = unpack50(x_stage)
    sse_full = sse_of50(p, data2, keep=keep, ek=ek, serial=False)
    t0 = find_step(np.asarray(vd, float), KCT_T0NOM)
    ISH = (int(0.415 / DT), int(0.460 / DT))
    IPK = (int(round((t0 + 0.002) / DT)), int(round((t0 + 0.080) / DT)))
    IHD = (int(0.48 / DT), int(0.60 / DT))
    IWT = (int(round((t0 + TAIL_WIN[0]) / DT)), int(round((t0 + TAIL_WIN[1]) / DT)))
    # —— deact 首段全分辨率整机迹线（M1/M2/M3/M4 判窗） ——
    n1 = int(9.1612 / DT)
    m1 = sim50(p, np.asarray(vd[:n1], float), DT, ek)
    c1 = np.asarray(cd[:n1], float)
    finite1 = np.all(np.isfinite(m1))
    pk = float(np.min(m1[IPK[0]:IPK[1]])) if finite1 else np.nan
    sh = float(np.min(m1[ISH[0]:ISH[1]])) if finite1 else np.nan
    hold = float(np.mean(m1[IHD[0]:IHD[1]])) if finite1 else np.nan
    seg = m1[IWT[0]:IWT[1]] if finite1 else np.full(IWT[1] - IWT[0], np.nan)
    sse_tail = float(np.sum((seg - c1[IWT[0]:IWT[1]]) ** 2)) if finite1 else np.inf
    a_fit, tau_fit = tail_fit_kernel(m1, IWT[0], IWT[1], DT) if finite1 else (None, None)
    M1 = bool(finite1 and M1_WIN[0] <= pk <= M1_WIN[1])
    M2 = bool(finite1 and M2_WIN[0] <= sh <= M2_WIN[1])
    M3 = bool(finite1 and M3_WIN[0] <= hold <= M3_WIN[1])
    M4 = bool(finite1 and sse_tail <= TAIL_SSE_MAX
              and tau_fit is not None and KCT_TAIL_WIN[0] <= tau_fit <= KCT_TAIL_WIN[1]
              and abs(a_fit) >= KCT_AMIN)
    # —— sa 段老判线（M5） ——
    msa = sim50(p, np.asarray(vsa, float), DT, ek)
    NOM = [17.15, 25.40, 33.66, 41.92]; SKIP = [0.6, 0.6, 0.05, 0.05]
    hrs = [halfrise(msa, find_step(np.asarray(vsa, float), t), s) for t, s in zip(NOM, SKIP)]
    crs = [creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60, mi40, mi00, mi60 = steady_ratios(msa)
    kc2_now = [bool(hrs[1] <= 1653.6), bool(hrs[0] <= 2499.9),
               bool(90.0 <= trise <= 275.0), bool(crs[0] > 0 and crs[1] > 0)]
    ks_now = [bool(0.05 <= r40 <= 0.30), bool(r60 <= 0.8 * r40)]
    if quick:
        decomp50, ap_ok, gm, gvm = {}, None, None, None
        kg_ok, kg_det = None, "quick中间件未复算"
    else:
        decomp50 = {}
        for pr in card["protos"]:
            v_, c_ = data2[pr]
            m_ = sim50(p, np.asarray(v_, float), DT, ek)
            r_ = (m_ - np.asarray(c_, float))[keep[pr]]
            decomp50[pr] = float(r_ @ r_)
        ap_ok = bool(decomp50["ap"] <= 5290.0)
        gm = gatepar47.gate_metrics47(R, p[:32])
        gvm = gv_midpoint50(p, ek)
        kg_ok, kg_det = (False, "门控仿真None→KG灭") if gm is None else gate165.kg_verdict(gm, gvm)
    # M5「不新增灭」：逐项对照 167 判决点灭集合（判线冻结口径唯一事实源=counter_167 json）
    if quick:
        new_fails, hrs1_ok, M5 = None, None, None
    else:
        d167 = json.load(open(os.path.join(HERE169, 'counter_167_K6a9Jub.json'), encoding='utf-8'))
        kc2_base, ks_base = list(map(bool, d167["KC2"])), list(map(bool, d167["KS"]))
        kg_base = bool(d167["KG"]); ap_base = bool(d167["decomp"]["ap"] <= 5290.0)
        new_fails = []
        for i in range(4):
            if (not kc2_now[i]) and kc2_base[i]:
                new_fails.append(f"KC2[{i}] 新增灭")
        for i in range(2):
            if (not ks_now[i]) and ks_base[i]:
                new_fails.append(f"KS[{i}] 新增灭")
        if (not kg_ok) and kg_base:
            new_fails.append("KG 新增灭")
        if (not ap_ok) and ap_base:
            new_fails.append("KC-U②ap 新增灭")
        hrs1_ok = bool(hrs[1] <= HRS1_M5_MAX)
        M5 = bool(len(new_fails) == 0 and hrs1_ok)
    # —— M6 逃逸+贴界+平脊 ——
    esc = escape_check32(p[:32])
    for nm, val, lo, hi in [("tau0u", p[26], 0.05, 200), ("z_tau", p[27], -0.05, 0.05),
                            ("M", p[28], 1.0, 50), ("vh_u", p[29], -80, 40),
                            ("k_u", p[30], 4, 40), ("beta", p[31], 0.0, 5.0),
                            ("p32", p[32], 0.005, 0.12), ("p33", p[33], 0.1, 300),
                            ("p34", p[34], -0.10, 0.05), ("p35", p[35], 0.05, 100),
                            ("g_L", p[36], 1e-4, 0.03), ("EK", ek, -95, -75)]:
        if val <= lo * 1.001 or val >= hi * 0.999:
            esc.append(f"槽 {nm}={val:.3g} 贴/越界")
    p0cmp = np.array(card["start_params"], float)
    rel = np.abs(p - p0cmp) / np.maximum(np.abs(p0cmp), 1e-12)
    if quick:
        improve, flat, M6 = None, None, None
        ok_all, branch = None, stage
        o80 = amp_m = amp_d = bio_p50 = fro2_p50 = None
    else:
        sse0c = sse_of50(p0cmp, data2, keep=keep, ek=float(card["EK_start"]), serial=False)
        improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
        flat = bool(rel.max() > 0.5 and improve < 0.05)
        M6 = bool(len(esc) == 0 and not flat)
        # —— 判决布尔与分支（预注册 §5） ——
        ok_all = bool(M1 and M2 and M3 and M4 and M5 and M6)
        if ok_all:
            branch = "A"
        elif M5 and M3 and M4 and (not M1 or not M2):
            branch = "B"
        elif not M5:
            branch = "C"
        else:
            branch = "未分类（原样记录）"
        # —— 预言对账字段（§4 P1-P7） ——
        tr1 = sim50_trace(p, np.asarray(vd[:n1], float), DT, ek)
        o80 = float(np.mean(tr1[int(0.40 / DT):ISH[0], 2])) if np.all(np.isfinite(tr1)) else np.nan
        i40 = int(round(find_step(np.asarray(vsa, float), 41.92) / DT))
        amp_m = float(np.mean(msa[i40 + 40000:i40 + 50000]))
        amp_d = float(np.mean(np.asarray(csa_raw, float)[i40 + 40000:i40 + 50000]))
        bio_p50 = float(p[35] * math.exp(-p[34] * 50.0))   # P6 rev A3⑧：装载期 Ic→O4 速率
        fio_p50 = float(p[33] * math.exp(p[34] * 50.0))
        fro2_p50 = bio_p50 / (fio_p50 + bio_p50) if (fio_p50 + bio_p50) > 0 else 0.0
    verdict = {
        "tag": "169", "model": MODEL_NAME, "stage": stage,
        "params": list(map(float, p)), "EK_used": ek,
        "x6_frozen": list(map(float, P6X)),
        "sse_full": sse_full, "sse_tail": sse_tail,
        "J_obj_final": None if nm_rec is None else nm_rec.get("J_best"),
        "decomp": decomp50,
        "M1_peak_nA": pk, "M2_step_nA": sh, "M3_hold_nA": hold,
        "M4_tau_s": tau_fit, "M4_a_nA": a_fit,
        "M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6,
        "M5_new_fails": new_fails, "hrs1_within_5pct": hrs1_ok,
        "half": hrs, "creep": crs, "trise": trise, "r40": r40, "r60": r60,
        "KC2": kc2_now, "KS": ks_now, "KG": bool(kg_ok), "kg_detail": kg_det,
        "gate": None if gm is None else list(map(float, gm)), "gv_midpoint": gvm,
        "KC_U_ap5290": ap_ok, "escape": esc, "flat": flat,
        "improve_vs_start": improve, "drift_max_pct_vs_start": float(rel.max()) * 100,
        "predictions": None if quick else {
            "P1_gL_in_[0.002,0.006]": [float(p[36]), bool(0.002 <= p[36] <= 0.006)],
            "P2_O80<0.05": [o80, bool(np.isfinite(o80) and o80 < 0.05)],
            "P3_p32_dev>20pct": [float(p[32]), float(p[32] / 0.023354 - 1.0) * 100],
            "P4_hrs1_within_5pct": [hrs[1], hrs1_ok],
            "P5_sse_full<23975": [sse_full, bool(sse_full < 23975.0)],
            "P6_edge_active": [float(p[33]), bio_p50, fro2_p50,
                               bool(p[33] > 0.1 and bio_p50 > 1.0 and fro2_p50 > 0.5)],
            "P7_sa40_amp_within_30pct": [amp_m, amp_d,
                                         bool(abs(amp_m - amp_d) <= 0.3 * abs(amp_d))],
        },
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "ok_all": ok_all,
        "J0_anchor": J0_ANCHOR, "sse0_anchor": SSE0_ANCHOR,
    }
    out = os.path.join(HERE169, card.get("out_json", "counter_169_model50.json"))
    json.dump(verdict, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[169] 落盘 {out}（stage={stage}）", flush=True)
    if quick:
        print(f"[169] 中间件快账：sse_full={sse_full:.4f} 峰={pk:.3f} 短步={sh:.3f} "
              f"保持={hold:.4f} sse_tail={sse_tail:.3f}（判官全件留待 final）", flush=True)
        return
    print(f"[169] sse_full={sse_full:.4f} sse_tail={sse_tail:.3f} nfev={nfev} "
          f"wall={time.time() - tt0:.0f}s", flush=True)
    print(f"[169] M1峰={pk:.3f}∈{M1_WIN}:{'过' if M1 else '灭'} "
          f"M2短步={sh:.3f}∈{M2_WIN}:{'过' if M2 else '灭'} "
          f"M3保持={hold:.4f}∈{M3_WIN}:{'过' if M3 else '灭'}", flush=True)
    print(f"[169] M4尾流:sse_tail={sse_tail:.2f}≤87∧τ={tau_fit}∈[1.0,2.5]∧|a|={a_fit}≥0.08:"
          f"{'过' if M4 else '灭'}", flush=True)
    print(f"[169] M5老判线:{'过' if M5 else '灭'}（新增灭={new_fails} hrs[1]={hrs[1]:.1f}"
          f"≤{HRS1_M5_MAX:.1f}:{hrs1_ok}） M6:{'过' if M6 else '灭'}（逃逸={esc} flat={flat}）",
          flush=True)
    print(f"[169] 判决布尔：{'全过（分支A）' if ok_all else '未全过'} 分支={branch}", flush=True)


def main():
    de_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    gate_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    card_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE169, "job_169_model50.json")
    fit169(card_path, de_workers, gate_workers)


if __name__ == "__main__":
    main()
