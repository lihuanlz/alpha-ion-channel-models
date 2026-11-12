# de173.py — 代码173 model 53 冻结版四臂重锚卡（预注册 rev A1 执行件，本地机 30 维拟合器）
# 用法：python de173.py [de_workers] [gate_workers] [card_path]
# model 53 = model 50 的四臂重修版（007 线收官裁决 2026-09-09 §四-3）：
#   臂1 数据口径：协议电压后移 1 采样点（官方 MexHH.c L101 "double shift = 0.1" 逐字对齐——
#        实验实际施加协议比设计协议晚 0.1ms，仿真协议时间边界 +shift）+ 全协议 |ΔV|>10mV
#        跳变后剔 50 点（5ms 容性窗，对齐官方 B2.1/objective_without_capacitance.m 8 处×50 点
#        口径；官方只拟合 sine 故只剔 8 处，我方四协议联合拟合故全协议剔——口径差异声明在案）
#   臂2 海绵封口：EK 钉死 −88.4mV（官方 Nernst 钉值，外审实锤官方 F11 无 E_K 自由度）、
#        g_L 钉死 0.0007μS（raw 保持段 −53.2pA/−80mV 独立实测；SHIFT 闭案外审盖章）
#   臂3 冻结内核：s≡1 硬编码（s 模块四参 p14/15/17/18 删除）、β 删除（kap≡1，c7e=c7）、
#        ve 删除（ve_a≡v，p16 删除）——间歇性复活池整体关停（再审视报告 J4 裁定）
#   臂4 判线复核：封口改写前后判线全量复算（anchor173 基线注册，判线冻结口径沿用 rev A3 §2）
# 优化坐标 30 维 = 38 − 6（s 池）− 1（EK）− 1（g_L）。物理向量仍 37 槽（冻结槽填常量，
#   内核忽略），X 深井六参冻结继承照旧，SHIFT=+0.1318nA 挂 sa 照旧（数据制备税闭案项）。
# 自由槽 FREE53 = [0..13, 19..30, 32..35]（14+12+4=30）。
# 纪律：判线与目标函数分离；判官 judge173.py 独立全量复算；锚未注册拒跑。
import json, math, os, sys, time
import io as _io
import contextlib as _cl
import numpy as np
from numba import njit

HERE173 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE173)
import runner_v148 as R
import run_all165 as RA
import gate165
import gatepar47
from runner166_m46 import u_dyn, DT
from runner166b_m46 import u_anchor_pen
import de169
from de169 import (SHIFT, W_TAIL, LAM, P6X, GX_MID, GX_K, evolve_x,
                   find_step, halfrise, creep, tail_rise, _ssv, steady_ratios,
                   escape_check32, tail_fit_kernel,
                   TAIL_WIN, TAIL_SSE_MAX, KCT_TAIL_WIN, KCT_AMIN, KCT_T0NOM,
                   M1_WIN, M2_WIN, M3_WIN, HRS1_M5_MAX, HOLD_TARGET, HOLD_WIDTH)

MODEL_NAME = "model53"

# ---- 封口常量（臂2，预注册 rev A1 §3 冻结）----
EK_FROZEN = -88.4             # mV，官方 Nernst 钉值（PMC5978315 Methods 逐字）
GL_FROZEN = 7.0e-4            # μS，raw 保持段独立实测（判别实验 2026-09-09：−53.2pA@−80mV→R≈1.5GΩ）

# ---- 冻结槽位（臂3）：s 模块四参 + β + ve ----
FROZEN_SLOTS = (14, 15, 16, 17, 18, 31)
FREE53 = [i for i in range(32) if i not in FROZEN_SLOTS] + [32, 33, 34, 35]   # 30 槽
LG53 = [i for i in (0, 1, 2, 4, 6, 7, 8, 10, 19, 21, 22, 23, 24, 25, 26)]     # 15 个 log10 槽
LG53_EXTRA = [33, 35]                                                        # 直边两 log 槽
assert len(FREE53) == 30

# ---- 冻结内核 sim53_trace：traj167 源码 + 臂3 补丁（s≡1/kap≡1/ve_a≡v）----
_src53 = open(os.path.join(HERE173, 'traj167_轨迹仪器化_2026-09-08.py'), encoding='utf-8') \
    .read().replace('cache=True', 'cache=False')
assert _src53.count('(0.6 / 25.693)') == 4, '价数硬编码点应为 4 处（fi7/bi7/fic/bic）'
_src53 = _src53.replace('(0.6 / 25.693)', '(p[32])')
# 补丁 A1：s 初值 → 常量 1
_S_INIT = "    s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0"
assert _src53.count(_S_INIT) == 1, 's 初值行未唯一命中'
_src53 = _src53.replace(_S_INIT, "    s = 1.0   # model53 臂3：s≡1 硬编码")
# 补丁 A2：循环内 s 演化四行+守卫 → 仅 ve_a=v
_S_DYN = """        if abs(p[17]) < 1.0:
            out[i, 0] = np.nan
            continue
        tau_v = p[14] * np.exp(p[18] * v)
        if tau_v < 1.0:
            out[i, 0] = np.nan
            continue
        s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
        s += dt * (s_inf - s) / tau_v
        ve_a = v + p[16] * s"""
assert _src53.count(_S_DYN) == 1, 's 演化块未唯一命中'
_src53 = _src53.replace(_S_DYN, "        ve_a = v   # model53 臂3：ve 删除")
# 补丁 A3：kap≡1（β 删除）
_KAP = "        kap = np.exp(-p[31] * s)"
assert _src53.count(_KAP) == 1, 'kap 行未唯一命中'
_src53 = _src53.replace(_KAP, "        kap = 1.0   # model53 臂3：β 删除")
# 补丁 B：C4⇌Ic 后注入 Ic⇌O4 直边（S3′，de169 补丁3 逐字沿用）
_ANCHOR53 = """            Tc = x7[8] + xiC
            x7[8] = Tc * frc + (x7[8] - Tc * frc) * eic
            xiC = Tc - x7[8]"""
_INJECT53 = _ANCHOR53 + """
        fio = p[33] * np.exp(p[34] * v) if len(p) > 35 else 0.0
        bio = p[35] * np.exp(-p[34] * v) if len(p) > 35 else 0.0
        lamo = fio + bio
        if lamo > 0.0:
            eio = np.exp(-lamo * dt)
            fro2 = bio / lamo
            To = x7[9] + xiC
            x7[9] = To * fro2 + (x7[9] - To * fro2) * eio
            xiC = To - x7[9]"""
assert _src53.count(_ANCHOR53) == 1, 'C4⇌Ic 锚点不唯一'
_src53 = _src53.replace(_ANCHOR53, _INJECT53)
_HARD167_53 = "'/mnt/agents/output/04_细胞线4/结果/代码167_垂直时标卡_2026-09-08/work/counter_167_K6a9Jub.json'"
assert _src53.count(_HARD167_53) == 1, 'traj167 顶层 167 判决点硬路径未找到'
_src53 = _src53.replace(_HARD167_53, "os.path.join(HERE, 'counter_167_K6a9Jub.json')")
_ns53 = {'__name__': 'traj53', '__file__': os.path.join(HERE173, 'traj167_轨迹仪器化_2026-09-08.py')}
with _cl.redirect_stdout(_io.StringIO()):
    exec(compile(_src53, 'traj53_src', 'exec'), _ns53)
sim53_trace = _ns53['sim47_trace']      # out(n,7)：[I, s≡1, o7, Ic, Isum, r1, r2]
try:
    os.remove(os.path.join(HERE173, 'traj167_deact_seg1.npz'))
except OSError:
    pass


# ---- model 53 整机装配（sim50 模板逐字：核迹线×u + X 电流 + 漏电；EK/g_L 封口钉值）----
def sim53(p, V, dt):
    """p 长度 37（冻结槽被内核忽略）。EK=EK_FROZEN、g_L=GL_FROZEN 钉死（臂2）。
    输出 = 冻结内核电流 × u(v) + g·ALPHA·x·gX·(v−EK)（X 冻结）+ g_L·(v−EK)。"""
    p = np.asarray(p, float)
    tr = sim53_trace(p, V, dt, EK_FROZEN)
    m = tr[:, 0]
    if not np.all(np.isfinite(m)):
        return m
    u = u_dyn(p, V, dt)
    xs = evolve_x(V, dt, P6X[0], P6X[1], P6X[2], P6X[3], P6X[4])
    gx = 1.0 / (1.0 + np.exp((V - GX_MID) / GX_K))
    return m * u + p[7] * P6X[5] * xs * gx * (V - EK_FROZEN) + GL_FROZEN * (V - EK_FROZEN)


# ---- 数据层（臂1）：时移 + 容性窗 + sa 伪影窗 + SHIFT ----
def shift_protocol_1pt(v):
    """协议电压后移 1 采样点（0.1ms），对齐官方 MexHH.c 的 +shift 口径：
    仿真协议 = 设计协议在 (t−0.1ms) 的值 ⇔ v_new[i]=v_orig[i−1]，v_new[0]=v_orig[0]。"""
    out = np.empty_like(v)
    out[0] = v[0]
    out[1:] = v[:-1]
    return out


def load_keep173(protos):
    """173 数据口径：load_cell 后电压后移 1 点；keep = sa 伪影窗（165 逐字）
    + 方波跳变后 50 点（5ms 容性窗，dV=10mV 检测，make_keep_mask 口径）。
    ap 协议特判（rev A1 §2 双录）：只剔 prepulse 段（t<0.5s，0.25/0.30s 两处方波）——
    ap 波形本体上升沿 0.1ms 采样 |ΔV|>10mV 属生理快速除极而非钳位跳变，
    官方 B2.1 剔窗仅针对 step-changes in the imposed voltage clamp，连续波形不剔。"""
    data_raw = R.load_cell("16704007", protos)
    data = {}
    for k, (v, c) in data_raw.items():
        vs = shift_protocol_1pt(np.asarray(v, float))
        n = min(len(vs), len(c))
        data[k] = (vs[:n], np.asarray(c, float)[:n])
    keep = {}
    for k, (v, c) in data.items():
        if k == "ap":
            km = np.ones(len(v), dtype=bool)
            dv = np.abs(np.diff(v))
            jumps = np.nonzero(dv > 10.0)[0]
            for j in jumps:
                if j * DT < 0.5:                 # 仅 prepulse 段方波
                    km[j + 1:j + 1 + 50] = False
            keep[k] = km
        else:
            keep[k] = R.make_keep_mask(v, 50)   # 跳变后 50 点（官方 5ms×10kHz 口径）
    tt = np.arange(len(data["steady_activation"][0])) * DT
    keep["steady_activation"][(tt >= 50.18) & (tt < 55.18)] = False   # sa +60mV 伪影窗照旧
    return data, keep


# ---- 优化坐标（30 维；冻结槽钉常量不进 x）----
def unpack53(x):
    """30 维优化坐标 → 37 维物理向量（冻结槽常量）。EK 恒 EK_FROZEN。"""
    x = np.asarray(x, float)
    assert len(x) == 30
    p = np.zeros(37)
    for j, i in enumerate(FREE53):
        p[i] = (10.0 ** x[j]) if (i in LG53 or i in LG53_EXTRA) else x[j]
    # 冻结槽填充（内核忽略，仅保持 37 槽兼容判官/工具件）
    p[14], p[15], p[16], p[17], p[18], p[31] = 16.0467, -52.056, 0.0, 17.52951, -0.037041, 0.0
    p[36] = GL_FROZEN
    return p, EK_FROZEN


def x_from_params53(p37):
    x = np.zeros(30)
    for j, i in enumerate(FREE53):
        x[j] = np.log10(p37[i]) if (i in LG53 or i in LG53_EXTRA) else p37[i]
    return x


def walls53(x):
    """30 维结构墙（walls50 逐字口径删冻结槽）。"""
    x = np.asarray(x, float)
    if not (RA.XC_LO <= x[0] <= RA.XC_HI):
        return False
    for j, i in enumerate(FREE53):
        v = (10.0 ** x[j]) if (i in LG53 or i in LG53_EXTRA) else x[j]
        if i in (22, 23, 24, 25) and x[j] < RA.XK_LO:
            return False
        if i == 26 and not (0.05 <= v <= 200.0):
            return False
        if i == 27 and not (-0.05 <= v <= 0.05):
            return False
        if i == 28 and not (1.0 <= v <= 50.0):
            return False
        if i == 29 and not (-80.0 <= v <= 40.0):
            return False
        if i == 30 and not (4.0 <= v <= 40.0):
            return False
        if i == 32 and not (0.005 <= v <= 0.12):
            return False
        if i == 33 and not (0.1 <= v <= 300.0):
            return False
        if i == 34 and not (-0.10 <= v <= 0.05):
            return False
        if i == 35 and not (0.05 <= v <= 100.0):
            return False
    return True


# ---- DE 界盒（de_bounds50 逐字口径删冻结槽；起跑点必须在盒内）----
def de_bounds53(x0):
    b38 = [(0.0, 0.0)] * 38
    for i in [1, 2, 4, 6, 7, 8, 10, 19, 21]:
        b38[i] = (-4.0, 4.0)
    b38[1] = (-6.0, 4.0)   # L7 槽下界放宽（rev A3 双录⑦沿用）
    b38[0] = (-2.0, 6.0)
    for i in (22, 23, 24, 25):
        b38[i] = (RA.XK_LO, 4.0)
    b38[26] = (math.log10(0.05), math.log10(200.0))
    for i in (3, 5, 9, 11):
        b38[i] = (-0.3, 0.3)
    b38[12] = (-35.0, 35.0); b38[13] = (-1.0, 1.0)
    b38[20] = (0.001, 0.1)
    b38[27] = (-0.05, 0.05); b38[28] = (1.0, 50.0)
    b38[29] = (-80.0, 40.0); b38[30] = (4.0, 40.0)
    b38[32] = (0.005, 0.12)
    b38[33] = (-1.0, math.log10(300.0))
    b38[34] = (-0.10, 0.05)
    b38[35] = (math.log10(0.05), 2.0)
    b = [b38[i] for i in FREE53]
    for j, (lo, hi) in enumerate(b):
        assert lo <= x0[j] <= hi, f"DE 盒不含起跑点 @自由槽{j}（物理槽{FREE53[j]}：{x0[j]}∉[{lo},{hi}]）"
    return b


NPOP53 = 180          # popsize=6 × 30 维（预注册 rev A1 §6 冻结）
DE_SEED53 = 20260910  # 预注册 rev A1 §6 冻结


def init_pop53(x0, bounds):
    """180 种群（de169 init_pop50 口径映射 30 维）：个体0=起跑点保底种子；
    A 组 90=逐槽抖动（log 槽 ±0.1dex / 线性槽 ×(1+10%)）；
    B 组 90=候选方向播种（L7×0.04/直边开 邻域对数抖动）——g_L 槽已钉死不再播种。"""
    rng = np.random.default_rng(DE_SEED53)
    lo = np.array([bb[0] for bb in bounds]); hi = np.array([bb[1] for bb in bounds])
    pop = np.empty((NPOP53, 30))
    pop[0] = x0
    j1 = FREE53.index(1); j33 = FREE53.index(33); j34 = FREE53.index(34); j35 = FREE53.index(35)
    for j in range(1, NPOP53):
        x = x0.copy()
        if j > NPOP53 // 2:
            x[j1] = math.log10(10.0 ** x0[j1] * 0.04) + rng.normal(0, 0.3)
            x[j33] = math.log10(3.0) + rng.normal(0, 0.5)
            x[j34] = -0.03 + rng.normal(0, 0.01)
            x[j35] = math.log10(1.0) + rng.normal(0, 0.5)
        for jj in range(30):
            if j > NPOP53 // 2 and jj in (j1, j33, j34, j35):
                continue
            i = FREE53[jj]
            if i in LG53 or i in LG53_EXTRA:
                x[jj] = x[jj] + rng.normal(0, 0.1)
            else:
                x[jj] = x[jj] * (1.0 + rng.normal(0, 0.1))
        pop[j] = np.clip(x, lo, hi)
    return pop


# ---- 残差与目标函数（de169 resid/obj 逐字口径，sim53/unpack53/walls53 替换）----
def _sse_proto53(args):
    p, v, c, dt = args
    try:
        m = sim53(p, v, DT if dt is None else dt)
    except (ZeroDivisionError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(m)):
        return None
    return m - c


def resid53_pair(p, data, keep, decim, serial):
    """返回 (rd, r_deact_raw)。非有限→None。"""
    prots = list(data.items())
    if serial or len(prots) <= 1:
        rs = [_sse_proto53((p, np.asarray(v, float), np.asarray(c, float), None))
              for pr, (v, c) in prots]
    else:
        rs = list(R._tpe().map(_sse_proto53,
                               [(p, np.asarray(v, float), np.asarray(c, float), None)
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


def sse_of53(p, data, keep=None, decim=None, serial=True):
    out = resid53_pair(p, data, keep, decim, serial)
    if out is None:
        return 1e30
    rd, _ = out
    tot = 0.0
    for pr, r in rd.items():
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def obj53_factory(gpool, serial, wins):
    """OBJ = sse_full(keep173 口径, sa+SHIFT) + w_tail·sse_tail + λ·窗项（de169 逐字）。"""
    ISH, IPK, IHD, IWT, HOLD_C = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT'], wins['HOLD_C']

    def obj(x, data, keep, lam, decim, model):
        assert model == MODEL_NAME
        x = np.asarray(x, float)
        if not walls53(x):
            return 1e12
        p, ek = unpack53(x)
        out = resid53_pair(p, data, keep, decim, serial)
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


# ---- 冻结锚（预注册 §6+纪律：anchor173 沙箱注册 2026-09-09 填入；None=拒跑）----
J0_ANCHOR = 1198468.446320     # anchor173 注册：169 点@173 口径 J0
SSE0_ANCHOR = 217823.006518    # anchor173 注册：169 点@173 口径 sse_full（时移+容性窗+封口+冻结）


def gv_midpoint53(p):
    """宏观 G–V 中点（gv_midpoint50 口径，sim53 替换）。"""
    from scipy.optimize import curve_fit
    vs = np.arange(-90, 51, 10.0)
    n_pre = int(2.0 / DT); n_hold = int(4.0 / DT); n_tl = int(0.3 / DT)
    pk = []
    for v in vs:
        vp = np.r_[np.full(n_pre, -100.0), np.full(n_hold, v), np.full(n_tl, -120.0)]
        m = sim53(p, vp, DT)
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


# ---- 主流程（镜像 fit169 骨架；单段 DE(dec5,150it) → NM(full,3000it)）----
WALL_BUDGET = 4.0 * 3600.0
WALL_STOP = WALL_BUDGET * 1.18


def fit173(card_path, de_workers, gate_workers):
    if J0_ANCHOR is None or SSE0_ANCHOR is None:
        print("[173] J0/SSE0 锚未注册（常量 None），按纪律拒跑——先在沙箱注册", flush=True)
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
    assert float(card.get("EK_start", EK_FROZEN)) == EK_FROZEN, "173 卡 EK 钉死 −88.4，卡面不符拒跑"
    data, keep = load_keep173(card["protos"])
    vsa, csa = data["steady_activation"]
    vd, cd = data["deactivation"]
    data2 = dict(data); data2["steady_activation"] = (vsa, np.asarray(csa, float) + SHIFT)
    # 窗索引（时移后协议现场定跳回时刻；判官同式锚）
    t0 = find_step(np.asarray(vd, float), KCT_T0NOM)
    ISH = (int(0.415 / DT), int(0.460 / DT))
    IPK = (int(round((t0 + 0.002) / DT)), int(round((t0 + 0.080) / DT)))
    IHD = (int(0.48 / DT), int(0.60 / DT))
    IWT = (int(round((t0 + TAIL_WIN[0]) / DT)), int(round((t0 + TAIL_WIN[1]) / DT)))
    HOLD_C = float(np.mean(np.asarray(cd, float)[IHD[0]:IHD[1]]))
    wins = dict(ISH=ISH, IPK=IPK, IHD=IHD, IWT=IWT, HOLD_C=HOLD_C)
    print(f"[173] 窗索引：t0={t0:.4f}s IPK={IPK} ISH={ISH} IHD={IHD} IWT={IWT} "
          f"数据保持流={HOLD_C:.4f}nA（臂1 时移后协议）", flush=True)
    x0 = x_from_params53(p_start)
    decim = card.get("decimate", None)
    de_maxiter = int(card.get("de_maxiter", 150))
    nm_maxiter = int(card.get("nm_maxiter", 3000))
    tt0 = time.time()
    print(f"[173] {card.get('tag')} de_workers={de_workers} gate_workers={gate_workers} "
          f"DE(dec5,{de_maxiter}it,NPOP={NPOP53})→NM(full,{nm_maxiter}it) seed={DE_SEED53}", flush=True)

    gpool = ProcessPoolExecutor(max_workers=gate_workers) if gate_workers > 1 else None
    obj_de = obj53_factory(gpool, True, wins)
    obj_nm = obj53_factory(gpool, False, wins)

    # —— 校验锚（裸调用，不计 nfev）——
    J0 = obj_nm(x0, data2, keep, lam, None, MODEL_NAME)
    sse0 = sse_of53(p_start, data2, keep=keep, serial=False)
    print(f"[173] 校验锚：sse0={sse0:.4f}（锚 {SSE0_ANCHOR}）、J0={J0:.4f}（锚 {J0_ANCHOR}）",
          flush=True)
    if abs(sse0 - SSE0_ANCHOR) > 1e-6 * SSE0_ANCHOR or abs(J0 - J0_ANCHOR) > 1e-3:
        print("[173] 锚未中，拒跑（内核/坐标一致性事故，原样记录）", flush=True)
        gpool and gpool.shutdown()
        return

    bounds = de_bounds53(x0)
    pop = init_pop53(x0, bounds)
    tp = ThreadPoolExecutor(de_workers)
    nfev_de = [0]
    state = {"gen": [0]}

    def f_de(x):
        nfev_de[0] += 1
        return obj_de(x, data2, keep, lam, decim, MODEL_NAME)

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        print(f"  [173] DE gen {state['gen'][0]} done ({el:.0f}s)", flush=True)
        if el > WALL_BUDGET:
            print(f"  [173] wall 超预算 {WALL_BUDGET / 3600:.1f}h（DE 段不可优雅中断，双录在案）",
                  flush=True)

    rde = differential_evolution(
        f_de, bounds, maxiter=de_maxiter, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED53, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    print(f"[173] DE 完：J={rde.fun:.1f} nfev={rde.nfev} ({time.time() - tt0:.0f}s) "
          f"msg={rde.message}", flush=True)
    de_rec = {"maxiter": de_maxiter, "npop": NPOP53, "seed": DE_SEED53, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message)}
    tp.shutdown()
    if rde.fun >= 1e11:
        print("[173] DE 末代最佳 J≥1e11（全墙）——早停条款判死记录，不进 NM", flush=True)
        _dump_verdict53(card, lam, data2, keep, np.asarray(rde.x, float), nfev_de[0],
                        de_rec, None, "de_allwall|" + str(rde.message), tt0, stage="de_allwall")
        gpool and gpool.shutdown()
        return
    _dump_verdict53(card, lam, data2, keep, np.asarray(rde.x, float), nfev_de[0],
                    de_rec, None, "de_only", tt0, stage="de_only")

    nfev_nm = [0]; best = {"f": None, "x": np.asarray(rde.x, float)}
    state2 = {"it": 0, "fchk": [None, 0]}

    class _EarlyStop(Exception):
        pass

    def cb_nm(xk):
        state2["it"] += 1
        if state2["it"] % 200 == 0:
            print(f"  [173] NM iter {state2['it']} bestJ={best['f']:.2f} "
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
            print(f"  [173] wall 超 {WALL_STOP / 3600:.2f}h（预算 4h+18%），自动停并双录", flush=True)
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
    print(f"[173] NM 完：bestJ={best['f']:.2f} nfev={nfev_nm[0]} ({time.time() - tt0:.0f}s) {msg}",
          flush=True)
    gpool and gpool.shutdown()

    _dump_verdict53(card, lam, data2, keep, np.asarray(x_stage, float),
                    nfev_de[0] + nfev_nm[0], de_rec, nm_rec, msg, tt0, stage="final")


def _dump_verdict53(card, lam, data2, keep, x_stage, nfev, de_rec, nm_rec, msg, tt0,
                    stage="final"):
    """判官（判线 M1-M6 全冻结沿用 rev A3 §2 口径；逃逸检查删冻结槽条目）。
    quick（stage≠final）：只落快账。"""
    quick = (stage != "final")
    data, _ = load_keep173(card["protos"])
    vsa, csa_raw = data["steady_activation"]
    vd, cd = data["deactivation"]
    p, ek = unpack53(x_stage)
    sse_full = sse_of53(p, data2, keep=keep, serial=False)
    t0 = find_step(np.asarray(vd, float), KCT_T0NOM)
    ISH = (int(0.415 / DT), int(0.460 / DT))
    IPK = (int(round((t0 + 0.002) / DT)), int(round((t0 + 0.080) / DT)))
    IHD = (int(0.48 / DT), int(0.60 / DT))
    IWT = (int(round((t0 + TAIL_WIN[0]) / DT)), int(round((t0 + TAIL_WIN[1]) / DT)))
    n1 = int(9.1612 / DT)
    m1 = sim53(p, np.asarray(vd[:n1], float), DT)
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
    msa = sim53(p, np.asarray(vsa, float), DT)
    NOM = [17.15, 25.40, 33.66, 41.92]; SKIP = [0.6, 0.6, 0.05, 0.05]
    hrs = [halfrise(msa, find_step(np.asarray(vsa, float), t), s) for t, s in zip(NOM, SKIP)]
    crs = [creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60, mi40, mi00, mi60 = steady_ratios(msa)
    kc2_now = [bool(hrs[1] <= 1653.6), bool(hrs[0] <= 2499.9),
               bool(90.0 <= trise <= 275.0), bool(crs[0] > 0 and crs[1] > 0)]
    ks_now = [bool(0.05 <= r40 <= 0.30), bool(r60 <= 0.8 * r40)]
    if quick:
        decomp53, ap_ok, gm, gvm = {}, None, None, None
        kg_ok, kg_det = None, "quick中间件未复算"
    else:
        decomp53 = {}
        for pr in card["protos"]:
            v_, c_ = data2[pr]
            m_ = sim53(p, np.asarray(v_, float), DT)
            r_ = (m_ - np.asarray(c_, float))[keep[pr]]
            decomp53[pr] = float(r_ @ r_)
        ap_ok = bool(decomp53["ap"] <= 5290.0)
        gm = gatepar47.gate_metrics47(R, p[:32])
        gvm = gv_midpoint53(p)
        kg_ok, kg_det = (False, "门控仿真None→KG灭") if gm is None else gate165.kg_verdict(gm, gvm)
    if quick:
        new_fails, hrs1_ok, M5 = None, None, None
    else:
        d167 = json.load(open(os.path.join(HERE173, 'counter_167_K6a9Jub.json'), encoding='utf-8'))
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
    # —— M6 逃逸+平脊（逃逸检查只查自由槽；EK/g_L/s 池钉死不在逃） ——
    esc = escape_check32(p[:32])
    for nm, val, lo, hi in [("tau0u", p[26], 0.05, 200), ("z_tau", p[27], -0.05, 0.05),
                            ("M", p[28], 1.0, 50), ("vh_u", p[29], -80, 40),
                            ("k_u", p[30], 4, 40),
                            ("p32", p[32], 0.005, 0.12), ("p33", p[33], 0.1, 300),
                            ("p34", p[34], -0.10, 0.05), ("p35", p[35], 0.05, 100)]:
        if val <= lo * 1.001 or val >= hi * 0.999:
            esc.append(f"槽 {nm}={val:.3g} 贴/越界")
    p0cmp = np.array(card["start_params"], float)
    rel = np.abs(p - p0cmp) / np.maximum(np.abs(p0cmp), 1e-12)
    rel_free = np.array([rel[i] for i in FREE53])   # 漂移统计只算自由槽（冻结槽 rel=0 不干扰）
    if quick:
        improve, flat, M6 = None, None, None
        ok_all, branch = None, stage
        o80 = amp_m = amp_d = bio_p50 = fro2_p50 = None
    else:
        sse0c = sse_of53(p0cmp, data2, keep=keep, serial=False)
        improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
        flat = bool(rel_free.max() > 0.5 and improve < 0.05)
        M6 = bool(len(esc) == 0 and not flat)
        ok_all = bool(M1 and M2 and M3 and M4 and M5 and M6)
        if ok_all:
            branch = "A"
        elif M5 and M3 and M4 and (not M1 or not M2):
            branch = "B"
        elif not M5:
            branch = "C"
        else:
            branch = "未分类（原样记录）"
        tr1 = sim53_trace(p, np.asarray(vd[:n1], float), DT, ek)
        o80 = float(np.mean(tr1[int(0.40 / DT):ISH[0], 2])) if np.all(np.isfinite(tr1)) else np.nan
        i40 = int(round(find_step(np.asarray(vsa, float), 41.92) / DT))
        amp_m = float(np.mean(msa[i40 + 40000:i40 + 50000]))
        amp_d = float(np.mean(np.asarray(csa_raw, float)[i40 + 40000:i40 + 50000]))
        bio_p50 = float(p[35] * math.exp(-p[34] * 50.0))
        fio_p50 = float(p[33] * math.exp(p[34] * 50.0))
        fro2_p50 = bio_p50 / (fio_p50 + bio_p50) if (fio_p50 + bio_p50) > 0 else 0.0
    # —— 173 预言对账（预注册 rev A1 §4 阈值；锚注册时基线填入） ——
    preds = None if quick else {
        "Q1_sse_full低于基线x0.9": [sse_full, Q1_SSE_BASE,
                                    None if Q1_SSE_BASE is None else bool(sse_full < 0.9 * Q1_SSE_BASE)],
        "Q2_判线过数不低于基线": [int(sum([bool(M1), bool(M2), bool(M3), bool(M4), bool(M5), bool(M6)])),
                                  Q2_LINE_BASE,
                                  None if Q2_LINE_BASE is None else
                                  bool(sum([bool(M1), bool(M2), bool(M3), bool(M4), bool(M5), bool(M6)])
                                       >= Q2_LINE_BASE)],
        "Q3_机制参数漂移榜top5": sorted(
            [(f"p{FREE53[j]}", float(rel_free[j])) for j in range(30)],
            key=lambda t: -t[1])[:5],
        "Q4_M3保持读数": hold,
        "Q5_直边活跃读数": [float(p[33]), bio_p50, fro2_p50],
        "Q6_o80读数": o80,
        "Q7_sa40_amp读数": [amp_m, amp_d],
    }
    verdict = {
        "tag": "173", "model": MODEL_NAME, "stage": stage,
        "params": list(map(float, p)), "EK_used": ek, "gL_frozen": GL_FROZEN,
        "x6_frozen": list(map(float, P6X)),
        "sse_full": sse_full, "sse_tail": sse_tail,
        "J_obj_final": None if nm_rec is None else nm_rec.get("J_best"),
        "decomp": decomp53,
        "M1_peak_nA": pk, "M2_step_nA": sh, "M3_hold_nA": hold,
        "M4_tau_s": tau_fit, "M4_a_nA": a_fit,
        "M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6,
        "M5_new_fails": new_fails, "hrs1_within_5pct": hrs1_ok,
        "half": hrs, "creep": crs, "trise": trise, "r40": r40, "r60": r60,
        "KC2": kc2_now, "KS": ks_now, "KG": bool(kg_ok), "kg_detail": kg_det,
        "gate": None if gm is None else list(map(float, gm)), "gv_midpoint": gvm,
        "KC_U_ap5290": ap_ok, "escape": esc, "flat": flat,
        "improve_vs_start": improve, "drift_max_pct_vs_start": float(rel_free.max()) * 100,
        "predictions": preds,
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "ok_all": ok_all,
        "J0_anchor": J0_ANCHOR, "sse0_anchor": SSE0_ANCHOR,
    }
    out = os.path.join(HERE173, card.get("out_json", "counter_173_model53.json"))
    json.dump(verdict, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[173] 落盘 {out}（stage={stage}）", flush=True)
    if quick:
        print(f"[173] 中间件快账：sse_full={sse_full:.4f} 峰={pk:.3f} 短步={sh:.3f} "
              f"保持={hold:.4f} sse_tail={sse_tail:.3f}（判官全件留待 final）", flush=True)
        return
    print(f"[173] sse_full={sse_full:.4f} sse_tail={sse_tail:.3f} nfev={nfev} "
          f"wall={time.time() - tt0:.0f}s", flush=True)
    print(f"[173] M1峰={pk:.3f}∈{M1_WIN}:{'过' if M1 else '灭'} "
          f"M2短步={sh:.3f}∈{M2_WIN}:{'过' if M2 else '灭'} "
          f"M3保持={hold:.4f}∈{M3_WIN}:{'过' if M3 else '灭'}", flush=True)
    print(f"[173] M4尾流:sse_tail={sse_tail:.2f}≤87∧τ={tau_fit}∈[1.0,2.5]∧|a|={a_fit}≥0.08:"
          f"{'过' if M4 else '灭'}", flush=True)
    print(f"[173] M5老判线:{'过' if M5 else '灭'}（新增灭={new_fails} hrs[1]={hrs[1]:.1f}"
          f"≤{HRS1_M5_MAX:.1f}:{hrs1_ok}） M6:{'过' if M6 else '灭'}（逃逸={esc} flat={flat}）",
          flush=True)
    print(f"[173] 判决布尔：{'全过（分支A）' if ok_all else '未全过'} 分支={branch}", flush=True)


# ---- 173 预言基线常量（anchor173 注册 2026-09-09 填入）----
Q1_SSE_BASE = 217823.006518   # 预言 Q1：终点 sse_full < 0.9×基线=196040.7（30 维重拟合赎回口径损失）
Q2_LINE_BASE = 3              # 预言 Q2：终点判线过数 ≥3（基线 M2/M5/M6 过）


def main():
    de_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    gate_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    card_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE173, "job_173_model53.json")
    fit173(card_path, de_workers, gate_workers)


if __name__ == "__main__":
    main()
