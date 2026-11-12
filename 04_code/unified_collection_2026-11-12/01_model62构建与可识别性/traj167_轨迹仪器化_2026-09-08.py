# traj167_轨迹仪器化_2026-09-08.py — 167 判决点全内部态轨迹 dump
# sim47_trace：逐字复制 runner_v148 v1.49b 的 model 47 块算法（初态/s更新/Thomas链/失活/Ic/r1r2/输出），
# 仅多返回内部态轨迹（s、o7、Ic、I层Σ、r1、r2、wr4）。u 轨迹由 u_dyn 精确弛豫复刻。
# 用途：整机运行轨迹的机制叙述与病灶定位（用户令：叙述整个运行轨迹+指出问题）。
# 跑法：PYTHONPATH=/mnt/agents/output/.pydeps python3 traj167_轨迹仪器化_2026-09-08.py
import json, os, sys
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/mnt/agents/output/04_细胞线4/结果/代码165_失活恢复拓扑卡_2026-09-05/work')
sys.path.insert(0, '/mnt/agents/output/04_细胞线4/结果/代码166_慢可用度加边卡_2026-09-07/work')
import runner_v148 as R
from runner166_m46 import u_dyn

DT = 1e-4


@njit(cache=True, fastmath=True)
def sim47_trace(p, V, dt, ek):
    """model 47 块的逐字仪器化复刻。返回 out(n,7)：[I, s, o7, Ic, Isum, r1, r2]；wr4 另行重算。"""
    n = len(V)
    out = np.empty((n, 7))
    x7 = np.zeros(10)
    A00 = np.empty(5); A01 = np.empty(5); A10 = np.empty(5); A11 = np.empty(5)
    U00 = np.empty(5); U11 = np.empty(5)
    CP00 = np.zeros(5); CP01 = np.zeros(5); CP10 = np.zeros(5); CP11 = np.zeros(5)
    DP0 = np.zeros(5); DP1 = np.zeros(5)
    xi5 = np.zeros(5)
    xiC = 0.0
    r1, r2 = 1.0, 1.0
    s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0
    x7[0] = 1.0
    for i in range(n):
        v = V[i]
        if abs(p[17]) < 1.0:
            out[i, 0] = np.nan
            continue
        tau_v = p[14] * np.exp(p[18] * v)
        if tau_v < 1.0:
            out[i, 0] = np.nan
            continue
        s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
        s += dt * (s_inf - s) / tau_v
        ve_a = v + p[16] * s
        aa7 = p[19] * np.exp(p[20] * ve_a)
        bb7 = p[21] * np.exp(-p[20] * ve_a)
        c7 = p[0]; L7 = p[1]; th7 = p[2]
        kap = np.exp(-p[31] * s)
        c7e = c7 * kap
        thp = 1.0
        for k in range(5):
            fk7 = (4.0 - k) * aa7
            if k == 4:
                fk7 = 0.0
            bk7 = k * bb7
            ok7 = c7e * L7 * thp
            A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
            A01[k] = -dt * c7e
            A10[k] = -dt * ok7
            A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7e)
            U00[k] = -dt * fk7
            U11[k] = -dt * th7 * fk7
            thp *= th7
        det = A00[0] * A11[0] - A01[0] * A10[0]
        i00 = A11[0] / det; i01 = -A01[0] / det
        i10 = -A10[0] / det; i11 = A00[0] / det
        sup = -dt * bb7
        CP00[0] = i00 * sup; CP01[0] = i01 * sup
        CP10[0] = i10 * sup; CP11[0] = i11 * sup
        DP0[0] = i00 * x7[0] + i01 * x7[1]
        DP1[0] = i10 * x7[0] + i11 * x7[1]
        for k in range(1, 5):
            s0u = U00[k - 1]; s1u = U11[k - 1]
            m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
            m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
            det = m00 * m11 - m01 * m10
            i00 = m11 / det; i01 = -m01 / det
            i10 = -m10 / det; i11 = m00 / det
            sup = -dt * ((k + 1) * bb7)
            CP00[k] = i00 * sup; CP01[k] = i01 * sup
            CP10[k] = i10 * sup; CP11[k] = i11 * sup
            rr0 = x7[2 * k] - s0u * DP0[k - 1]
            rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
            DP0[k] = i00 * rr0 + i01 * rr1
            DP1[k] = i10 * rr0 + i11 * rr1
        x7[8] = DP0[4]; x7[9] = DP1[4]
        for k in range(3, -1, -1):
            x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
            x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
        fi7 = p[22] * np.exp((0.6 / 25.693) * v)
        bi7 = p[23] * np.exp(-(0.6 / 25.693) * v)
        lam7 = fi7 + bi7
        if lam7 > 0.0:
            ei7 = np.exp(-lam7 * dt)
            fro = bi7 / lam7
            for k in range(5):
                Tk = x7[2 * k + 1] + xi5[k]
                x7[2 * k + 1] = Tk * fro + (x7[2 * k + 1] - Tk * fro) * ei7
                xi5[k] = Tk - x7[2 * k + 1]
        fic = p[24] * np.exp((0.6 / 25.693) * v)
        bic = p[25] * np.exp(-(0.6 / 25.693) * v)
        lamc = fic + bic
        if lamc > 0.0:
            eic = np.exp(-lamc * dt)
            frc = bic / lamc
            Tc = x7[8] + xiC
            x7[8] = Tc * frc + (x7[8] - Tc * frc) * eic
            xiC = Tc - x7[8]
        o7 = x7[1] + x7[3] + x7[5] + x7[7] + x7[9]
        k3t = p[4] * np.exp(p[3] * v)
        k4t = p[6] * np.exp(-p[5] * v)
        trt = 1.0 / (k3t + k4t)
        kc3t = p[8] * np.exp(p[9] * v)
        kc4t = p[10] * np.exp(-p[11] * v)
        tct = 1.0 / (kc3t + kc4t)
        r1 += dt * ((k4t * trt) - r1) / trt
        r2 += dt * ((kc4t * tct) - r2) / tct
        zr4 = min(max(p[12] + p[13] * v, -30.0), 30.0)
        wr4 = 1.0 / (1.0 + np.exp(-zr4))
        out[i, 0] = (p[7] * o7 * (wr4 * r1 + (1.0 - wr4) * r2) * (v - ek))
        out[i, 1] = s; out[i, 2] = o7; out[i, 3] = xiC
        out[i, 4] = xi5[0] + xi5[1] + xi5[2] + xi5[3] + xi5[4]
        out[i, 5] = r1; out[i, 6] = r2
    return out


d = json.load(open('/mnt/agents/output/04_细胞线4/结果/代码167_垂直时标卡_2026-09-08/work/counter_167_K6a9Jub.json', encoding='utf-8'))
p = np.asarray(d['params'], float); EK = float(d['EK_used'])
print(f"167 判决点：sse={d['sse']:.2f}  β={d['beta']:.4f}  EK={EK:.4f}")
print(f"c7={p[0]:.4f}/s  L7={p[1]:.5f}  θ={p[2]:.4f}  τ_s指前={p[14]:.2f}s  s∞半值={p[15]:.2f}mV  "
      f"ve耦合={p[16]:.2f}  s∞宽={p[17]:.4f}  z_τs={p[18]:.5f}")
print(f"τ_s(+50)={p[14]*np.exp(p[18]*50):.1f}s  τ_s(−120)={p[14]*np.exp(p[18]*(-120)):.0f}s  "
      f"s∞(+50)={1/(1+np.exp(-(50-p[15])/p[17])):.3f}  s∞(−120)={1/(1+np.exp(-(-120-p[15])/p[17])):.2e}")
print(f"u: τ0u={p[26]:.4f} z_τ={p[27]:.6f} M={p[28]:.4f} vh_u={p[29]:.2f} k_u={p[30]:.3f}；"
      f"τ_u(+50)={p[26]*np.exp(p[27]*50):.3f}s  u∞(+50)={1+(p[28]-1)/(1+np.exp(-(50-p[29])/p[30])):.4f}")

data = R.load_cell('16704007', ['deactivation', 'steady_activation', 'ap'])

# ===== 第一现场：deactivation 首段（−80→+50 2s→−120 6.5s） =====
Vd, Cd = data['deactivation']
n1 = int(9.1612 / DT)
tr = sim47_trace(p, Vd[:n1], DT, EK)
u1 = u_dyn(p, Vd[:n1], DT)
tt = np.arange(n1) * DT
print("\n===== deactivation 首段内部态轨迹（核电流/整机×u；单位 nA） =====")
print(f"{'t(s)':>7} {'V':>6} {'s':>8} {'O占据':>8} {'Ic':>8} {'I层':>7} {'r1':>6} {'r2':>6} "
      f"{'u':>7} {'核I':>9} {'整机I':>9} {'数据':>9}")
for tq in [0.40, 0.65, 0.70, 1.0, 1.5, 2.0, 2.5, 2.66,
           2.68, 2.71, 2.76, 2.86, 2.96, 3.16, 3.66, 4.66, 6.66, 9.0]:
    i = int(tq / DT)
    print(f"{tt[i]:7.2f} {Vd[i]:6.0f} {tr[i,1]:8.4f} {tr[i,2]:8.4f} {tr[i,3]:8.4f} {tr[i,4]:7.4f} "
          f"{tr[i,5]:6.3f} {tr[i,6]:6.3f} {u1[i]:7.4f} {tr[i,0]:9.4f} {tr[i,0]*u1[i]:9.4f} {Cd[i]:9.4f}")

np.savez(os.path.join(HERE, 'traj167_deact_seg1.npz'),
         t=tt, v=Vd[:n1], c=Cd[:n1], tr=tr, u=u1)
print("落盘 traj167_deact_seg1.npz")

# ===== 第二现场：steady_activation 末步（+40，t0≈41.92） =====
Vsa, Csa = data['steady_activation']
i40 = int(41.0 / DT); i41 = int(47.5 / DT)
tr2 = sim47_trace(p, Vsa[i40:i41], DT, EK)
u2 = u_dyn(p, Vsa[i40:i41], DT)
print("\n===== steady_activation +40 步段（步起点 41.92s；hrs[3] 判窗） =====")
print(f"{'t(s)':>7} {'V':>6} {'s':>8} {'O占据':>8} {'Ic':>8} {'u':>7} {'核I':>9} {'整机I':>9} {'数据':>9}")
for tq in [41.90, 41.95, 42.02, 42.12, 42.42, 42.92, 43.92, 45.92, 46.92, 47.42]:
    i = int((tq - 41.0) / DT)
    print(f"{tq:7.2f} {Vsa[i40+i]:6.0f} {tr2[i,1]:8.4f} {tr2[i,2]:8.4f} {tr2[i,3]:8.4f} "
          f"{u2[i]:7.4f} {tr2[i,0]:9.4f} {tr2[i,0]*u2[i]:9.4f} {Csa[i40+i]:9.4f}")

# ===== 第三现场：AP 协议全段（u 的舞台） =====
Vap, Cap = data['ap']
tr3 = sim47_trace(p, Vap, DT, EK)
u3 = u_dyn(p, Vap, DT)
nap = len(Vap)
print(f"\n===== AP 协议（全长 {nap*DT:.2f}s；u 逐搏积分） =====")
print(f"{'t(s)':>7} {'V':>6} {'s':>8} {'O占据':>8} {'u':>7} {'核I':>9} {'整机I':>9} {'数据':>9}")
for tq in np.linspace(0, (nap - 1) * DT, 13):
    i = int(tq / DT)
    print(f"{i*DT:7.2f} {Vap[i]:6.1f} {tr3[i,1]:8.4f} {tr3[i,2]:8.4f} {u3[i]:7.4f} "
          f"{tr3[i,0]:9.4f} {tr3[i,0]*u3[i]:9.4f} {Cap[i]:9.4f}")
print(f"u 全段范围：[{u3.min():.4f}, {u3.max():.4f}]；s 全段范围：[{tr3[:,1].min():.4f}, {tr3[:,1].max():.4f}]")

print("\n仪器化完。")
