# lei211_C1_合成回收.py
# C1 对照（预注册 §3）：合成 sinactiv 测试段，回收 h_ss 档值误差须 <10%
# 物理：+20 预脉冲后 m0=0.9, h0=0.02（全失活）；测试段 dh/dt=(h_ss-h)/τ_rec(V)，
#       dm/dt=(0-m)/τ_deact(V)；I=G·m·h·(V-E_rev)+噪声(σ=11pA, 实测量级)
# 用 J1 统计量（A=极值-末段, DF校正归一）回收 h(V)/h(-140)
import numpy as np

DT = 2e-4
E_REV = -83.9
G = 22.0  # nS
SIG = 11.0

# 测试条件（群体中位数）
H_TRUE = {-140: 1.00, -120: 0.95, -100: 0.88, -60: 0.23, -40: 0.09, -20: 0.055}
TREC = {-140: 0.0043, -120: 0.0053, -100: 0.0065, -60: 0.02, -40: 0.05, -20: 0.1}
TDEC = {-140: 0.05, -120: 0.06, -100: 0.08, -60: 0.3, -40: 0.8, -20: 1.5}

rng = np.random.default_rng(7)
print("V | h_true | h_rec | err%")
ok = True
recs = {}
for V, h_ss in H_TRUE.items():
    t = np.arange(0, 0.5, DT)
    m = 0.9 * np.exp(-t / TDEC[V])
    h = h_ss + (0.02 - h_ss) * np.exp(-t / TREC[V])
    I = G * m * h * (V - E_REV) + rng.normal(0, SIG, len(t))
    # J1 统计量
    late = np.mean(I[-int(0.05 / DT):])
    win = I[: int(0.2 / DT)]
    pk = np.min(win) if V < E_REV else np.max(win)
    A = pk - late
    recs[V] = A / (V - E_REV)
for V in H_TRUE:
    h_rec = recs[V] / recs[-140]
    err = abs(h_rec - H_TRUE[V] / H_TRUE[-140]) / (H_TRUE[V] / H_TRUE[-140]) * 100
    flag = "PASS" if err < 10 else "FAIL"
    if err >= 10:
        ok = False
    print(f"{V:5d} | {H_TRUE[V]:.3f} | {h_rec:.3f} | {err:.1f}% {flag}")
print("C1", "PASS" if ok else "FAIL", "（判线：全档误差<10%）")
