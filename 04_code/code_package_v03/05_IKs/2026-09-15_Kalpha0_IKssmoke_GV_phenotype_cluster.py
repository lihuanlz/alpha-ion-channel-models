"""Kα-1 冒烟：对 IKs activation 类协议逐文件提 (V1/2, k, 尾巴衰减)，看构型簇结构。
尾巴峰 = 阶跃结束后 -40mV 段头 200ms 内最大外向幅值。
"""
import pyabf, os, json, glob
import numpy as np
from scipy.optimize import curve_fit

D = r"04_细胞线4/数据/iks_chan_8226585"
inv = json.load(open(os.path.join(D, "inventory.json")))
PROTO = {"IKs activation -80", "KCNQ act 4sec - low to high",
         "KCNQ act 4sec - high to low EQ+MA", "KCNQ GV",
         "KCNQ act 4sec - high to low start +80"}

def boltz(V, Vh, k):
    return 1.0 / (1.0 + np.exp(-(V - Vh) / k))

def analyze(path):
    a = pyabf.ABF(path)
    dt = 1.0 / a.sampleRate
    Vs, amps, taus = [], [], []
    for s in range(a.sweepCount):
        a.setSweep(s)
        e = a.sweepEpochs
        if len(e.types) < 5:
            continue
        # epoch 2 = 4s 阶跃, epoch 3 = -40 尾巴
        v_step = float(e.levels[2]); p3, p4 = int(e.p1s[3]), int(e.p2s[3])
        v_tail = float(e.levels[3])
        if abs(v_tail + 40) > 5:
            continue
        y = a.sweepY.astype(float)
        seg = y[p3:p3 + min(p4 - p3, int(0.2 / dt))]  # 尾巴头 200ms
        if len(seg) < 10:
            continue
        pk = float(np.max(seg))
        Vs.append(v_step); amps.append(pk)
        # 尾巴单指数 τ（头 200ms 窗，峰后衰减）
        i_pk = int(np.argmax(seg))
        dec = seg[i_pk:]
        if len(dec) > 30 and pk > 0:
            t = np.arange(len(dec)) * dt * 1000.0
            try:
                popt, _ = curve_fit(lambda tt, A, tau, C: A * np.exp(-tt / tau) + C,
                                    t, dec, p0=[dec[0], 200.0, 0.0],
                                    bounds=([0, 5, -np.inf], [np.inf, 3000, np.inf]),
                                    maxfev=4000)
                taus.append(float(popt[1]))
            except Exception:
                taus.append(np.nan)
        else:
            taus.append(np.nan)
    if len(Vs) < 8:
        return None
    Vs = np.array(Vs); amps = np.array(amps)
    base = np.percentile(amps, 5)
    amps_n = amps - base
    mx = amps_n.max()
    if mx <= 0:
        return None
    amps_n /= mx
    order = np.argsort(Vs)
    try:
        popt, _ = curve_fit(boltz, Vs[order], amps_n[order], p0=[25, 19],
                            bounds=([-80, 2], [120, 80]), maxfev=8000)
        Vh, k = float(popt[0]), float(popt[1])
        r2 = 1 - np.sum((amps_n[order] - boltz(Vs[order], Vh, k)) ** 2) / np.sum((amps_n[order] - amps_n.mean()) ** 2)
    except Exception:
        Vh, k, r2 = np.nan, np.nan, np.nan
    tau_med = float(np.nanmedian(taus)) if taus else np.nan
    # 波形瞬时度：+60 阶跃段头 50ms 幅值 / 段末幅值（瞬时 onset 判据）
    return dict(Vh=Vh, k=k, r2=r2, tau_deact40=tau_med, n_steps=len(Vs))

out = []
for x in inv:
    if x.get("protocol") not in PROTO:
        continue
    p = os.path.join(D, x["file"])
    try:
        r = analyze(p)
        if r:
            r.update(file=x["file"], protocol=x["protocol"])
            out.append(r)
    except Exception as e:
        pass

json.dump(out, open(os.path.join(D, "smoke_gv.json"), "w"), indent=1)
print(f"analyzed {len(out)} files")
vhs = np.array([r["Vh"] for r in out if not np.isnan(r["Vh"])])
print(f"V1/2 分布: n={len(vhs)}")
hist, edges = np.histogram(vhs, bins=np.arange(-100, 130, 10))
for h, e in zip(hist, edges):
    if h: print(f"  {e:+5.0f}..{e+10:+4.0f}: {'#'*h} {h}")
# WT 锚
sel = [r for r in out if not np.isnan(r["Vh"]) and 18 <= r["Vh"] <= 33 and 12 <= r["k"] <= 28 and r["r2"] > 0.9]
print(f"\nWT 表型候选 (V1/2 18-33, k 12-28, R2>0.9): {len(sel)}")
for r in sorted(sel, key=lambda z: z["Vh"]):
    print(f"  {r['file'][:42]:44s} V1/2={r['Vh']:+6.1f} k={r['k']:5.1f} R2={r['r2']:.3f} tau40={r['tau_deact40']:7.1f}ms")
