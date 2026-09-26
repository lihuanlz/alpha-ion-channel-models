# -*- coding: utf-8 -*-
"""
Kα-2 full run: digitization of Fedida 2024 Fig 3B wt EQ (blue dots) tau_act-V
Preregistration: 预注册_Kα2_IKs跨论文τact对拍_2026-09-15.md (v1.0)
Method: programmatic axis-tick detection + HSV blue segmentation + row-width criterion separating filled disks from thin curves/error bars
Output: digitized table CSV + overlay verification PNG + verdict-component JSON
"""
import json
import numpy as np
from PIL import Image, ImageDraw

IMG = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\数据\iks_chan_8226585\fedida2024_fig3.jpg"
OUT_CSV = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv"
OUT_PNG = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_Fedida2024_Fig3B_叠加核验.png"
OUT_JSON = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_判词组件.json"

# ---------- 1. read image, panel B search region ----------
im = Image.open(IMG).convert("RGB")
W, H = im.size  # 763 x 762
a = np.asarray(im).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]

# Panel B coarse region (original-image coordinates, fixed by smoke visual inspection)
PX0, PX1, PY0, PY1 = 290, 745, 15, 410

# ---------- 2. axis calibration: detect y-axis vertical line, x-axis horizontal line, ticks ----------
dark = (R < 90) & (G < 90) & (B < 90)

sub = dark[PY0:PY1, PX0:PX1]
colsum = sub.sum(axis=0)
rowsum = sub.sum(axis=1)

# y axis: column with the longest vertical black line
yaxis_x = PX0 + int(np.argmax(colsum))
# x axis: row with the longest horizontal black line to the right of the y axis
sub2 = dark[PY0:PY1, yaxis_x:PX1]
rowsum2 = sub2.sum(axis=1)
xaxis_y = PY0 + int(np.argmax(rowsum2))

# x ticks: short vertical black segments 2-6 px below the x axis
tick_band = dark[xaxis_y + 2:xaxis_y + 7, yaxis_x:PX1]
tick_cols = np.where(tick_band.sum(axis=0) >= 3)[0]
# clustering
def clusters(idx):
    out = []
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for v in idx[1:]:
        if v - p > 2:
            out.append((s + p) / 2)
            s = v
        p = v
    out.append((s + p) / 2)
    return out

xticks = [yaxis_x + c for c in clusters(tick_cols)]
# also look for -60/-30 left of the x axis origin (left of the y axis)
tick_band_l = dark[xaxis_y + 2:xaxis_y + 7, PX0:yaxis_x]
xticks_l = [PX0 + c for c in clusters(np.where(tick_band_l.sum(axis=0) >= 3)[0])]
xticks_all = sorted(xticks_l + xticks)

# y ticks: short horizontal black segments 2-6 px left of the y axis
ytick_band = dark[PY0:xaxis_y, yaxis_x - 7:yaxis_x - 2]
yticks = [PY0 + r for r in clusters(np.where(ytick_band.sum(axis=1) >= 3)[0])]
yticks = sorted(yticks)

print(f"[calib] y-axis x={yaxis_x}  x-axis y={xaxis_y}")
print(f"[calib] raw x ticks: {[round(t,1) for t in xticks_all]}")
print(f"[calib] raw y ticks: {[round(t,1) for t in yticks]}")

# --- robust x-tick grid: main spacing = median of gaps >30px; after grid snapping, drop spurious ticks >3px off-grid ---
xt = np.array(sorted(xticks_all))
gaps = np.diff(xt)
main_gap = np.median(gaps[gaps > 30])  # main spacing ~46.4 px / 30 mV
# anchor: y-axis line = 0 mV, so grid points = yaxis_x + main_gap*k (k integer)
k_raw = np.rint((xt - yaxis_x) / main_gap)
grid_x = yaxis_x + k_raw * main_gap
off = np.abs(xt - grid_x)
keep = off < 3.0
assert keep.sum() >= 6, f"too few on-grid ticks: {keep.sum()}"
dropped = xt[~keep]
xt_k = k_raw[keep]
xt_v = xt[keep]
# linear fit pixel = a + b*k (using on-grid ticks)
b, a = np.polyfit(xt_k, xt_v, 1)
resid = np.abs((a + b * xt_k) - xt_v)
assert resid.max() < 3.0, f"x-tick grid fit residual too large: {resid}"
x0pix = a                   # 0 mV pixel (k=0)
sx = b / 30.0               # px/mV
print(f"[calib] x grid: 0mV@{x0pix:.1f}px  sx={sx:.4f}px/mV  {len(xt_v)} ticks max residual={resid.max():.2f}px  dropped spurious={list(np.round(dropped,1))}")
assert abs(pix := (x0pix - yaxis_x)) < 3.0, f"0mV grid point does not coincide with the y axis: {pix}"

yticks = np.array(yticks)
dy = np.diff(yticks)
assert np.std(dy) / np.mean(dy) < 0.05, f"y-tick spacing uneven: {dy}"
sy = np.mean(dy)  # px/s
# y0 (tau=0) should coincide with the x axis: extrapolate from the tick band
y0pix = yticks[-1] + sy * 1.0  # lowest detected tick = 1s
assert abs(y0pix - xaxis_y) < 2.0, f"extrapolated tau0={y0pix} vs x axis={xaxis_y} mismatch"
y0pix = xaxis_y  # pinned to the x-axis line
print(f"[calib] y grid: tau0@{y0pix}px  sy={sy:.3f}px/s  check tau10={y0pix-10*sy:.1f}px")

def pix2V(x):
    return (x - x0pix) / sx

def V2pix(v):
    return x0pix + v * sx

def pix2tau(y):
    return (y0pix - y) / sy

# ---------- 3. blue segmentation ----------
blue = (B > 120) & (B - R > 60) & (B - G > 60)

# ---------- 4. locate the filled disk at each voltage ----------
results = []
vis = im.copy()
dr = ImageDraw.Draw(vis)

for V in range(-60, 181, 10):
    xp = V2pix(V)
    xlo, xhi = int(round(xp)) - 4, int(round(xp)) + 5
    # search range: inside the panel B plotting area (right of y axis, above x axis)
    ylo, yhi = PY0, xaxis_y - 1
    strip = blue[ylo:yhi, xlo:xhi]
    rowcnt = strip.sum(axis=1)
    rows = np.where(rowcnt >= 5)[0]  # disk rows
    if len(rows) == 0:
        results.append((V, None, "no disk"))
        continue
    # cluster row bands; take the band with the largest total blue mass (excludes thin error-bar cap bands)
    bands = []
    s = p = rows[0]
    for v in rows[1:]:
        if v - p > 2:
            bands.append((s, p))
            s = v
        p = v
    bands.append((s, p))
    mass = [rowcnt[b0:b1 + 1].sum() for b0, b1 in bands]
    b0, b1 = bands[int(np.argmax(mass))]
    # centroid row (within the band, weighted by row width)
    w = rowcnt[b0:b1 + 1].astype(float)
    ycen = ylo + (np.arange(b0, b1 + 1) * w).sum() / w.sum()
    tau = pix2tau(ycen)
    results.append((V, tau, f"band {b1-b0+1} rows mass {int(max(mass))}"))
    dr.ellipse([xp - 5, ycen - 5, xp + 5, ycen + 5], outline=(255, 0, 255), width=1)
    dr.text((xp + 6, ycen - 4), f"{V}:{tau:.2f}", fill=(255, 0, 255))

# ---------- 5. output ----------
with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    f.write("V_mV,tau_s,note\n")
    for V, tau, note in results:
        f.write(f"{V},{'' if tau is None else round(tau,3)},{note}\n")

print("\n[digitized] V -> tau_act (s)")
for V, tau, note in results:
    print(f"  {V:+5d} mV : {'  --  ' if tau is None else f'{tau:6.3f}'}  ({note})")

# verdict components
tau60 = dict((V, t) for V, t, _ in results).get(60)
tau60 = None if tau60 is None else float(tau60)
verdict = {
    "tau60_fedida_s": tau60,
    "ours_tau60_median_s": 1.491,
    "K2_1_band": [0.7455, 2.982],
    "K2_1_pass": None if tau60 is None else bool(0.7455 <= tau60 <= 2.982),
    "K2_2_band": [1.40, 2.60],
    "K2_2_pass": None if tau60 is None else bool(1.40 <= tau60 <= 2.60),
    "calib": {"sx_px_per_mV": float(sx), "sy_px_per_s": float(sy), "y0pix": float(y0pix), "x0pix": float(x0pix)},
}
json.dump(verdict, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
vis.save(OUT_PNG)
print(f"\n[verdict components] tau60_fedida={tau60}  K2-1={verdict['K2_1_pass']}  K2-2={verdict['K2_2_pass']}")
print(f"[saved] {OUT_CSV}")
print(f"[saved] {OUT_PNG}")
print(f"[saved] {OUT_JSON}")
