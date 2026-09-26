# -*- coding: utf-8 -*-
"""
Kα-2 正式跑：Fedida 2024 Fig 3B wt EQ（蓝点）τact-V 数字化
预注册：预注册_Kα2_IKs跨论文τact对拍_2026-09-15.md（v1.0）
方法：轴刻度程序检测 + HSV 蓝色分割 + 行宽判据分离填充圆盘 vs 细曲线/误差线
输出：数字化表 CSV + 叠加核验图 PNG + 判词组件 JSON
"""
import json
import numpy as np
from PIL import Image, ImageDraw

IMG = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\数据\iks_chan_8226585\fedida2024_fig3.jpg"
OUT_CSV = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv"
OUT_PNG = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_Fedida2024_Fig3B_叠加核验.png"
OUT_JSON = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_判词组件.json"

# ---------- 1. 读图，面板 B 搜索区域 ----------
im = Image.open(IMG).convert("RGB")
W, H = im.size  # 763 x 762
a = np.asarray(im).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]

# 面板 B 粗区域（原图坐标，冒烟目检确定）
PX0, PX1, PY0, PY1 = 290, 745, 15, 410

# ---------- 2. 轴标定：检测 y 轴竖线、x 轴横线、刻度 ----------
dark = (R < 90) & (G < 90) & (B < 90)

sub = dark[PY0:PY1, PX0:PX1]
colsum = sub.sum(axis=0)
rowsum = sub.sum(axis=1)

# y 轴：最长竖直黑线列
yaxis_x = PX0 + int(np.argmax(colsum))
# x 轴：y 轴右侧区域最长水平黑线行
sub2 = dark[PY0:PY1, yaxis_x:PX1]
rowsum2 = sub2.sum(axis=1)
xaxis_y = PY0 + int(np.argmax(rowsum2))

# x 刻度：x 轴下方 2~6 px 的短竖黑段
tick_band = dark[xaxis_y + 2:xaxis_y + 7, yaxis_x:PX1]
tick_cols = np.where(tick_band.sum(axis=0) >= 3)[0]
# 聚类
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
# x 轴左侧也要找 -60/-30（在 y 轴左侧）
tick_band_l = dark[xaxis_y + 2:xaxis_y + 7, PX0:yaxis_x]
xticks_l = [PX0 + c for c in clusters(np.where(tick_band_l.sum(axis=0) >= 3)[0])]
xticks_all = sorted(xticks_l + xticks)

# y 刻度：y 轴左侧 2~6 px 短横黑段
ytick_band = dark[PY0:xaxis_y, yaxis_x - 7:yaxis_x - 2]
yticks = [PY0 + r for r in clusters(np.where(ytick_band.sum(axis=1) >= 3)[0])]
yticks = sorted(yticks)

print(f"[标定] y轴 x={yaxis_x}  x轴 y={xaxis_y}")
print(f"[标定] x刻度原始: {[round(t,1) for t in xticks_all]}")
print(f"[标定] y刻度原始: {[round(t,1) for t in yticks]}")

# --- x 刻度稳健网格：主间距取 >30px 间隔中位，网格吸附后剔除离网格 >3px 的伪刻度 ---
xt = np.array(sorted(xticks_all))
gaps = np.diff(xt)
main_gap = np.median(gaps[gaps > 30])  # 主间距 ~46.4 px / 30 mV
# 锚定：y 轴竖线=0mV，故网格点 = yaxis_x + main_gap*k（k 整数）
k_raw = np.rint((xt - yaxis_x) / main_gap)
grid_x = yaxis_x + k_raw * main_gap
off = np.abs(xt - grid_x)
keep = off < 3.0
assert keep.sum() >= 6, f"网格内刻度过少: {keep.sum()}"
dropped = xt[~keep]
xt_k = k_raw[keep]
xt_v = xt[keep]
# 线性拟合 像素 = a + b*k（用网格内刻度）
b, a = np.polyfit(xt_k, xt_v, 1)
resid = np.abs((a + b * xt_k) - xt_v)
assert resid.max() < 3.0, f"x刻度网格拟合残差过大: {resid}"
x0pix = a                   # 0 mV 像素（k=0）
sx = b / 30.0               # px/mV
print(f"[标定] x网格: 0mV@{x0pix:.1f}px  sx={sx:.4f}px/mV  刻度{len(xt_v)}个 残差max={resid.max():.2f}px  剔除伪刻度={list(np.round(dropped,1))}")
assert abs(pix := (x0pix - yaxis_x)) < 3.0, f"0mV网格点与y轴不重合: {pix}"

yticks = np.array(yticks)
dy = np.diff(yticks)
assert np.std(dy) / np.mean(dy) < 0.05, f"y刻度间距不齐: {dy}"
sy = np.mean(dy)  # px/s
# y0（τ=0）应与 x 轴重合：从刻度带外推
y0pix = yticks[-1] + sy * 1.0  # 最下检测刻度=1s
assert abs(y0pix - xaxis_y) < 2.0, f"外推τ0={y0pix} 与x轴={xaxis_y} 不符"
y0pix = xaxis_y  # 钉死在 x 轴线上
print(f"[标定] y网格: τ0@{y0pix}px  sy={sy:.3f}px/s  校验τ10={y0pix-10*sy:.1f}px")

def pix2V(x):
    return (x - x0pix) / sx

def V2pix(v):
    return x0pix + v * sx

def pix2tau(y):
    return (y0pix - y) / sy

# ---------- 3. 蓝色分割 ----------
blue = (B > 120) & (B - R > 60) & (B - G > 60)

# ---------- 4. 逐电压位找填充圆盘 ----------
results = []
vis = im.copy()
dr = ImageDraw.Draw(vis)

for V in range(-60, 181, 10):
    xp = V2pix(V)
    xlo, xhi = int(round(xp)) - 4, int(round(xp)) + 5
    # 搜索范围：面板 B 绘图区内（y 轴右、x 轴上）
    ylo, yhi = PY0, xaxis_y - 1
    strip = blue[ylo:yhi, xlo:xhi]
    rowcnt = strip.sum(axis=1)
    rows = np.where(rowcnt >= 5)[0]  # 圆盘行
    if len(rows) == 0:
        results.append((V, None, "无圆盘"))
        continue
    # 行带聚类，取蓝色总质量最大的带（排除误差棒盖帽细带）
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
    # 质心行（带内以行宽加权）
    w = rowcnt[b0:b1 + 1].astype(float)
    ycen = ylo + (np.arange(b0, b1 + 1) * w).sum() / w.sum()
    tau = pix2tau(ycen)
    results.append((V, tau, f"带{b1-b0+1}行 质量{int(max(mass))}"))
    dr.ellipse([xp - 5, ycen - 5, xp + 5, ycen + 5], outline=(255, 0, 255), width=1)
    dr.text((xp + 6, ycen - 4), f"{V}:{tau:.2f}", fill=(255, 0, 255))

# ---------- 5. 输出 ----------
with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    f.write("V_mV,tau_s,note\n")
    for V, tau, note in results:
        f.write(f"{V},{'' if tau is None else round(tau,3)},{note}\n")

print("\n[数字化] V -> tau_act (s)")
for V, tau, note in results:
    print(f"  {V:+5d} mV : {'  --  ' if tau is None else f'{tau:6.3f}'}  ({note})")

# 判词组件
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
print(f"\n[判词组件] tau60_fedida={tau60}  K2-1={verdict['K2_1_pass']}  K2-2={verdict['K2_2_pass']}")
print(f"[落盘] {OUT_CSV}")
print(f"[落盘] {OUT_PNG}")
print(f"[落盘] {OUT_JSON}")
