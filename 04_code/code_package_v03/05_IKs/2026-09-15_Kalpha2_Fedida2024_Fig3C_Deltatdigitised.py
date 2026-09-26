# -*- coding: utf-8 -*-
"""
Kα-2 补件（预注册 K2-3，登记不判）：Fedida 2024 Fig 3C wt EQ Δt(+40/+60) 数字化
"""
import numpy as np
from PIL import Image, ImageDraw

IMG = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\数据\iks_chan_8226585\fedida2024_fig3.jpg"
OUT_PNG = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_Fedida2024_Fig3C_叠加核验.png"

im = Image.open(IMG).convert("RGB")
a = np.asarray(im).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
dark = (R < 90) & (G < 90) & (B < 90)
blue = (B > 120) & (B - R > 60) & (B - G > 60)

# 面板 C 粗区域（原图坐标）：左下图
PX0, PX1, PY0, PY1 = 0, 300, 400, 762

sub = dark[PY0:PY1, PX0:PX1]
colsum = sub.sum(axis=0)
rowsum = sub.sum(axis=1)
yaxis_x = PX0 + int(np.argmax(colsum))
sub2 = dark[PY0:PY1, yaxis_x:PX1]
xaxis_y = PY0 + int(np.argmax(sub2.sum(axis=1)))
print(f"[C标定] y轴 x={yaxis_x}  x轴 y={xaxis_y}")

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

# x 刻度（轴下方）
tb = dark[xaxis_y + 2:xaxis_y + 7, PX0:PX1]
xticks = sorted(PX0 + c for c in clusters(np.where(tb.sum(axis=0) >= 3)[0]))
# y 刻度（轴左侧）
yb = dark[PY0:xaxis_y, yaxis_x - 7:yaxis_x - 2]
yticks = sorted(PY0 + r for r in clusters(np.where(yb.sum(axis=1) >= 3)[0]))
print(f"[C标定] x刻度: {xticks}")
print(f"[C标定] y刻度: {yticks}")

# 面板 C x 轴标签（从全图目检）：-60, -10, 40, 90（间隔 50 mV）
# y 轴：0, 0.4, 0.8, 1.2, 1.6, 2
xt = np.array(xticks, float)
gaps = np.diff(xt)
print("[C标定] x间隔:", gaps)
# y 轴穿零风格？检查：x=-60 是最左刻度 → 0mV 位 = xt[0] + 60/50*gap
main_gap = np.median(gaps)
sx = main_gap / 50.0
x0pix = xt[0] + 60.0 * sx  # 0 mV 像素（-60 + 60 = 0）
print(f"[C标定] 假定x刻度起自-60：sx={sx:.4f}px/mV  0mV@{x0pix:.1f}  y轴@{yaxis_x}  差={x0pix-yaxis_x:.1f}px")

yt = np.array(yticks, float)
dg = np.diff(yt)
sy = np.median(dg) / 0.4  # 0.4 s 一格
# τ=0 与 x 轴重合校验
y0pix = xaxis_y
print(f"[C标定] sy={sy:.2f}px/s  y间隔={dg}  最上刻度={yt[0]} -> {(y0pix-yt[0])/sy:.2f}s（应≈2.0）")

def V2pix(v):
    return x0pix + v * sx

vis = im.copy()
dr = ImageDraw.Draw(vis)
out = {}
for V in [40, 60]:
    xp = V2pix(V)
    xlo, xhi = int(round(xp)) - 4, int(round(xp)) + 5
    strip = blue[PY0:xaxis_y - 1, xlo:xhi]
    rowcnt = strip.sum(axis=1)
    rows = np.where(rowcnt >= 5)[0]
    if len(rows) == 0:
        out[V] = None
        print(f"[C] +{V} mV: 无圆盘")
        continue
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
    w = rowcnt[b0:b1 + 1].astype(float)
    ycen = PY0 + (np.arange(b0, b1 + 1) * w).sum() / w.sum()
    dt = (y0pix - ycen) / sy
    out[V] = float(dt)
    dr.ellipse([xp - 5, ycen - 5, xp + 5, ycen + 5], outline=(255, 0, 255), width=1)
    dr.text((xp + 6, ycen - 4), f"+{V}:{dt:.2f}s", fill=(255, 0, 255))
    print(f"[C] +{V} mV: Δt = {dt:.3f} s  (带{b1-b0+1}行)")

vis.save(OUT_PNG)
print(f"[落盘] {OUT_PNG}")
import json
json.dump({"dt40_s": out.get(40), "dt60_s": out.get(60)},
          open(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型\2026-09-15_Kα2_Δt登记.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
