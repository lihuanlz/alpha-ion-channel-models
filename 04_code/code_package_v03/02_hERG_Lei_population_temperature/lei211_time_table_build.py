# lei211_时间表生成.py
# 从 lei211_温度汇总.json 生成完整时间表 md（五温度 × 全电压，每格标判决等级）
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(ROOT, "lei211_温度汇总.json"), encoding="utf-8"))
S = D["各批"]
COLS = [("herg25oc1", 25), ("herg27oc1", 27), ("herg30oc1", 30),
        ("herg33oc1", 33), ("herg37合并", 37)]
SIN_V7 = ["-140", "-120", "-100", "-80", "-60", "-40", "-20"]
SIN_V10 = SIN_V7 + ["0", "20", "40"]
ACT_V = ["-50", "-35", "-20", "-5", "10", "25", "40"]
C1_VOID = {"-100", "-60", "-20"}  # C1 作废档（沿用 25°C 规则）

def grade(cv, n, line=0.3, nmin=53):
    if cv is None or n is None or n < nmin:
        return "数据不足"
    return "封卷" if cv < line else "登记"

def cell(x, scale=1.0, nd=3, line=0.3):
    if x["median"] is None:
        return "—"
    v = x["median"] * scale
    g = grade(x["cv"], x["n"], line)
    mark = {"封卷": "★", "登记": "○", "数据不足": "△"}[g]
    return f"{v:.{nd}f} {mark}<br>n{x['n']} CV{x['cv']:.2f}" if x['cv'] is not None else f"{v:.{nd}f} {mark}<br>n{x['n']}"

L = []
L.append("# 完整时间表：Lei 五温度 × 全电压 群体常数表（判决标注版）\n")
L.append("2026-09-21 · 细胞线 4 hERG · 数据源 Lei et al. 2019 Part I/II（CHO，autoLC + 官方再漏减）")
L.append("五温度点 25/27/30/33/37 °C（37 °C = herg37oc3+37oc4 合并 n=105）；群体统计量 = 中位数，逐批提取同一管道。")
L.append("标注：★=封卷（CV<0.3，E_rev CV<0.10；n≥53）○=登记 △=数据不足 ※=C1 作废档（只登记不判决）\n")

# 1 τ_rec
L.append("## 1. τ_rec(V,T) 恢复时间常数 [ms]（sinactiv DoE）\n")
L.append("| V \\ T | 25 | 27 | 30 | 33 | 37 |")
L.append("|---|---|---|---|---|---|")
for V in SIN_V7:
    row = [f"{V} mV"]
    for b, _ in COLS:
        x = S[b]["trec"][V]
        row.append(cell(x, 1000, 2))
    L.append("| " + " | ".join(row) + " |")
L.append("\nQ10(25→37)：−140 档 2.84 ★（R²=0.991）、−120 档 2.84 ★（R²=0.991）\n")

# 2 τ_deact
L.append("## 2. τ_deact(V,T) 去激活双指数 [s]（staircase 下跳尾）\n")
L.append("| 量 \\ T | 25 | 27 | 30 | 33 | 37 |")
L.append("|---|---|---|---|---|---|")
for V in ("-60", "-40"):
    for comp in ("tau1", "tau2"):
        row = [f"{V} mV {comp}"]
        for b, _ in COLS:
            x = S[b]["tdeact"][V][comp]
            row.append(cell(x, 1.0, 3))
        L.append("| " + " | ".join(row) + " |")
L.append("\n注：37 °C 批 τ2 已贴近 0.5 s 阶梯窗上限量级，慢分量识别窗口受限（判词卡 §T1-J3 登记）；Q10(τ2,−40)=7.6 出带 → 登记。\n")

# 3 大补跳
L.append("## 3. 大补跳 −80→+40 [ms]（staircase）\n")
L.append("| 量 \\ T | 25 | 27 | 30 | 33 | 37 |")
L.append("|---|---|---|---|---|---|")
row = ["τ_inact(+40)（τ1）"]
for b, _ in COLS:
    row.append(cell(S[b]["bigstep"]["tau1_失活"], 1000, 1))
L.append("| " + " | ".join(row) + " |")
row = ["τ_act(+40)（τ2）"]
for b, _ in COLS:
    row.append(cell(S[b]["bigstep"]["tau2_激活"], 1000, 1))
L.append("| " + " | ".join(row) + " |")
L.append("\nQ10：τ_inact 1.83 ★（线性不成立，27–33 °C 网格平台）、τ_act 1.50 ★（线性不成立，30 °C 非单调）。\n")

# 4 E_rev
L.append("## 4. E_rev(T) 逐细胞反转电位 [mV]\n")
L.append("| T | 中位 | n | CV | 判决 |")
L.append("|---|---|---|---|---|")
for b, T in COLS:
    x = S[b]["E_rev"]
    g = "封卷" if (x["cv"] is not None and x["cv"] < 0.10) else "登记"
    mark = "★" if g == "封卷" else "○"
    L.append(f"| {T} °C | {x['median']:.2f} | {x['n']} | {x['cv']:.4f} | {mark}{g} |")
L.append("\nE_rev(T) 斜率 −0.58 mV/°C（方向 Nernst ✓，出 [0.05,0.35] 带 → 登记）；37 °C 中位 −92.04 mV 与 model62/149 拟合 −92.9 mV 互证。\n")

# 5 h_ss
L.append("## 5. h_ss(V,T) 稳态失活（DF 校正，逐细胞归一）\n")
L.append("| V \\ T | 25 | 27 | 30 | 33 | 37 |")
L.append("|---|---|---|---|---|---|")
for V in SIN_V10:
    row = [f"{V} mV"]
    for b, _ in COLS:
        x = S[b]["h_ss"][V]
        c = cell(x, 1.0, 3)
        if V in C1_VOID:
            c = c.replace("★", "※").replace("○", "※")
        row.append(c)
    L.append("| " + " | ".join(row) + " |")
L.append("")

# 6 m_ss
L.append("## 6. m_ss(V,T) 表观稳态激活（1 s 脉冲下界）\n")
L.append("| V \\ T | 25 | 27 | 30 | 33 | 37 |")
L.append("|---|---|---|---|---|---|")
for V in ACT_V:
    row = [f"{V} mV"]
    for b, _ in COLS:
        x = S[b]["m_ss"][V]
        row.append(cell(x, 1.0, 3))
    L.append("| " + " | ".join(row) + " |")
L.append("")

# 7 J5 (37°C)
j5 = S["herg37合并"]["J5"]
L.append("## 7. J5 AP 前向（37 °C 合并，n=102–105）\n")
L.append("| 协议 | A1 过 | A2 过 | n | 回弹峰比中位 |")
L.append("|---|---|---|---|---|")
for f, lab in (("ap05hz", "0.5 Hz"), ("ap1hz", "1 Hz"), ("ap2hz", "2 Hz")):
    x = j5["per_freq"][f]
    rb = "—" if x["reb_median"] is None else f"{x['reb_median']:.3f}"
    L.append(f"| {lab} | {x['A1']} | {x['A2']} | {x['n']} | {rb} |")
m = j5["合并"]
L.append(f"\nA3 频率方向 {j5['A3']['pass']}/{j5['A3']['n']}；合并双过 {m['k']}/{m['n']} = {m['p']*100:.1f}%，"
         f"Wilson 95% CI [{m['lo']:.3f}, {m['hi']:.3f}] → 登记（下界 <0.6）。\n")

fp = os.path.join(ROOT, "..", "结果", "时间表_Lei五温度_2026-09-21.md")
open(fp, "w", encoding="utf-8").write("\n".join(L))
print("saved", os.path.abspath(fp))
