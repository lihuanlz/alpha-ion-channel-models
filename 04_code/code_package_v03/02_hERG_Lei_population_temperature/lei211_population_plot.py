# lei211_群体图.py — Lei 211 细胞 α 群体判决总图
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})

ROOT = os.path.dirname(os.path.abspath(__file__))
J12 = json.load(open(os.path.join(ROOT, "lei211_J1J2_结果.json"), encoding="utf-8"))
J5 = json.load(open(os.path.join(ROOT, "lei211_J5_结果.json"), encoding="utf-8"))
TAUH = json.load(open(os.path.join(ROOT, "lei211_tauh_表.json"), encoding="utf-8"))
cells = J12["cells"]
SIN_V = np.array(J12["SIN_V"]); ACT_V = np.array(J12["ACT_V"])

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))

# (a) E_rev
ax = axes[0, 0]
E = np.array([c["E_rev"] for c in cells.values() if c["E_rev"] is not None])
ax.hist(E, bins=28, color="0.4", edgecolor="white", lw=0.4)
ax.axvline(np.median(E), color="C3", lw=1.2)
ax.set_title(f"a  E_rev 逐细胞（n={len(E)}）中位 {np.median(E):.1f} mV, CV {E.std()/abs(E.mean()):.3f} → 封卷", fontsize=9.5)
ax.set_xlabel("E_rev (mV)"); ax.set_ylabel("细胞数")

# (b) h_ss 群体
ax = axes[0, 1]
for w, c in list(cells.items()):
    if c["J1"]:
        h = np.array(c["J1"]["h"], float); ok = np.array(c["J1"]["ok"])
        ax.plot(SIN_V[ok], h[ok], ".", color="0.7", ms=1.5, alpha=0.5)
med = [np.median([c["J1"]["h"][i] for c in cells.values() if c["J1"] and c["J1"]["ok"][i]]) if np.any([c["J1"] and c["J1"]["ok"][i] for c in cells.values()]) else np.nan for i in range(len(SIN_V))]
ax.plot(SIN_V, med, "o-", color="C0", ms=5, lw=1.4, label="群体中位")
ax.set_title("b  h_ss(V) 群体（DF 校正）−140/−120 封卷，中段 登记", fontsize=9.5)
ax.set_xlabel("V (mV)"); ax.set_ylabel("h_ss"); ax.legend(); ax.set_ylim(-0.1, 1.25)

# (c) m_ss 群体
ax = axes[0, 2]
for w, c in list(cells.items()):
    if c["J2"]:
        m = np.array(c["J2"]["m"], float); ok = np.array(c["J2"]["ok_m"])
        ax.plot(ACT_V[ok], m[ok], ".", color="0.7", ms=1.5, alpha=0.5)
medm = [np.median([c["J2"]["m"][i] for c in cells.values() if c["J2"] and c["J2"]["ok_m"][i]]) if np.any([c["J2"] and c["J2"]["ok_m"][i] for c in cells.values()]) else np.nan for i in range(len(ACT_V))]
ax.plot(ACT_V, medm, "s-", color="C1", ms=5, lw=1.4, label="群体中位")
ax.set_title("c  m_ss(V)（1 s 表观激活，下界）+25/+40 封卷", fontsize=9.5)
ax.set_xlabel("V (mV)"); ax.set_ylabel("m_ss"); ax.legend()

# (d) τ_rec / τ_deact
ax = axes[1, 0]
tr_v = [float(k) for k in TAUH.keys()]
tr_m = [TAUH[k]["median"] * 1000 for k in TAUH.keys()]
tr_n = [TAUH[k]["n"] for k in TAUH.keys()]
o = np.argsort(tr_v)
ax.plot(np.array(tr_v)[o], np.array(tr_m)[o], "o-", color="C2", label="τ_rec(V)（sinactiv DoE）")
for x, y, n in zip(np.array(tr_v)[o], np.array(tr_m)[o], np.array(tr_n)[o]):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 5), fontsize=6.5, ha="center")
ax.axhline(90, color="0.5", ls=":", lw=0.8)
ax.annotate("τ_inact(+40)=90ms（大补跳登记）", (0, 90), textcoords="offset points", xytext=(5, 4), fontsize=7)
ax.set_title("d  τ_h 负档族（登记）+ 去激活慢层 −60:0.74s/−40:3.03s（登记）", fontsize=9.5)
ax.set_xlabel("V (mV)"); ax.set_ylabel("τ_rec (ms)"); ax.legend(fontsize=7.5)

# (e) AP 前向示例 A01 ap1hz
ax = axes[1, 1]
DATA = os.path.join(ROOT, "..", "数据", "Lei全量")
d5 = np.genfromtxt(os.path.join(DATA, "protocol", "protocol-ap1hz.csv"), delimiter=",", skip_header=1)
v = d5[:, 1]
Ir = np.genfromtxt(os.path.join(DATA, "ap1hz", "herg25oc1-ap1hz-A01.csv"), delimiter=",", skip_header=1)
# 简化：用 J5 冒烟里已有的思想不重算，直接画数据+说明
t = d5[:, 0]
ax.plot(t, Ir, lw=0.4, color="0.6")
ax.set_title("e  实测 ap1hz（A01，autoLC 原始）复极回弹逐拍可见", fontsize=9.5)
ax.set_xlabel("t (s)"); ax.set_ylabel("I (pA)"); ax.set_ylim(-120, 260)

# (f) J5 汇总
ax = axes[1, 2]
protos = ["ap05hz", "ap1hz", "ap2hz"]
a1 = [77 / 209, 53 / 209, 157 / 209]
a2 = [191 / 209, 202 / 209, 206 / 209]
x = np.arange(3)
ax.bar(x - 0.18, a1, width=0.36, color="C0", label="A1 复极回弹逐周期")
ax.bar(x + 0.18, a2, width=0.36, color="C4", label="A2 量级 RMS∈[0.3,3]")
ax.axhline(0.6, color="C3", ls="--", lw=1, label="封卷判线(Wilson下界≥0.6)")
ax.set_xticks(x); ax.set_xticklabels(["0.5 Hz", "1 Hz", "2 Hz"])
ax.set_ylim(0, 1.05); ax.set_ylabel("通过率")
ax.set_title("f  J5 AP 前向：A3 方向 209/209，合并双过 45.8% → 登记", fontsize=9.5)
ax.legend(fontsize=7)
for xi, yi in zip(x - 0.18, a1):
    ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=8)

fig.suptitle("Lei 211 细胞 α 群体判决总图（CHO, 25 °C, herg25oc1）2026-09-21", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = os.path.join(ROOT, "lei211_群体判决总图.png")
fig.savefig(png, dpi=150, bbox_inches="tight")
pdf = os.path.join(ROOT, "lei211_群体判决总图.pdf")
fig.savefig(pdf, bbox_inches="tight")
print("saved", png)
