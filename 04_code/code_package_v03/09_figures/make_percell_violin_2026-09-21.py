# make_percell_violin_2026-09-21.py
# Four-channel per-cell parameter unified processing: long-table CSV + violin figure + failure registry table
# Data all come from the result files of the corresponding sealed verdict cards (same conventions as the seals)
import os, json, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))          # 04_细胞线4\文章
LINE = os.path.dirname(ROOT)                                # 04_细胞线4
AM = os.path.join(LINE, "α模型")
RES = os.path.join(LINE, "结果")
FIGD = os.path.join(ROOT, "figures_v01")
os.makedirs(FIGD, exist_ok=True)

BATCH_T = {"herg25oc1": 25, "herg27oc1": 27, "herg30oc1": 30,
           "herg33oc1": 33, "herg37oc3": 37, "herg37oc4": 37}

LONG = []   # per-cell parameter long table
FAIL = []   # failure registry

def add(channel, dataset, group, temp, cell, param, voltage, value, unit, ok=True, note=""):
    if value is None:
        return
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return
    except TypeError:
        return
    LONG.append(dict(channel=channel, dataset=dataset, group=group, temp_C=temp,
                     cell_id=str(cell), parameter=param, voltage_mV=voltage,
                     value=float(value), unit=unit, ok=int(ok), note=note))

def fail(channel, dataset, cid, scope, item, cat, detail, source):
    FAIL.append(dict(channel=channel, dataset=dataset, cell_or_file=str(cid),
                     scope=scope, failed_item=item, category=cat,
                     detail=detail, source=source))

# ============ 1. hERG · Lei six batches (670 wells) ============
SIN_V = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]
ACT_V = [-50, -35, -20, -5, 10, 25, 40]
for batch, T in BATCH_T.items():
    j12 = json.load(open(os.path.join(AM, f"lei211_J1J2_结果_{batch}.json"), encoding="utf-8"))
    j3 = json.load(open(os.path.join(AM, f"lei211_J3_结果_{batch}.json"), encoding="utf-8"))
    tauh = json.load(open(os.path.join(AM, f"lei211_tauh_逐孔_{batch}.json"), encoding="utf-8"))
    sv = j12["SIN_V"]; av = j12["ACT_V"]
    n_erev_missing = 0
    hss_fail_by_v = {v: [] for v in sv}
    tauh_missing_by_v = {v: 0 for v in [-140, -120, -100, -80, -60, -40, -20]}
    for w, c in j12["cells"].items():
        # E_rev
        er = c.get("E_rev")
        if er is None or (isinstance(er, float) and math.isnan(er)):
            n_erev_missing += 1
            fail("hERG", f"Lei-{batch}", w, "细胞", "E_rev", "A 数据质量",
                 "staircase 下坡未过零，E_rev 不可定位（漏减/电导异常）", "逐细胞重算（J1J2 JSON）")
        else:
            add("hERG", f"Lei-{batch}", "CHO", T, w, "E_rev", "", er, "mV")
        # h_ss
        for i, v in enumerate(sv):
            okv = c["J1"]["ok"][i]
            hv = c["J1"]["h"][i]
            if okv and hv == hv:
                add("hERG", f"Lei-{batch}", "CHO", T, w, "h_ss", v, hv, "")
            else:
                hss_fail_by_v[v].append(w)
        # m_ss
        for i, v in enumerate(av):
            if c["J2"]["ok_m"][i]:
                mv = c["J2"]["m"][i]
                if mv == mv:
                    add("hERG", f"Lei-{batch}", "CHO", T, w, "m_ss", v, mv, "")
        # tau_rec per well
        th = tauh.get(w, {})
        for v in [-140, -120, -100, -80, -60, -40, -20]:
            key = str(v)
            if key in th and th[key] == th[key]:
                add("hERG", f"Lei-{batch}", "CHO", T, w, "tau_rec", v, th[key] * 1000.0, "ms")
            else:
                tauh_missing_by_v[v] += 1
        # τ_deact / τ_act / τ_inact（J3）
        j3c = j3.get(w, {}).get("J3", {})
        for v in ["-60", "-40"]:
            tr = j3c.get("tdeact", {}).get(v, [])
            taus2 = [t["tau2"] for t in tr if t.get("tau2")]
            if taus2:
                add("hERG", f"Lei-{batch}", "CHO", T, w, "tau_deact_slow", v, float(np.median(taus2)), "s")
        bs = j3c.get("tact", {}).get("40_bigstep")
        if bs:
            if bs.get("tau1"):
                add("hERG", f"Lei-{batch}", "CHO", T, w, "tau_inact", 40, bs["tau1"] * 1000.0, "ms")
            if bs.get("tau2"):
                add("hERG", f"Lei-{batch}", "CHO", T, w, "tau_act", 40, bs["tau2"] * 1000.0, "ms")
    # batch-level failure registry
    if n_erev_missing:
        pass  # already registered per cell
    for v, cells in hss_fail_by_v.items():
        if cells:
            cat = "B 协议限制" if v == -80 else "A 数据质量"
            why = ("保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性）"
                   if v == -80 else "h_ss 该档 ok=false（等时近似失真/小信号）")
            fail("hERG", f"Lei-{batch}", f"{len(cells)}孔:{'/'.join(cells[:6])}{'...' if len(cells)>6 else ''}",
                 "电压档", f"h_ss({v})", cat, why, "逐细胞重算（J1J2 JSON）")
    for v, n in tauh_missing_by_v.items():
        if n:
            fail("hERG", f"Lei-{batch}", f"{n}孔", "电压档", f"tau_rec({v})", "A 数据质量",
                 "DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值", "逐细胞重算（TAUH 逐孔 JSON）")

# ============ 2. Nav1.5 ============
# Nα-4 Tarasov single-channel / multi-channel patches
na4 = list(csv.DictReader(open(os.path.join(AM, "2026-09-16_Nα4_单通道负40单点锚定判决_结果.csv"), encoding="utf-8-sig")))
files_tau, files_late = set(), set()
for r in na4:
    mod, met = r["modality"], r["metric"]
    grp = {"mult-ch": "多通道膜片", "single-ch": "单通道膜片", "dKPQ": "ΔKPQ膜片"}[mod]
    if met == "tau_decay_ms":
        add("Nav1.5", "Tarasov2026-Dryad", grp, 22, r["file"], "tau_decay", -40, float(r["value"]), "ms")
        files_tau.add((mod, r["file"]))
    elif met == "po_peak":
        add("Nav1.5", "Tarasov2026-Dryad", grp, 22, r["file"], "Po_peak", -40, float(r["value"]), "")
    elif met == "late_pct":
        add("Nav1.5", "Tarasov2026-Dryad", grp, 22, r["file"], "late_pct", -40, float(r["value"]), "%")
        files_late.add((mod, r["file"]))
missing_late = sorted(f for f in files_tau if f[0] == "mult-ch" and f not in files_late)
for mod, f in missing_late:
    fail("Nav1.5", "Tarasov2026-Dryad", f, "膜片", "late_pct", "A 数据质量",
         "多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项）", "Nα4 结果 CSV 对账")
# Nα-3 Tarasov/Lei whole-cell foot
na3 = list(csv.DictReader(open(os.path.join(AM, "2026-09-16_Nα3_Nav15激活脚部可携带性判决_逐细胞.csv"), encoding="utf-8-sig")))
for r in na3:
    if r["ok"] == "True":
        if r["s_ref"]:
            add("Nav1.5", f"Nα3-{r['src']}", r["group"], 22, r["cell"], "s_ref", "", float(r["s_ref"]), "")
        if r["Vh_full"]:
            add("Nav1.5", f"Nα3-{r['src']}", r["group"], 22, r["cell"], "V_half_inact", "", float(r["Vh_full"]), "mV")
        if r["E_rev"]:
            add("Nav1.5", f"Nα3-{r['src']}", r["group"], 22, r["cell"], "E_rev", "", float(r["E_rev"]), "mV")
    else:
        cat = "A 数据质量" if "弦点低于" in r["reason"] else "C 方法边界"
        fail("Nav1.5", f"Nα3-{r['src']}", r["cell"], "细胞", "激活脚部 s_ref", cat,
             r["reason"], "Nα3 逐细胞 CSV")
# Nα-1 Lei whole-cell 3-cell tau_h(V)
na1 = list(csv.DictReader(open(os.path.join(AM, "2026-09-15_Nα1_Nav15失活快分量恒定性判决_结果.csv"), encoding="utf-8-sig")))
for r in na1:
    if r.get("skipped"):
        continue
    try:
        v = float(r["V"]); ta = float(r["tau_a"])
    except (ValueError, TypeError):
        continue
    add("Nav1.5", "Lei-Nav-HEK35", "全细胞", 35, r["cell"], "tau_h", v, ta, "ms")

# Nα-2 whole-cell tau_h(-40) (verdict card A4 table, three cells)
na2 = json.load(open(os.path.join(AM, "2026-09-15_Nα2_Nav15整通道一次验证_结果.json"), encoding="utf-8"))
a4 = na2["A_verdicts"]["A4"]["table"]
for vk, blk in a4.items():
    for i, tv in enumerate(blk.get("taus", [])):
        add("Nav1.5", "Nα2-Lei-Nav35", "全细胞", 35, f"wc{i+1}", "tau_h", float(vk), tv, "ms")

# ============ 3. IKs · Chan 2023 ============
ka1 = list(csv.DictReader(open(os.path.join(AM, "2026-09-15_Kα1_IKs激活去激活恒定性判决_逐文件表.csv"), encoding="utf-8-sig")))
n_pool = 0
for r in ka1:
    if r["in_pool"] != "1":
        continue
    n_pool += 1
    fam = "1s尾巴家族" if "low" in r["protocol"].lower() else r["protocol"]
    if r["Vh"]:
        add("IKs", "Chan2023-Zenodo8226585", fam, 22, r["file"], "V_half_act", "", float(r["Vh"]), "mV")
    if r["k"]:
        add("IKs", "Chan2023-Zenodo8226585", fam, 22, r["file"], "k_act", "", float(r["k"]), "mV")
    if r["tau_app_s"]:
        add("IKs", "Chan2023-Zenodo8226585", fam, 22, r["file"], "tau_app", -40, float(r["tau_app_s"]), "s")
fail("IKs", "Chan2023-Zenodo8226585", "2022_03_11_0005", "细胞", "τ_app(−40)", "C 方法边界",
     "边界可疑细胞：inst=0.50 贴排除线、amp=201 pA 池内最小、τ_app=0.020 s 贴拟合下界；判0按条文纳入、判2登记为敏感点",
     "Kα1 判词卡 §三.2")
fail("IKs", "Chan2023-Zenodo8226585", "4个ABF(未具名)", "文件", "原始读取", "A 数据质量",
     "336 个 ABF 中 4 个 float 格式不可读，剔除；另 193→163 重复导出去重", "Kα1 判词卡 §三.6")
fail("IKs", "Chan2023-Zenodo8226585", "+30/+40/+50/+60档≥1/3细胞", "电压档", "τ_act(V)", "B 协议限制",
     "4 s 激活协议对 IKs(τ 1.5–3 s) 过短，拟合贴 2.5 s 上界，按 caveat A 全部除名——判1 无票可投",
     "Kα1 判词卡 判1")

# ============ 4. CaV1.2 · Ren 2022 ============
ca1 = json.load(open(os.path.join(AM, "2026-09-15_Cα1_CaV12失活温度律与电荷载子判决_结果.json"), encoding="utf-8"))
for grp in ["Ca2", "Ba2"]:
    for f, rec in ca1["groups"].get(grp, {}).items():
        rs = rec.get("recs_summary", {})
        q = rec.get("q10", {})
        if grp == "Ca2" and q.get("Q10"):
            add("CaV1.2", "Ren2022-g3msb", "Ca2+ carrier", "31-37", f, "Q10_tau_slow", "", q["Q10"], "")
            if q.get("tau37_ms"):
                add("CaV1.2", "Ren2022-g3msb", "Ca2+ carrier", 37, f, "tau_slow_37C", "", q["tau37_ms"], "ms")
        n_sw, n_va = rs.get("n_sweeps", 0), rs.get("n_valid", 0)
        f23, f4 = rs.get("fail_G2G3", 0), rs.get("fail_G4", 0)
        if f23 or f4:
            fail("CaV1.2", "Ren2022-g3msb", f, "sweep", "QC门 G2G3/G4", "A 数据质量",
                 f"{n_sw} sweeps 中 G2G3 失败 {f23}、G4 失败 {f4}、有效 {n_va}", "Cα1 结果 JSON")
ca4 = json.load(open(os.path.join(AM, "2026-09-15_Cα4_斜坡IV温度律_结果.json"), encoding="utf-8"))
for f, rec in ca4["groups"].get("Ca2", {}).items():
    if rec.get("E_chord_med") is not None:
        add("CaV1.2", "Ren2022-g3msb", "Ca2+ carrier", "31-37", f, "E_chord", "", rec["E_chord_med"], "mV")

# ============ 5. registry: verdict-card-level known failures (not recomputed here) ============
fail("hERG", "Beattie2018-HEK9", "16704007", "细胞", "AP/sine 前向", "D 前向判决",
     "九细胞前向验证：sine 7/9、AP 8/9——007 为已知问题细胞（死 sweep/坏节段史，见病灶审计卡）", "四线总览 v5 §1")
fail("hERG", "Lei-37°C批", "全体80/105孔", "电压档", "h_ss(−80)", "B 协议限制",
     "等时简并电压档；37 °C 批 80 孔通过但中位 0.857 CV 0.15——数值仅供参考不入判决", "Lei Q10 判词卡 异常登记1")
fail("hERG", "Lei-37°C批", "83/105孔", "电压档", "h_ss(−100)", "B 协议限制",
     "高温下该档 ok=false 过多，n=22 低于门槛 → 数据不足登记", "Lei Q10 判词卡 异常登记2")
fail("hERG", "Lei-25°C批", "统计档", "方法", "C1 合成回收 −100/−60/−20", "C 方法边界",
     "等时近似在恢复/去激活竞速档 +10~30% 偏置，三档统计量作废不判决", "Lei211 判词卡 C1")
fail("Nav1.5", "Lei-Nav-HEK35", "3细胞中1细胞", "细胞", "C2 整迹验证", "D 前向判决",
     "C2 1/3，机制清楚（见 Nα2 判词卡）", "四线总览 v5 §1")
fail("Nav1.5", "Tarasov2026-Dryad", "池级", "参数", "τ_decay(−40)/Po_peak", "E 真实物理",
     "池内散布 CV 0.77/0.60 失控但判4 ΔKPQ 方向 13× 正确——散布为 modal gating 真实涨落，非失败",
     "Nα4 判词卡 判1/2/4")

# ============ 6. write long-table CSV ============
long_csv = os.path.join(RES, "逐细胞参数总表_四通道_2026-09-21.csv")
with open(long_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(LONG[0].keys()))
    w.writeheader(); w.rows = LONG
    for row in LONG:
        w.writerow(row)

fail_csv = os.path.join(RES, "失败细胞登记表_四通道_2026-09-21.csv")
with open(fail_csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(FAIL[0].keys()))
    w.writeheader()
    for row in FAIL:
        w.writerow(row)
print("long rows:", len(LONG), "| fail rows:", len(FAIL))

# ============ 7. violin figure ============
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

def L(channel, param, voltage=None, group=None, temp=None):
    out = []
    for r in LONG:
        if r["channel"] != channel or r["parameter"] != param or not r["ok"]:
            continue
        if voltage is not None and r["voltage_mV"] != voltage:
            continue
        if group is not None and r["group"] != group:
            continue
        if temp is not None and r["temp_C"] != temp:
            continue
        out.append(r["value"])
    return np.array(out)

def violin(ax, data_list, labels, color, logy=False, star=None, star_label="", tick_fs=6.5):
    pos = np.arange(1, len(data_list) + 1)
    ns = [0] * len(data_list)
    for i, d in enumerate(data_list):
        d = np.asarray(d)
        d = d[np.isfinite(d)]
        if len(d) >= 4:
            vp = ax.violinplot([d], positions=[pos[i]], widths=0.72,
                               showmeans=False, showmedians=False, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.28); b.set_edgecolor(color); b.set_linewidth(0.8)
        if len(d):
            xj = pos[i] + np.random.default_rng(7).uniform(-0.16, 0.16, len(d))
            ax.scatter(xj, d, s=3.5, c="k", alpha=0.45, linewidths=0, zorder=3)
            ax.hlines(np.median(d), pos[i]-0.28, pos[i]+0.28, colors=color, lw=1.6, zorder=4)
            ns[i] = len(d)
    ax.set_xticks(pos)
    ax.set_xticklabels(["{}\n({})".format(l, n) for l, n in zip(labels, ns)], fontsize=tick_fs)
    if logy:
        ax.set_yscale("log")
    if star is not None:
        ax.scatter([pos[-1]+0.0], [star[0]], marker="*", s=70, c="#B22222", zorder=5)
        ax.annotate(star_label, (pos[-1], star[0]), textcoords="offset points",
                    xytext=(6, 4), fontsize=6, color="#B22222")
    ax.spines[["top", "right"]].set_visible(False)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
np.random.seed(7)
TEMPS = [25, 27, 30, 33, 37]

# a: hERG E_rev × T
ax = axes[0, 0]
violin(ax, [L("hERG", "E_rev", temp=t) for t in TEMPS], ["%d" % t for t in TEMPS],
       "#4C72B0", star=(-92.9,), star_label="model62 fit −92.9", tick_fs=6.0)
ax.set_ylabel("per-cell $E_{rev}$ (mV)")
ax.set_xlabel("temperature (°C)")
ax.set_title("hERG · Lei CHO 670 wells", fontsize=7)

# b: hERG h_ss limb 25 °C
ax = axes[0, 1]
violin(ax, [L("hERG", "h_ss", voltage=v, temp=25) for v in [-140, -120, -100]],
       ["−140", "−120", "−100"], "#4C72B0")
ax.set_ylabel("per-cell $h_{ss}$")
ax.set_xlabel("voltage (mV)")
ax.set_title("hERG · $h_{ss}$ negative limb, 25 °C", fontsize=7)

# c: hERG τ_rec(−120) × T
ax = axes[0, 2]
violin(ax, [L("hERG", "tau_rec", voltage=-120, temp=t) for t in TEMPS],
       ["%d" % t for t in TEMPS], "#4C72B0", logy=True, star=(3.04,), star_label="Beattie HEK 3.04 ms", tick_fs=6.0)
ax.set_ylabel(r"per-cell $\tau_{rec}$(−120 mV) (ms)")
ax.set_xlabel("temperature (°C)")
ax.set_title("hERG · recovery τ travels: $Q_{10}$=2.84, $R^2$=0.991", fontsize=7)

# d: Nav1.5 τ(−40) by modality
ax = axes[1, 0]
d1 = L("Nav1.5", "tau_decay", group="多通道膜片")
d2 = L("Nav1.5", "tau_decay", group="单通道膜片")
d3 = np.array([r["value"] for r in LONG if r["dataset"]=="Nα2-Lei-Nav35" and r["parameter"]=="tau_h" and r["voltage_mV"]==-40.0])
violin(ax, [d1, d2, d3], ["multi-\nch", "single-\nch", "whole-\ncell"],
       "#DD8452", logy=True, tick_fs=6.0)
cv1 = np.std(d1)/np.mean(d1) if len(d1) else float("nan")
ax.set_title(f"Nav1.5 · τ(−40 mV): a distribution, not a constant (CV={cv1:.2f})", fontsize=7)
ax.set_ylabel("per-patch / per-cell τ (ms)")
ax.set_xlabel("recording modality")

# e: IKs V½ / k / τ_app
ax = axes[1, 1]
e1 = L("IKs", "V_half_act"); e2 = L("IKs", "k_act")
violin(ax, [e1, e2], ["$V_{1/2}$ act", "$k$ act"], "#55A868")
ax.set_ylabel("mV")
ax2 = ax.twinx()
e3 = L("IKs", "tau_app")
xj = 3 + np.random.default_rng(7).uniform(-0.16, 0.16, len(e3))
ax2.scatter(xj, e3, s=3.5, c="k", alpha=0.45, linewidths=0)
ax2.hlines(np.median(e3), 3-0.28, 3+0.28, colors="#55A868", lw=1.6)
ax2.set_ylabel(r"$\tau_{app}$(−40 mV) (s)", color="#55A868")
ax2.spines["top"].set_visible(False)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"$V_{{1/2}}$ act\nn={len(e1)}", f"$k$ act\nn={len(e2)}",
                    f"$\\tau_{{app}}$(-40)\nn={len(e3)}"], fontsize=6.5)
ax.set_title("IKs · Chan 2023 WT pool (22 °C)", fontsize=7)

# f: CaV1.2 E_chord / Q10(τ_slow)
ax = axes[1, 2]
f1 = L("CaV1.2", "E_chord")
violin(ax, [f1], ["$E_{chord}$"], "#8172B3")
ax.set_ylabel("mV")
ax2 = ax.twinx()
f2 = L("CaV1.2", "Q10_tau_slow")
xj = 2 + np.random.default_rng(7).uniform(-0.16, 0.16, len(f2))
ax2.scatter(xj, f2, s=3.5, c="k", alpha=0.45, linewidths=0)
ax2.hlines(np.median(f2), 2-0.28, 2+0.28, colors="#8172B3", lw=1.6)
ax2.set_ylabel(r"$Q_{10}(\tau_{slow})$", color="#8172B3")
ax2.spines["top"].set_visible(False)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"$E_{{chord}}$\nn={len(f1)}", f"$Q_{{10}}$($\\tau_{{slow}}$)\nn={len(f2)}"], fontsize=6.5)
ax.set_title("CaV1.2 · Ren 2022 (Ca2+ carrier)", fontsize=7)

for i, ax in enumerate(axes.flat):
    ax.text(-0.22, 1.04, "abcdef"[i], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
fig.tight_layout(w_pad=2.2, h_pad=1.6)
png = os.path.join(FIGD, "EDFig_percell_violin_v01.png")
pdf = os.path.join(FIGD, "EDFig_percell_violin_v01.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight")
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print('PNG pixels:', Image.open(png).size)
fig.savefig(pdf, bbox_inches="tight")
print("saved:", png)

# ============ 8. failure registry MD ============
CATS = ["A 数据质量", "B 协议限制", "C 方法边界", "D 前向判决", "E 真实物理"]
md = ["# 失败细胞登记表 · 四通道（2026-09-21）",
      "",
      "**口径**：逐细胞/逐电压档/逐 sweep 三级，全部来自封卷判词卡对应结果文件的重算或直接转录；类别定义——A 数据质量（记录本身异常或信噪不足）、B 协议限制（协议窗/等时简并/高温窗限）、C 方法边界（拟合贴界/等时近似偏置）、D 前向判决（模型预测不过，非数据问题）、E 真实物理（登记为非失败）。",
      "",
      f"**总计 {len(FAIL)} 条登记**（含批次聚合行；Lei 线逐电压档聚合行为每批每档一行，该档涉及细胞数见 detail 或 cell_or_file 列）。",
      ""]
for cat in CATS:
    rows = [r for r in FAIL if r["category"] == cat]
    if not rows:
        continue
    md.append(f"## {cat}（{len(rows)} 条）")
    md.append("")
    md.append("| 通道 | 数据集 | 细胞/文件 | 范围 | 失败项 | 说明 | 来源 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['channel']} | {r['dataset']} | {r['cell_or_file']} | {r['scope']} | {r['failed_item']} | {r['detail']} | {r['source']} |")
    md.append("")
md_path = os.path.join(RES, "失败细胞登记表_四通道_2026-09-21.md")
open(md_path, "w", encoding="utf-8").write("\n".join(md))
print("saved:", md_path)
