# -*- coding: utf-8 -*-
# 全族统计分拆：τ 紧 vs V1/2 松（Channelpedia Kv 族）
# 判据（跑前钉死）：
#   主图指标（每 通道×温度）：
#     tau_disp  = 激活 τ 的跨细胞 log10-SD 中位数（取 V∈[0,+40] 窗口，无量纲）
#     mid_disp  = 激活 V1/2 的跨细胞 SD / 该通道 Boltzmann 斜率 k 的中位数（无量纲）
#   定律支持：全体 通道×温度 中 mid_disp/tau_disp > 1 的比例 ≥ 80%，且中位比 > 2
#   温度律复核：同通道 15 vs 35°C 的 τ 中位比换算 Q10（20°C 跨度）
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "features" / "features_all.csv")

MIN_CELLS = 15           # 每 通道×温度 至少细胞数
TAU_WIN = (0, 40)        # τ_act 统计电压窗
rows = []
for (ch, T), g in df.groupby(["channel", "temp_nominal"]):
    cells = g.cell_id.nunique()
    if cells < MIN_CELLS:
        continue
    # τ_act 跨细胞 log10-SD（逐电压，再取中位）
    ta = g[(g.parameter == "tau_act") & (g.voltage_mV.between(*TAU_WIN)) & (g.value > 0) & (g.r2 >= 0.7)]
    logsd_by_v = ta.groupby("voltage_mV")["value"].apply(lambda s: np.log10(s).std())
    logsd_by_v = logsd_by_v[ta.groupby("voltage_mV")["cell_id"].nunique() >= MIN_CELLS]
    tau_disp = float(logsd_by_v.median()) if len(logsd_by_v) else np.nan
    # V1/2_act 跨细胞 SD 与 k
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)]["value"]
    kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)]["value"]
    mid_sd = float(vh.std()) if len(vh) >= MIN_CELLS else np.nan
    k_med = float(kk.median()) if len(kk) >= MIN_CELLS else np.nan
    mid_disp = mid_sd / k_med if np.isfinite(mid_sd) and np.isfinite(k_med) else np.nan
    # 失活中点
    vhi = g[(g.parameter == "Vhalf_inact") & (g.r2 >= 0.8)]["value"]
    vhi_sd = float(vhi.std()) if len(vhi) >= MIN_CELLS else np.nan
    rows.append(dict(channel=ch, temp_C=T, n_cells=cells,
                     tau_logSD=tau_disp, n_voltages=int(len(logsd_by_v)),
                     Vhalf_act_SD_mV=mid_sd, k_act_median=k_med, mid_disp=mid_disp,
                     Vhalf_inact_SD_mV=vhi_sd))
s = pd.DataFrame(rows)
s["ratio"] = s.mid_disp / s.tau_logSD
s.to_csv(ROOT / "features" / "family_split.csv", index=False)

# 温度律复核：τ_act 在 +20..+40 的中位 τ，15 vs 35°C → Q10
q10_rows = []
for ch, g in df.groupby("channel"):
    for V in [20, 30, 40]:
        t15 = g[(g.temp_nominal == 15) & (g.parameter == "tau_act") & (g.voltage_mV == V) & (g.value > 0)]["value"].median()
        t35 = g[(g.temp_nominal == 35) & (g.parameter == "tau_act") & (g.voltage_mV == V) & (g.value > 0)]["value"].median()
        if np.isfinite(t15) and np.isfinite(t35) and t35 > 0:
            q10_rows.append(dict(channel=ch, voltage_mV=V,
                                 tau15=t15, tau35=t35, Q10=float((t15/t35)**(10/20))))
q = pd.DataFrame(q10_rows)
q.to_csv(ROOT / "features" / "family_q10.csv", index=False)

ok = s.dropna(subset=["ratio"])
verdict = dict(
    n_channel_temp=len(s), n_with_ratio=int(len(ok)),
    frac_ratio_gt1=float((ok.ratio > 1).mean()) if len(ok) else None,
    median_ratio=float(ok.ratio.median()) if len(ok) else None,
    law_supported=bool(len(ok) and (ok.ratio > 1).mean() >= 0.8 and ok.ratio.median() > 2),
    q10_median=float(q.Q10.median()) if len(q) else None,
)
with open(ROOT / "features" / "family_verdict.json", "w", encoding="utf-8") as f:
    json.dump(verdict, f, ensure_ascii=False, indent=2)

# 头条图
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
order = ok.groupby("channel")["ratio"].median().sort_values(ascending=False).index
dat = [ok[ok.channel == c].ratio.values for c in order]
ax.boxplot(dat, labels=order, showfliers=False)
ax.axhline(1, color="r", ls="--", lw=1)
ax.set_ylabel("mid_disp / tau_disp"); ax.set_title("各通道：中点散布 / 时间常数散布（>1 支持定律）")
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
ax = axes[1]
ax.scatter(ok.tau_logSD, ok.mid_disp, c=ok.temp_C, cmap="coolwarm", alpha=0.7)
mx = max(ok.tau_logSD.max(), ok.mid_disp.max())
ax.plot([0, mx], [0, mx], "r--", lw=1)
ax.set_xlabel("τ 跨细胞 log10-SD（无量纲）"); ax.set_ylabel("V½ 跨细胞 SD / k（无量纲）")
ax.set_title("每点 = 通道×温度（色=温度）；红线上方 = V½ 更散")
fig.tight_layout()
fig.savefig(ROOT / "features" / "headline_family.png", dpi=160, bbox_inches="tight")
print(json.dumps(verdict, ensure_ascii=False, indent=2))
print(s.sort_values("ratio", ascending=False).to_string(index=False))
